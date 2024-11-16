

# 数据导入
import pandas as pd
import numpy as np
data = pd.read_csv('data.csv')
data.hist
print(data.describe())
# pd.crosstab(data['FILE_TYPE'],data['y'],rownames = ['FILE_TYPE'])
print(data['y'].value_counts())


# 初始数据分析
corr_matrix = data.corr()
print(corr_matrix)
x = data.iloc[:, :-1]    # 切片，得到输入x
y = data.iloc[:, -1]     # 切片，得到标签y



# 两种不同的采样方法
from sklearn.model_selection import train_test_split
# 方法1：使用SMOTE方法进行过抽样处理
from imblearn.over_sampling import SMOTE     # 过抽样处理库SMOTE942
model_smote = SMOTE(random_state=0)          # 建立SMOTE模型对象 随机过采样
x_smote_resampled, y_smote_resampled = model_smote.fit_resample(x,y) 
                                             # 输入数据并作过抽样处理
smote_resampled = pd.concat([x_smote_resampled, y_smote_resampled],axis=1) 
                                             # 按列合并数据框
groupby_data_smote = smote_resampled.groupby('y').count()    
print (groupby_data_smote)     # 经过SMOTE处理后的数据中 “y” 的分布
x1_train, x1_test, y1_train, y1_test = train_test_split(x_smote_resampled, y_smote_resampled, test_size =0.3)

# 方法2：使用RandomUnderSampler方法进行欠抽样处理
from imblearn.under_sampling import RandomUnderSampler 
                                             # 欠抽样处理库RandomUnderSampler
model_RandomUnderSampler = RandomUnderSampler()     # 随机欠采样（下采样）
x_RandomUnderSampler_resampled, y_RandomUnderSampler_resampled =model_RandomUnderSampler.fit_resample(x,y) 
                                             # 输入数据并作欠抽样处理
RandomUnderSampler_resampled =pd.concat([x_RandomUnderSampler_resampled, y_RandomUnderSampler_resampled], axis= 1) 
                                             # 按列合并数据框
groupby_data_RandomUnderSampler =RandomUnderSampler_resampled.groupby('y').count()
print (groupby_data_RandomUnderSampler) 
                           # 经过RandomUnderSampler处理后的数据中 “y” 的分布
x2_train, x2_test, y2_train, y2_test = train_test_split(x_RandomUnderSampler_resampled, y_RandomUnderSampler_resampled, test_size =0.3)



print(smote_resampled)
print(RandomUnderSampler_resampled)




# 2.利用过采样的数据得到训练集和测试集
def acu_curve(y,prob): #画出roc曲线
    fpr1,tpr1,threshold1=roc_curve(y,prob)
    roc_auc1=auc(fpr1,tpr1)
    plt.figure()
    lw=2
    plt.plot(fpr1,tpr1,color="darkorange",lw=lw,label="ROC curve(area=%0.2f)"%roc_auc1)
    plt.plot([0,1],[0,1],color="navy",lw=lw,linestyle="--")
    plt.xlim([0.0,1.0])
    plt.ylim([0.0,1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC")
    plt.legend(loc="lower right")
    plt.show()

def lift_curve_(result): #画出lift曲线
    result.columns = ['target','proba']
    # 'target': real labels of the data
    # 'proba': probability predictions for such data
    result_ = result.copy()
    proba_copy = result.proba.copy()
    for i in range(10):
        point1 = scoreatpercentile(result_.proba, i*(100/10))
        point2 = scoreatpercentile(result_.proba, (i+1)*(100/10))
        proba_copy[(result_.proba >= point1) & (result_.proba <= point2)] = ((i+1))
    result_['grade'] = proba_copy
    df_gain = result_.groupby(by=['grade'], sort=True).sum()/(len(result)/10)*100
    plt.plot(df_gain['target'], color='red')
    for xy in zip(df_gain['target'].reset_index().values):
        plt.annotate("%s" % round(xy[0][1],2), xy=xy[0], xytext=(-20, 10), textcoords='offset points')  
    plt.plot(df_gain.index,[sum(result['target'])*100.0/len(result['target'])]*len(df_gain.index), color='blue')
    plt.title('Lift Curve')
    plt.xlabel('分类')
    plt.ylabel('Bad Rate (%)')
    plt.xticks([1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0])
    fig = plt.gcf()
    fig.set_size_inches(10,8)
    plt.savefig("train.png")
    plt.show()


#过采样核svm
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.metrics import roc_curve,auc
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import scoreatpercentile
from scipy.stats import ks_2samp
model1=SVC(kernel='poly',probability=True,C=1)
model1.fit(x1_train,y1_train)


#高斯核0.716
#多项式核0.73
#sigmoid0.63
#线性0.70
    
#计算准确值与ks值
def ks_calc_auc(x_test,y_test):
     '''
     功能: 计算KS值，输出对应分割点和累计分布函数曲线
     '''
     decision_scores = model1.decision_function(x_test)
     fpr,tpr,thresholds = roc_curve(y_test,decision_scores)
     ks = max(tpr-fpr)
     return ks
y1_true = pd.Series(y1_test)
pred1 = model1.predict(x1_test)
ks1 = ks_calc_auc(x1_test,y1_test)
print("precision rate_1: " + str(accuracy_score(y1_test, pred1)))
print("ks_1:" + str(ks1))

#绘制roc曲线
y1_score = model1.decision_function(x1_test)
print('AUC_1 = ' + str(roc_auc_score(y1_test,y1_score)))
acu_curve(y1_test,y1_score)

#绘制lift曲线
proba1 = model1.predict_proba(x1_test)
list1_ = list(zip(y1_test))
y1_test_ = pd.DataFrame(list1_,columns = ['y1_test'])
target1 = y1_test_['y1_test']

predict_proba1 = []
for i in range(len(proba1)):
    pi = proba1[i][1]
    predict_proba1.append(pi)
proba1 = np.array(predict_proba1)

list1_ = list(zip(target1,proba1))
result1 = pd.DataFrame(list1_ ,columns = ['target1','proba1'])
lift_curve_(result1)


#欠采样核svm
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
model2 = SVC(kernel='poly',probability=True)
model2.fit(x2_train,y2_train)
pred = model2.predict(x2_test)
y2_score=model1.decision_function(x2_test)
#高斯核0.669
#多项式核0.669
#sigmoid0.43
#线性0.665


#计算ks值
def ks_calc_auc(x_test,y_test):
     '''
     功能: 计算KS值，输出对应分割点和累计分布函数曲线图
     '''
     decision_scores = model2.decision_function(x_test)
     fpr,tpr,thresholds = roc_curve(y_test,decision_scores)
     ks = max(tpr-fpr)
     return ks
y2_true = pd.Series(y2_test)
pred2 = model2.predict(x2_test)
ks2 = ks_calc_auc(x2_test,y2_test)
print("precision rate_2: " + str(accuracy_score(y2_test, pred2)))
print("ks_2:" + str(ks2))

#绘制roc曲线
acu_curve(y2_test,y2_score)
print('AUC_2 = ' + str(roc_auc_score(y2_test,y2_score)))

# 绘制lift曲线
proba2 = model2.predict_proba(x2_test)
list2_ = list(zip(y2_test))
y2_test_ = pd.DataFrame(list2_,columns = ['y2_test'])
target2 = y2_test_['y2_test']

predict_proba2 = []
for i in range(len(proba2)):
    pi = proba2[i][1]
    predict_proba2.append(pi)
proba2 = np.array(predict_proba2)

list2_ = list(zip(target2,proba2))
result2 = pd.DataFrame(list2_ ,columns = ['target2','proba2'])
lift_curve_(result2)