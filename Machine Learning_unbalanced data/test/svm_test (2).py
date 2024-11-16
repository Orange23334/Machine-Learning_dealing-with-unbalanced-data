# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 13:54:15 2022

@author: Jiacheng
"""

# -*- coding: utf-8 -*-
from numpy import argmax

"""
Created on Tue Apr 19 19:46:07 2022

@author: Jiacheng
"""


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
    
def ks_calc_auc(x_test,y_test,model):
     #功能: 计算KS值，输出对应分割点和累计分布函数曲线
     decision_scores = model.decision_function(x_test)
     fpr,tpr,thresholds = roc_curve(y_test,decision_scores)
     ks = max(tpr-fpr)
     return ks
 
def lift_curve_(result): #画出lift曲线
    result.columns = ['target','proba']
    # 'target': real labels of the data
    # 'proba': probability predictions for bad account
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


from sklearn.metrics import accuracy_score, recall_score
from sklearn.svm import SVC
from sklearn.metrics import roc_curve,auc
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import scoreatpercentile
from scipy.stats import ks_2samp
from sklearn.metrics import roc_auc_score

#过采样核svm
model1=SVC(kernel='rbf',probability=True,class_weight='balanced')
model1.fit(x1_train,y1_train)
y1_score=model1.decision_function(x1_test)

#高斯核0.716
#多项式核0.73
#sigmoid0.63
#线性0.70
    
#计算准确值与ks值
y1_true = pd.Series(y1_test)  #测试集的真实标签
pred1 = model1.predict(x1_test) #测试集的预测标签

ks1 = ks_calc_auc(x1_test,y1_test,model1)
print(pd.crosstab( y1_true , pred1))
print("precision rate_1: " + str(accuracy_score(y1_test, pred1)))
a = pd.crosstab( y1_true , pred1)
print("1_precision rate_1:" + str(a[1][1]/(a[0][1]+a[1][1])))
print("ks_1:" + str(ks1))
#绘制roc曲线
y1_score = model1.decision_function(x1_test)
print('AUC = ' + str(roc_auc_score(y1_test,y1_score)))
acu_curve(y1_test,y1_score)
#绘制lift曲线
proba1 = model1.predict_proba(x1_test)  #对测试集的预测概率
list1_ = list(zip(y1_test))
y1_test_ = pd.DataFrame(list1_,columns = ['y1_test'])
target1 = y1_test_['y1_test']


predict_proba1 = []
for i in range(len(proba1)):
    pi = proba1[i][1]  #判断为坏客户的概率
    predict_proba1.append(pi)
proba1 = np.array(predict_proba1)

list1_ = list(zip(target1,proba1))
result1 = pd.DataFrame(list1_ ,columns = ['target1','proba1'])

lift_curve_(result1)




# 测试
test_data = pd.read_csv('test.csv')
x_test = test_data.iloc[:, :-1]
y_test = test_data.iloc[:, -1]  

#计算准确值与ks值
y_true = pd.Series(y_test)  #测试集的真实标签
pred = model1.predict(x_test) #测试集的预测标签

ks = ks_calc_auc(x_test,y_test,model1)
print(pd.crosstab( y_true , pred))
print("precision rate_1: " + str(accuracy_score(y_test, pred)))
a = pd.crosstab( y_true , pred)
print("1_precision rate_1:" + str(a[1][1]/(a[0][1]+a[1][1])))
print("ks_1:" + str(ks1))

#绘制roc曲线
y_pred = model1.predict(x_test)
y_true = pd.Series(y_test)
decision_scores = model1.decision_function(x_test)
print('AUC = ' + str(roc_auc_score(y_test,decision_scores)))
fprs, tprs, thresholds = roc_curve(y_test,decision_scores)
J = tprs - fprs
idx = argmax(J)
threshold = thresholds[idx]
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
    # 阈值移动
y_pred1 = model1.predict_proba(x_test)
yy = y_pred1[:, 1]
y_pred2 = (yy > threshold)
accuracy2 = accuracy_score(y_test, y_pred2)
recall2 = recall_score(y_test, y_pred2)
print(threshold)
print("accuracy = "+str(accuracy))
print("accuracy2 = "+str(accuracy2))
print("recall = "+str(recall))
print("recall2 = "+str(recall2))

#绘制lift曲线
proba = model1.predict_proba(x_test)  #对测试集的预测概率
list_ = list(zip(y_test))
y_test_ = pd.DataFrame(list_,columns = ['y_test'])
target = y_test_['y_test']


predict_proba = []
for i in range(len(proba)):
    pi = proba[i][1]  #判断为坏客户的概率
    predict_proba.append(pi)
proba = np.array(predict_proba)

list_ = list(zip(target,proba))
result = pd.DataFrame(list_ ,columns = ['target','proba'])

lift_curve_(result)





























