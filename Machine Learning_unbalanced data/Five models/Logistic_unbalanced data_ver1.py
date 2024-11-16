# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 20:54:03 2022

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
# 方法1：使用SMOTE方法进行过抽样处理
from imblearn.over_sampling import SMOTE     # 过抽样处理库SMOTE942
model_smote = SMOTE(random_state=0)          # 建立SMOTE模型对象 随机过采样
x_smote_resampled, y_smote_resampled = model_smote.fit_resample(x,y) 
                                             # 输入数据并作过抽样处理
smote_resampled = pd.concat([x_smote_resampled, y_smote_resampled],axis=1) 
                                             # 按列合并数据框
groupby_data_smote = smote_resampled.groupby('y').count()    
print (groupby_data_smote)     # 经过SMOTE处理后的数据中 “y” 的分布

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

print(smote_resampled)
print(RandomUnderSampler_resampled)


# 三种不同的训练模型
# 模型1：Logistic模型
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression()
from scipy.stats import ks_2samp
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

# 以logistic regression模型，定义计算ks值的函数
def ks_calc_auc(x_test,y_test):
     '''
     功能: 计算KS值，输出对应分割点和累计分布函数曲线图
     '''
     decision_scores = log_reg.decision_function(x_test)
     fpr,tpr,thresholds = roc_curve(y_test,decision_scores)
     ks = max(tpr-fpr)
     return ks

# 定义绘制lift曲线的函数
from scipy.stats import scoreatpercentile
def lift_curve_(result):
    result.columns = ['target','proba']
    # 'target': real labels of the data
    # 'proba': probability for "y_pred = 1"
    result_ = result.copy()
    proba_copy = result.proba.copy()
    for i in range(10):
        point1 = scoreatpercentile(result_.proba, i*(100/10))
        point2 = scoreatpercentile(result_.proba, (i+1)*(100/10))
        proba_copy[(result_.proba >= point1) & (result_.proba <= point2)] = ((i+1))
    result_['grade'] = proba_copy
    number1 =result_.groupby(by=['grade'], sort=True).size()
    for i in range(len(result_)):
        if result_['target'][i] != 1:
            result_.drop(i,axis = 0,inplace = True)
    number2 = result_.groupby(by=['grade'], sort=True).size() 
    df_gain = (number2 / number1)*100
    plt.plot(df_gain, color='red')
    for xy in zip(df_gain.reset_index().values):
        plt.annotate("%s" % round(xy[0][1],2), xy=xy[0], 
                     xytext=(-20, 10), textcoords='offset points')  
    plt.title('Lift Curve')
    plt.xlabel('分类')
    plt.ylabel('Bad Rate (%)')
    plt.xticks([1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0])
    fig = plt.gcf()
    fig.set_size_inches(10,8)
    plt.savefig("train.png")
    plt.show()


# 1.如果不对类别不平衡问题进行处理
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size =0.3)
log_reg.fit(x, y)

    # 找出混淆矩阵
y_pred = log_reg.predict(x_test)
y_true = pd.Series(y_test)
print(pd.crosstab( y_true , y_pred))

    # 找出准确率和ks值
ks = ks_calc_auc(x_test,y_test)
print("precision rate: " + str(log_reg.score(x_test, y_test)))
a = pd.crosstab( y_true , y_pred)
print("1_precision rate:" + str(a[1][1]/(a[0][1]+a[1][1])))
print("ks: " + str(ks))

    
    # 绘制AUC曲线，并求出面积
decision_scores = log_reg.decision_function(x_test)
print('AUC = ' + str(roc_auc_score(y_test,decision_scores)))
fprs, tprs, thresholds = roc_curve(y_test,decision_scores)
plt.plot(fprs,tprs)
plt.xlabel('fpr')
plt.ylabel('tpr')
plt.show()

    # 绘制lift曲线
proba = log_reg.predict_proba(x_test)
list_ = list(zip(y_test))
y_test_ = pd.DataFrame(list_,columns = ['y_test'])
target = y_test_['y_test']

predict_proba = []
for i in range(len(proba)):
    pi = proba[i][1]
    predict_proba.append(pi)
proba = np.array(predict_proba)

list_ = list(zip(target,proba))
result = pd.DataFrame(list_ ,columns = ['target','proba'])
lift_curve_(result)



# 2.利用过采样的数据得到训练集和测试集
x1_train, x1_test, y1_train, y1_test = train_test_split(x_smote_resampled, y_smote_resampled, test_size =0.3)
log_reg.fit(x_smote_resampled, y_smote_resampled)

    # 找出混淆矩阵
y1_pred = log_reg.predict(x1_test)
y1_true = pd.Series(y1_test)
print(pd.crosstab( y1_true , y1_pred))

    # 找出准确率和ks值
ks1 = ks_calc_auc(x1_test,y1_test)
print("precision rate_1: " + str(log_reg.score(x1_test, y1_test)))
a = pd.crosstab( y1_true , y1_pred)
print("1_precision rate_1:" + str(a[1][1]/(a[0][1]+a[1][1])))
print("ks_1:" + str(ks1))
    
    # 绘制AUC曲线，并求出面积
decision_scores1 = log_reg.decision_function(x1_test)
print('AUC_1 = ' + str(roc_auc_score(y1_test,decision_scores1)))
fprs1, tprs1, thresholds = roc_curve(y1_test,decision_scores1)
plt.plot(fprs1,tprs1)
plt.xlabel('fpr1')
plt.ylabel('tpr1')
plt.show()

    # 绘制lift曲线
proba1 = log_reg.predict_proba(x1_test)
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


# 3.利用欠采样的数据得到训练集和测试集
x2_train, x2_test, y2_train, y2_test = train_test_split(x_RandomUnderSampler_resampled, y_RandomUnderSampler_resampled, test_size =0.3)
log_reg.fit(x_smote_resampled, y_smote_resampled)

    # 找出混淆矩阵
y2_pred = log_reg.predict(x2_test)
y2_true = pd.Series(y2_test)
print(pd.crosstab( y2_true , y2_pred ))

    # 找出准确率和ks值
ks2 = ks_calc_auc(x2_test,y2_test)
print("precision rate_2: " + str(log_reg.score(x2_test, y2_test)))
a = pd.crosstab( y2_true , y2_pred)
print("1_precision rate_2:" + str(a[1][1]/(a[0][1]+a[1][1])))
print("ks_2:" + str(ks2))
    
    # 绘制AUC曲线，并求出面积
decision_scores2 = log_reg.decision_function(x2_test)
print('AUC_2 = ' + str(roc_auc_score(y2_test,decision_scores2)))
fprs2, tprs2, thresholds = roc_curve(y2_test,decision_scores2)
plt.plot(fprs2,tprs2)
plt.xlabel('fpr1')
plt.ylabel('tpr1')
plt.show()

    # 绘制lift曲线
proba2 = log_reg.predict_proba(x2_test)
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