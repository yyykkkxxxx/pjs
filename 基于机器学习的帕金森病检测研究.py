#!/usr/bin/env python
# coding: utf-8

# In[1]:


#导入常用的库
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['font.sans-serif']=['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False    # 用来正常显示负号
pd.options.display.max_columns = None #显示完整打印数据
pd.options.display.max_rows = None
plt.rcParams['figure.dpi'] = 150 # 修改图片分辨率
import numpy as np
import warnings
warnings.filterwarnings("ignore") # 忽略警告


# In[2]:


# 读取数据
data = pd.read_csv('parkinsons.data')
data.head() # 打印数据前五行


# ## 1 数据探索与可视化分析

# In[3]:


# 查看数据形状
data.shape


# In[4]:


# 查看数据类型分布
data.dtypes


# In[5]:


# 数据描述
data.describe()


# In[6]:


# 检查数据每列缺失值个数，发现不存在缺失值
data.isnull().sum()


# In[7]:


# 可视化数据缺失情况
import missingno as msno
plt.figure()
p=msno.bar(data,color='g')
plt.title('可视化数据缺失情况',fontsize=20)


# In[8]:


# 统计标签分布情况,发现数据比较平衡
print(data['status'].value_counts())
plt.figure()
plt.pie(data['status'].value_counts(),labels=data['status'].value_counts().index,autopct='%1.2f%%',explode=(0.02,0))
plt.title('标签分布图')
plt.show()


# In[9]:


# 核密度图，MDVP:Fo(Hz)
plt.figure()
def kdeplot(feature,xlabel):
    plt.title("KDE for {0}".format(feature))
    ax0 = sns.kdeplot(data[data['status'] == 0][feature].dropna(), color= 'navy', label= 'status：0', shade='True')
    ax1 = sns.kdeplot(data[data['status'] == 1][feature].dropna(), color= 'orange', label= 'status：1',shade='True')
    plt.xlabel(xlabel)
    #设置字体大小
    plt.rcParams.update({'font.size': 16})
    plt.legend(fontsize=10)
kdeplot('MDVP:Fo(Hz)','MDVP:Fo(Hz)')
plt.show()


# In[10]:


# 核密度图，MDVP:Fhi(Hz)
plt.figure()
def kdeplot(feature,xlabel):
    plt.title("KDE for {0}".format(feature))
    ax0 = sns.kdeplot(data[data['status'] == 0][feature].dropna(), color= 'navy', label= 'status：0', shade='True')
    ax1 = sns.kdeplot(data[data['status'] == 1][feature].dropna(), color= 'orange', label= 'status：1',shade='True')
    plt.xlabel(xlabel)
    #设置字体大小
    plt.rcParams.update({'font.size': 16})
    plt.legend(fontsize=10)
kdeplot('MDVP:Fhi(Hz)','MDVP:Fhi(Hz)')
plt.show()


# In[11]:


# 核密度图，MDVP:Flo(Hz)
plt.figure()
def kdeplot(feature,xlabel):
    plt.title("KDE for {0}".format(feature))
    ax0 = sns.kdeplot(data[data['status'] == 0][feature].dropna(), color= 'navy', label= 'status：0', shade='True')
    ax1 = sns.kdeplot(data[data['status'] == 1][feature].dropna(), color= 'orange', label= 'status：1',shade='True')
    plt.xlabel(xlabel)
    #设置字体大小
    plt.rcParams.update({'font.size': 16})
    plt.legend(fontsize=10)
kdeplot('MDVP:Flo(Hz)','MDVP:Flo(Hz)')
plt.show()


# In[12]:


# 核密度图，MDVP:Jitter(%)
plt.figure()
def kdeplot(feature,xlabel):
    plt.title("KDE for {0}".format(feature))
    ax0 = sns.kdeplot(data[data['status'] == 0][feature].dropna(), color= 'navy', label= 'status：0', shade='True')
    ax1 = sns.kdeplot(data[data['status'] == 1][feature].dropna(), color= 'orange', label= 'status：1',shade='True')
    plt.xlabel(xlabel)
    #设置字体大小
    plt.rcParams.update({'font.size': 16})
    plt.legend(fontsize=10)
kdeplot('MDVP:Jitter(%)','MDVP:Jitter(%)')
plt.show()


# In[13]:


# 绘制D2和PPE散点图
plt.figure()
plt.scatter(x=data['D2'], y=data['PPE'],alpha=0.7)
plt.xlabel('D2', fontsize=15)
plt.ylabel('PPE', fontsize=15)
plt.show()


# In[14]:


# 绘制spread2和spread1散点图
plt.figure()
plt.scatter(x=data['spread2'], y=data['spread1'],alpha=0.7)
plt.xlabel('spread2', fontsize=15)
plt.ylabel('spread1', fontsize=15)
plt.show()


# In[15]:


# 绘制DFA和RPDE散点图
plt.figure()
plt.scatter(x=data['DFA'], y=data['RPDE'],alpha=0.7)
plt.xlabel('DFA', fontsize=15)
plt.ylabel('RPDE', fontsize=15)
plt.show()


# In[16]:


# 绘制HNR和NHR散点图
plt.figure()
plt.scatter(x=data['HNR'], y=data['NHR'],alpha=0.7)
plt.xlabel('HNR', fontsize=15)
plt.ylabel('NHR', fontsize=15)
plt.show()


# ## 2 数据预处理

# In[17]:


# 删除建模无关列
data = data.drop('name',axis=1)
data.head()


# In[18]:


# 画热力图，数值为两个变量之间的相关系数,从图中可以看出每个变量之间的相关性（正相关还是负相关），以及相关性大小。
plt.figure(figsize=(20,16))
import seaborn as sns
cov = data.corr().round(2)
sns.heatmap(cov,annot=True,linewidths=0.1,vmin=-1,)
plt.title('相关系数热力图',size=20)


# In[19]:


# 分离特征和标签
x = data.drop(['status'],axis=1)
y = data['status']


# In[20]:


# 数据分割
# 7比3划分训练集，测试集，设置随机种子random_state=2022，保证实验能够复现
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, 
                                                    y,
                                                    test_size=0.3,
                                                    stratify=y,
                                                    random_state=2022)


# In[21]:


# 特征选择
# 基于xgboost算法的特征重要性评分
import xgboost as xgb
model_XGB=xgb.XGBClassifier(verbosity=0,
                            max_depth=6,
                            learning_rate=0.3,
                            n_estimators=100,
                            random_state=2022)
model_XGB.fit(x_train,y_train)
data_after1 = pd.DataFrame(model_XGB.feature_importances_, 
                          columns=['importance'])
data_after2 = pd.DataFrame(model_XGB.feature_importances_,
                          index=x_train.columns, columns=['特征重要性'])
data_after2 = data_after2.sort_values(by='特征重要性',ascending=False)


# In[22]:


# 对基于xgboost算法的特征重要性评分的特征进行排序，升序为了画图，后面有降序的
aa = pd.DataFrame(x_train.columns,columns=['feature'])
feature = pd.concat([aa,data_after1],axis=1)
features_import = feature.sort_values(by='importance',ascending=True)
features_import


# In[23]:


# 特征重要性评分可视化
plt.figure(figsize=(15,8))
plt.barh(features_import['feature'], features_import['importance'], height=0.7, 
         color='#008792', 
         edgecolor='#005344') # 更多颜色可参见颜色大全
plt.xlabel('feature importance',fontsize=20) # x 轴
plt.ylabel('features',fontsize=20) # y轴
plt.title('Feature Importances',fontsize=20) # 标题
for a,b in zip( features_import['importance'],features_import['feature']): #
    plt.text(a+0.0005, b,'%.3f'%float(a),verticalalignment='center') # 可以修改保留小数位数
plt.show()


# In[24]:


# 降序
features_import1 = feature.sort_values(by='importance',ascending=False)
features_import1


# In[25]:


# 选择xgb算法的特征重要性评分前10的特征进行建模，阈值0.03
x_train = x_train.loc[:,data_after2.index[:10]]
x_test = x_test.loc[:,data_after2.index[:10]]


# In[26]:


x_train.head()


# In[27]:


x_test.head()


# ## 3 构建预测模型

# In[28]:


# 定义评价指标
from sklearn.metrics import precision_score, recall_score, f1_score ,roc_curve, auc,confusion_matrix ,accuracy_score,roc_auc_score
def roc_curve_and_score(y_test, pred_proba):
    roc_auc = roc_auc_score(y_test.ravel(), pred_proba.ravel())
    return roc_auc
def try_different_method(model):
    model.fit(x_train,y_train)
    yuce = model.predict(x_test) 
    print('测试集')
    precision = precision_score(y_test, yuce)
    recall = recall_score(y_test, yuce)
    f1score = f1_score(y_test, yuce)
    accuracy=accuracy_score(y_test, yuce)
    auc = roc_curve_and_score(y_test, model.predict_proba(x_test)[:, 1])
    print("AUC:", auc)
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("f1_score：", f1score)


# In[29]:


# 建立不同的模型
# 导入需要的包
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


# 随机森林
model_RF = RandomForestClassifier(random_state=0,
                                 max_depth=6
                                 )

model_NB  = GaussianNB()
# 逻辑回归
model_LR = LogisticRegression(random_state=42)

# 输出测试集评价指标
print('随机森林模型评分如下：')
try_different_method(model_RF)
print('贝叶斯模型评分如下：')
try_different_method(model_NB)
print('逻辑回归模型评分如下：')
try_different_method(model_LR)


# In[30]:


# RF混淆矩阵
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
plt.figure()
y_pred1 = model_RF.predict(x_test)
matrix = confusion_matrix(y_test, y_pred1)
dataframe = pd.DataFrame(matrix, index=['0','1'], columns=['0','1'])
sns.heatmap(dataframe, annot=True, cbar=None, cmap="Blues",fmt='.5g',square=True)
plt.title("RF Confusion Matrix"), plt.tight_layout()
plt.ylabel("True Class"), plt.xlabel("Predicted Class")
plt.show()


# In[31]:


# NB混淆矩阵
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
plt.figure()
y_pred2 = model_NB.predict(x_test)
matrix = confusion_matrix(y_test, y_pred2)
dataframe = pd.DataFrame(matrix, index=['0','1'], columns=['0','1'])
sns.heatmap(dataframe, annot=True, cbar=None, cmap="Blues",fmt='.5g',square=True)
plt.title("NB Confusion Matrix"), plt.tight_layout()
plt.ylabel("True Class"), plt.xlabel("Predicted Class")
plt.show()


# In[32]:


# LR混淆矩阵
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
plt.figure()
y_pred3 = model_LR.predict(x_test)
matrix = confusion_matrix(y_test, y_pred3)
dataframe = pd.DataFrame(matrix, index=['0','1'], columns=['0','1'])
sns.heatmap(dataframe, annot=True, cbar=None, cmap="Blues",fmt='.5g',square=True)
plt.title("LR Confusion Matrix"), plt.tight_layout()
plt.ylabel("True Class"), plt.xlabel("Predicted Class")
plt.show()


# In[33]:


# 绘制ROC曲线
def roc_curve_and_score(y_test, pred_proba):
    fpr, tpr, _ = roc_curve(y_test.ravel(), pred_proba.ravel())
    roc_auc = roc_auc_score(y_test.ravel(), pred_proba.ravel())
    return fpr, tpr, roc_auc
plt.figure()
plt.rcParams.update({'font.size': 14})
plt.grid()

fpr, tpr, roc_auc = roc_curve_and_score(y_test, model_RF.predict_proba(x_test)[:, 1])
plt.plot(fpr, tpr, color='b', lw=2,
         label='AUC RF={0:.4f}'.format(roc_auc))

fpr, tpr, roc_auc = roc_curve_and_score(y_test, model_NB.predict_proba(x_test)[:, 1])
plt.plot(fpr, tpr, color='lime', lw=2,
         label='AUC NB={0:.4f}'.format(roc_auc))
fpr, tpr, roc_auc = roc_curve_and_score(y_test, model_LR.predict_proba(x_test)[:, 1])
plt.plot(fpr, tpr, color='m', lw=2,
         label='AUC LR={0:.4f}'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
plt.legend(loc="lower right")
plt.title('ROC Curve')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('1 - Specificity')
plt.ylabel('Sensitivity')
plt.show()


# In[ ]:




