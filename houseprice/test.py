import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score

from matplotlib import *
train_df = pd.read_csv('./input/train.csv',index_col=0)
test_df=pd.read_csv('./input/test.csv',index_col=0)

prices = pd.DataFrame({"price":train_df['SalePrice'],
             "log(price+1)":np.log1p(train_df['SalePrice'])})
print(prices.head())
y_train = np.log1p(train_df.pop('SalePrice'))
#79列和79列的两个数据集合并
all_df = pd.concat((train_df,test_df),axis=0)
#这一步把这个标签的数据类型转为string
all_df['MSSubClass']=all_df['MSSubClass'].astype(str)
#统计每个词出现的次数
print(all_df['MSSubClass'].value_counts())

#对所有类别category数据进行one-hot处理
all_dummy_df = pd.get_dummies(all_df)
print(all_dummy_df.head())

#对所有numerical数据进行处理，首先看一下缺失值
print(all_dummy_df.isnull().sum().sort_values(ascending=False).head(10))

#用平均值来填补缺失值
mean_cols=all_dummy_df.mean()
print(mean_cols.head(10))
all_dummy_df = all_dummy_df.fillna(mean_cols)
print(all_dummy_df.isnull().sum().sum())

#对于numerical数据进行标准化，这一步并非必要的，但是regression的分类器需要用到
numeric_cols = all_df.columns[all_df.dtypes!='object']
print(numeric_cols)

#要使这里的数据点更平滑，除了使用logp之外，还可以用这个：
numeric_col_means = all_dummy_df.loc[:,numeric_cols].mean()
numeric_col_std=all_dummy_df.loc[:,numeric_cols].std()
all_dummy_df.loc[:,numeric_cols]=(all_dummy_df.loc[:,numeric_cols]-numeric_col_means)/numeric_col_std

#下面把数据集再分开成train和test
dummy_train_df = all_dummy_df.loc[train_df.index]
dummy_test_df = all_dummy_df.loc[test_df.index]


#下面选模型来试着跑一下 高维数据
#Ridge Regression 这个模型对于多因子的数据集，可以方便的把所有特征都无脑的塞进去

#首先把dataframe转为numpy的array
X_train = dummy_train_df.values
X_test = dummy_test_df.values

#为了确定选取何种模型，使用Sklearn自带的方法来测试模型：
alphas = np.logspace(-3,2,50)
#存储每次交叉验证的结果
test_scores=[]
#存下所有的cv值，看哪个alpha更好，也就是调参
for alpha in alphas:
    clf = Ridge(alpha)
    test_score = np.sqrt(-cross_val_score(clf,X_train,y_train,cv=10,scoring="neg_mean_squared_error"))
    test_scores.append(np.mean(test_score))

#这里绘图有问题
# import matplotlib
# # matplotlib.use("Agg")
# # import matplotlib.pyplot as plt
# # plt.plot(alphas,test_scores)
# # plt.title("Alpha vs CV Error")

from sklearn.ensemble import RandomForestRegressor
max_features = [.1, .3, .5, .7, .9, .99]
test_scores = []
for max_feat in max_features:
    clf = RandomForestRegressor(n_estimators=200, max_features=max_feat)
    test_score = np.sqrt(-cross_val_score(clf, X_train, y_train, cv=5, scoring='neg_mean_squared_error'))
    test_scores.append(np.mean(test_score))

ridge = Ridge(alpha=15)
rf = RandomForestRegressor(n_estimators=500,max_features=.3)

ridge.fit(X_train,y_train)
rf.fit(X_train,y_train)

y_ridge = np.expm1(ridge.predict(X_test))
y_rf=np.expm1(rf.predict(X_test))
#集成方法
y_final = (y_ridge + y_rf) / 2

#提交结果的格式是ID-房价预测
submission_df = pd.DataFrame(data= {'Id' : test_df.index, 'SalePrice': y_final})
print(submission_df.head(10))