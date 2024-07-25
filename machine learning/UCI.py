import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, LeaveOneOut
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# 加载数据集
df_auto = pd.read_csv('d:\\Desktop\\Lab\\tests\\machine learning\\iris.data', header=None)
df_housing = pd.read_csv('d:\\Desktop\\Lab\\tests\\machine learning\\iris.data', header=None)

# 预处理数据集
X_auto = df_auto.drop(0, axis=1)  # 假设第一列是目标变量
X_auto = df_auto.drop(4, axis=1)
y_auto = df_auto[0]

X_housing = df_housing.drop(1, axis=1)  # 假设最后一列是目标变量
X_housing = df_housing.drop(4, axis=1)
y_housing = df_housing[1]

# 定义模型
model = LinearRegression()

# 10折交叉验证
cv = 10
scores_auto_10fold = cross_val_score(model, X_auto, y_auto, cv=cv, scoring='neg_mean_squared_error')
scores_housing_10fold = cross_val_score(model, X_housing, y_housing, cv=cv, scoring='neg_mean_squared_error')

# 留一法交叉验证
loo_auto = LeaveOneOut()
loo_housing = LeaveOneOut()

scores_auto_loocv = cross_val_score(model, X_auto, y_auto, cv=loo_auto, scoring='neg_mean_squared_error')
scores_housing_loocv = cross_val_score(model, X_housing, y_housing, cv=loo_housing, scoring='neg_mean_squared_error')

# 计算平均MSE
mse_auto_10fold = -scores_auto_10fold.mean()
mse_housing_10fold = -scores_housing_10fold.mean()
mse_auto_loocv = -scores_auto_loocv.mean()
mse_housing_loocv = -scores_housing_loocv.mean()

print(f'Auto MPG - 10-Fold CV MSE: {mse_auto_10fold}, LOOCV MSE: {mse_auto_loocv}')
print(f'Housing - 10-Fold CV MSE: {mse_housing_10fold}, LOOCV MSE: {mse_housing_loocv}')