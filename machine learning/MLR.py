import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

# 设置随机种子，确保结果可复现
np.random.seed(0)

# 生成数据集的参数
n_samples = 100  # 样本数量
n_features = 2   # 自变量数量

# 生成自变量数据
X1 = np.random.randn(n_samples)
X2 = np.random.randn(n_samples)

# 生成因变量数据，模拟 Y = 2*X1 + 3*X2 + N(0, 1) 的关系
beta0 = 1.0
beta1 = 2.0
beta2 = 3.0
epsilon = np.random.normal(0, 1, n_samples)

Y = beta0 + beta1*X1 + beta2*X2 + epsilon

# 创建数据集的DataFrame
data = {
    'X1': X1,
    'X2': X2,
    'Y': Y
}
df = pd.DataFrame(data)

# 为X添加截距项
X = sm.add_constant(df[['X1', 'X2']])

# 使用statsmodels拟合多元线性回归模型
model = sm.OLS(df['Y'], X).fit()

# 打印模型摘要
print(model.summary())

# 可视化拟合曲线
# 首先生成用于绘图的数据点
x1_plot = np.linspace(df['X1'].min(), df['X1'].max(), 100)
x2_plot = np.linspace(df['X2'].min(), df['X2'].max(), 100)
x1_plot, x2_plot = np.meshgrid(x1_plot, x2_plot)
X_plot = sm.add_constant(np.column_stack((x1_plot.ravel(), x2_plot.ravel())))

# 预测Y值
Y_plot = model.predict(X_plot)

# 绘制拟合曲面
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df['X1'], df['X2'], df['Y'], color='blue', label='Actual data')
ax.plot_surface(x1_plot, x2_plot, Y_plot.reshape(100, 100), color='red', alpha=0.5, label='Fitted surface')
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('Y')
ax.legend()
plt.show()