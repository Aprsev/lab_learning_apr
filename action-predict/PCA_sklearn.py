# -*- coding: utf-8 -*-
"""
主成份分析求解DEMO(调用sklearn)
本代码来自老饼讲解-机器学习:www.bbbdata.com
"""
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
# 加载数据
iris   = load_iris()    
X      = iris.data       # 样本X
x_mean = X.mean(axis=0)  # 样本的中心 
 
# 用PCA对X进行主成份分析
clf = PCA()   # 初始化PCA对象
clf.fit(X)    # 对X进行主成份分析
 
# 打印结果
print('主成份系数矩阵A:\n A=',clf.components_)
print('主成份方差var:',clf.explained_variance_)
print('主成份贡献占比(方差占比)Pr:',clf.explained_variance_ratio_)
 
# 获取主成份数据
y = clf.transform(X)                # 通过调用transform方法获取主成份数据  
y2= (X-x_mean)@clf.components_.T    # 通过调用公式计算主成份数据 