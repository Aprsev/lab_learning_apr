import numpy as np
from sklearn.datasets import load_iris
 
# 加载数据
iris = load_iris()    
X    = iris.data
x_mean = X.mean(axis=0)  # 样本的中心 
 
# 通过SVD分解,得到A与XA每列的方差var
U,S,VT = np.linalg.svd((X-x_mean)/np.sqrt(X.shape[0]-1)) # 注意，numpy的SVD分解出的是US(VT)
A      = VT.T                                            # 主成份系数矩阵A
var    = S*S                                             # 方差                                    
pr     = var/var.sum()                                   # 方差占比
 
#打印结果
print('主成份系数矩阵A:\n A=',A)
print('主成份方差var:',var)
print('主成份贡献占比(方差占比)Pr:',pr)
 
# 获取主成份数据
y= (X-x_mean)@A                                          # 通过调用公式计算主成份数据  