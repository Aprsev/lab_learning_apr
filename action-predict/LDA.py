# 导入所需库
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
 
# 加载数据集
iris = load_iris()
X = iris.data  # 特征矩阵
y = iris.target  # 类别标签
 
# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
 
# 实例化LDA模型
lda = LDA()
 
# 使用训练数据拟合模型
lda.fit(X_train, y_train)
 
# 对测试集进行预测
y_pred = lda.predict(X_test)
 
# 计算预测准确率
accuracy = accuracy_score(y_test, y_pred)
print("LDA模型的预测准确率为：", accuracy)
 
# 讲解关键步骤：
# 1. 导入LDA类：从sklearn库中导入LDA类，用于实现线性判别分析。
# 2. 加载数据集：这里使用内置的iris数据集作为示例，实际应用中可以替换为自己的数据。
# 3. 数据分割：将数据集划分为训练集和测试集，以评估模型在未知数据上的表现。
# 4. 创建并拟合模型：创建一个LDA对象，然后使用训练集的数据和对应标签对模型进行训练。
# 5. 预测：利用训练好的LDA模型对测试集数据进行预测，得到预测类别标签。
# 6. 评估：计算预测标签与实际标签之间的准确率，评估模型的性能。
 
# 注意：LDA默认会进行降维，但可以通过设置参数`n_components`来控制降维后的维度数目。
# 若要进行纯粹的分类而不降维，可以设置`solver='svd'`以及`shrinkage=None`（在较新版本的sklearn中）。