import numpy as np

# 计算欧氏距离
def euclidean_distance(matrix1, matrix2):
    return np.linalg.norm(matrix1 - matrix2)

# 定义神经网络类
class MatrixSimilarityNetwork:
    def __init__(self):
        np.random.seed(1)
        self.weights_input_hidden = 2 * np.random.random((4, 4)) - 1
        self.weights_hidden_output = 2 * np.random.random((4, 1)) - 1
    
    def train(self, matrix_pairs, similarities, iterations):
        for iteration in range(iterations):
            for i in range(len(matrix_pairs)):
                input_matrix = matrix_pairs[i].flatten()  # 展平输入矩阵
                predicted_similarity = self.predict(input_matrix)
                error = similarities[i] - predicted_similarity
                
                # 反向传播
                output_delta = error
                hidden_error = output_delta.dot(self.weights_hidden_output.T)
                
                # 计算 hidden_delta，使用 sigmoid 激活函数的导数
                sigmoid_input = input_matrix.dot(self.weights_input_hidden)
                sigmoid_output = 1 / (1 + np.exp(-sigmoid_input))
                sigmoid_gradient = sigmoid_output * (1 - sigmoid_output)
                hidden_delta = hidden_error * sigmoid_gradient
                
                # 更新权重
                self.weights_hidden_output += input_matrix.reshape(-1, 1) * output_delta
                self.weights_input_hidden += input_matrix.reshape(-1, 1) * hidden_delta
    
    def predict(self, input_matrix):
        hidden_layer_input = np.dot(input_matrix, self.weights_input_hidden)
        hidden_layer_output = 1 / (1 + np.exp(-hidden_layer_input))
        
        output_layer_input = np.dot(hidden_layer_output, self.weights_hidden_output)
        predicted_similarity = output_layer_input
        
        return predicted_similarity

# 示例数据
matrix1 = np.array([[1, 2], [3, 4]])
matrix2 = np.array([[2, 3], [4, 5]])
matrix_pairs = [matrix1, matrix2]
similarities = [euclidean_distance(matrix1, matrix2), 0.5]  # 使用欧氏距离作为相似性

# 创建神经网络实例并训练
network = MatrixSimilarityNetwork()
network.train(matrix_pairs, similarities, iterations=1000)

# 预测并输出结果
for i in range(len(matrix_pairs)):
    input_matrix = matrix_pairs[i].flatten()
    predicted_similarity = network.predict(input_matrix)
    print(f"Matrix pair {i+1}: Predicted similarity: {predicted_similarity}, Actual similarity: {similarities[i]}")
