import numpy as np

# 定义 sigmoid 函数和其导数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# 定义神经网络类
class NeuralNetwork:
    def __init__(self):
        # 初始化随机种子，以便每次运行得到相同结果
        np.random.seed()
        
        # 初始化权重，连接输入层到隐藏层 (2x2 矩阵) 和隐藏层到输出层 (2x1 矩阵)
        self.weights_input_hidden = 2 * np.random.random((3, 3)) - 1
        self.weights_hidden_output = 2 * np.random.random((3, 1)) - 1
    
    def train(self, inputs, outputs, iterations):
        for iteration in range(iterations):
            # 前向传播
            hidden_layer_input = np.dot(inputs, self.weights_input_hidden)
            hidden_layer_output = sigmoid(hidden_layer_input)
            
            output_layer_input = np.dot(hidden_layer_output, self.weights_hidden_output)
            output = sigmoid(output_layer_input)
            
            # 计算误差
            error = outputs - output
            
            # 反向传播
            # 输出层误差和梯度
            output_gradient = sigmoid_derivative(output)
            output_delta = error * output_gradient
            
            # 隐藏层误差和梯度
            hidden_error = output_delta.dot(self.weights_hidden_output.T)
            hidden_gradient = sigmoid_derivative(hidden_layer_output)
            hidden_delta = hidden_error * hidden_gradient
            
            # 更新权重
            self.weights_hidden_output += hidden_layer_output.T.dot(output_delta)
            self.weights_input_hidden += inputs.T.dot(hidden_delta)
    
    def predict(self, inputs):
        # 前向传播预测
        hidden_layer_input = np.dot(inputs, self.weights_input_hidden)
        hidden_layer_output = sigmoid(hidden_layer_input)
        
        output_layer_input = np.dot(hidden_layer_output, self.weights_hidden_output)
        predicted_output = sigmoid(output_layer_input)
        
        return predicted_output

# 定义输入数据集和对应的输出
inputs = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]])
outputs = np.array([[0], [0],[1],[0],[1],[1] ,[1], [1]])

for i in range(1):
    # 创建神经网络实例并训练
    neural_network = NeuralNetwork()
    neural_network.train(inputs, outputs, 1000000)

    # 预测并输出结果
    print("Predictions after training:")
    for input in inputs:
        print(f"Input: {input} Predicted Output: {neural_network.predict(input)}")
