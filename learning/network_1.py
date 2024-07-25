import numpy as np

# 输入数据和标签
x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

flag=True

while flag:
    # 定义神经网络
    class NeuralNetwork:
        def __init__(self):
            np.random.seed()
            self.weights = 2 * np.random.random((2, 1)) - 1
            print(self.weights)
    
        def sigmoid(self, x):
            return 1 / (1 + np.exp(-x))
    
        def sigmoid_derivative(self, x):
            return x * (1 - x)
    
        def train(self, x, y, iterations):
            for i in range(iterations):
                output = self.predict(x)
                error = y - output
                adjustment = np.dot(x.T, error * self.sigmoid_derivative(output))
                self.weights += adjustment*0.1
    
        def predict(self, x):
            return self.sigmoid(np.dot(x, self.weights))
        
    # 训练神经网络
    neural_network = NeuralNetwork()
    neural_network.train(x, y, 1000)
    
    # 预测
    print(neural_network.predict(np.array([0, 0])),end="")
    print(neural_network.predict(np.array([0, 1])),end="")
    print(neural_network.predict(np.array([1, 0])),end="")
    print(neural_network.predict(np.array([1, 1])))
    if np.array_equal( neural_network.predict(np.array([0, 0])) , neural_network.predict(np.array([0, 1]))) and np.array_equal( neural_network.predict(np.array([1, 0])) , neural_network.predict(np.array([0, 1]))):
        continue
    else:
        flag=False
        break
