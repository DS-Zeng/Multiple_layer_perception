import numpy as np

class Sigmoid(object):
    def __init__(self):
        pass

    def forward(self, x):
        self.output = 1 / (1 + np.exp(-x))
        return self.output
    
    def backward(self, grad):
        return grad * self.output * (1 - self.output)
    
class ReLU(object):
    def __init__(self):
        pass

    def forward(self, input):  
        self.input = input
        output = np.maximum(0, input)
        return output
    
    def backward(self, top_diff):  
        bottom_diff = top_diff
        bottom_diff[self.input<0] = 0
        return bottom_diff
    
class FC(object):
    def __init__(self, num_input, num_output, l2_lambda=0.0):
        self.num_input = num_input
        self.num_output = num_output
        self.l2_lambda = l2_lambda

    def init_param(self):
        self.W = self.W = np.random.randn(self.num_output, self.num_input) * np.sqrt(2.0 / (self.num_output + self.num_input)) # Xavier初始化：(num_output, num_input) 
        self.B = np.zeros((self.num_output,))  # 偏置矩阵 (num_categories, )

    def forward(self, X):
        self.X = X
        self.batch_size = X.shape[0]
        self.Y = np.dot(self.X, self.W.T) + self.B
        return self.Y

    def backward(self, top_diff, lr=0.01):
        self.learning_rate = lr   
        self.dW = np.dot(top_diff.T, self.X) + self.l2_lambda * self.W # (num_output, num_input)
        self.dB = np.squeeze(np.sum(top_diff, axis=0, keepdims=True)) # (num_output, )
        
        # 更新权重和偏置
        self.W -= self.learning_rate * self.dW
        self.B -= self.learning_rate * self.dB  

        bottom_diff = np.dot(top_diff, self.W)

        return bottom_diff
    
class Softmax(object):
    def __init__(self):
        self.output = None  # 保存前向传播的输出值
    
    def forward(self, x):
        self.batch_size = x.shape[0]
        # 为了数值稳定性，减去每行的最大值
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        self.output = exp_x / np.sum(exp_x, axis=1, keepdims=True)
        return self.output
    
    def backward(self, top_diff):
        bottom_diff = top_diff / self.batch_size # (batch_size, num_output)
        # print("Softmax botton diff: ", bottom_diff[0])
        return bottom_diff