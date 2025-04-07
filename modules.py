import numpy as np

class FC(object):
    def __init__(self, num_input, num_output, reg_lambda=0.0):
        # Initialization
        self.num_input = num_input
        self.num_output = num_output
        self.reg_lambda = reg_lambda  # L2 Regularization strength
        
    def init_param(self, std=0.01):
        # Initialize parameters
        self.weight = np.random.normal(loc=0.0, scale=std, size=(self.num_input, self.num_output))
        self.bias = np.zeros([1, self.num_output])
    
    def forward(self, input):
        # Forward propagation
        self.input = input
        # Y = WX + b
        self.output = np.matmul(input, self.weight) + self.bias
        # L2 regularization loss
        self.reg_loss = 0.5 * self.reg_lambda * np.sum(self.weight ** 2)  
        return self.output
    
    def backward(self, top_diff):
        # Backward propagation
        # Gradient with regularization term
        self.d_weight = np.dot(self.input.T, top_diff) + self.reg_lambda * self.weight
        self.d_bias = np.sum(top_diff, axis=0)
        # Gradient of loss with respect to input
        bottom_diff = np.dot(top_diff, self.weight.T)
        return bottom_diff
    
    def update_param(self, lr):
        # Update parameters
        self.weight = self.weight - lr * self.d_weight
        self.bias = self.bias - lr * self.d_bias


    def load_param(self, weight, bias):
        # Load parameter
        assert self.weight.shape == weight.shape
        assert self.bias.shape == bias.shape
        self.weight = weight
        self.bias = bias
    def save_param(self):  
        # Save parameter
        return self.weight, self.bias
    


class ReLU(object):
    def __init__(self):
        pass
    def forward(self, input):  
        # Forward propagation of ReLU
        self.input = input
        output = np.maximum(0, input)
        return output
    
    def backward(self, top_diff):  
        # Backward propagation of ReLU
        bottom_diff = top_diff
        bottom_diff[self.input<0] = 0
        return bottom_diff
    
class Sigmoid(object):
    def __init__(self):
        pass
    
    def forward(self, input):
        # Forward propagation of the Sigmoid function
        self.output = 1 / (1 + np.exp(-input))
        return self.output
    
    def backward(self, top_diff):
        # Backward propagation of the Sigmoid function
        bottom_diff = top_diff * self.output * (1 - self.output)
        return bottom_diff

class Tanh(object):
    def __init__(self):
        pass
    
    def forward(self, input):
        # Forward propagation of the Tanh function
        self.output = np.tanh(input)
        return self.output
    
    def backward(self, top_diff):
        # Backward propagation of the Tanh function
        bottom_diff = top_diff * (1 - self.output**2)
        return bottom_diff



class Softmax(object):
    def __init__(self):
        pass
    def forward(self, input):  
        # Forward propagation
        input_max = np.max(input, axis=1, keepdims=True)
        input_exp = np.exp(input - input_max)
        self.prob = input_exp / np.sum(input_exp, axis=1, keepdims=True)
        return self.prob
    
    def entropy_loss(self, label):  
        # Loss calculation
        self.batch_size = self.prob.shape[0]
        self.label_onehot = np.zeros_like(self.prob)
        self.label_onehot[np.arange(self.batch_size), label] = 1.0
        loss = -np.sum(np.log(self.prob) * self.label_onehot) / self.batch_size
        return loss
    
    def backward(self):  
        # Backward propagation
        bottom_diff = (self.prob - self.label_onehot) / self.batch_size
        return bottom_diff