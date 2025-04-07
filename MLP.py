import numpy as np
from modules import FC, Sigmoid, Softmax, ReLU
from utils import load_cifar10_data, one_hot_encode, load_mnist

class MLP(object):
    def __init__(self, layers=[], lr=0.01): 
        self.layers_dim = layers[1:-1] # 中间层维度
        self.dim_input = layers[0]
        self.dim_output = layers[-1]
        self.layers = [] # 中间层，例：【FC, Sigmoid, ...】
        self.learning_rate = lr

    def build_model(self):
        prev_dim = self.dim_input
        for layer_dim in self.layers_dim:
            FC_layer = FC(prev_dim, layer_dim) # 全连接层
            FC_layer.init_param()
            self.layers.append(FC_layer)
            prev_dim = layer_dim
            self.layers.append(ReLU()) # 激活层

        last_layer = FC(prev_dim, self.dim_output)
        last_layer.init_param()
        self.layers.append(last_layer) # 最后一层线性层，不激活

        self.layers.append(Softmax()) # 输出层

    def forward(self, X): # X: [batch_size, dim_input]
        for layer in self.layers:
            X = layer.forward(X)
        self.Y = X
        return self.Y
    
    def backward(self, topdiff):
        dloss = self.layers[-1].backward(topdiff) 
        # 从倒数第二层到第一层反向传播
        for i in range(len(self.layers) - 2, -1, -1): 
            # 传递梯度到前一层
            if isinstance(self.layers[i], FC):
                dloss = self.layers[i].backward(dloss, lr=self.learning_rate)
            else:
                dloss = self.layers[i].backward(dloss)

    def evaluate(self, targets, X_test=None):  # target: one-hot encoding [batch_size, dim_output]
        if X_test is None:
            predicted_indices = np.argmax(self.Y, axis=1) 
        else:
            output = self.forward(X_test)
            predicted_indices = np.argmax(output, axis=1)
        
        target_indices = np.argmax(targets, axis=1) 
        correct_predictions = np.sum(predicted_indices == target_indices) 
        accuracy = correct_predictions / len(targets) 
        return accuracy

    def train(self, X_train, y_train, X_val, y_val, num_epochs, batch_size=1000, decay_factor=0.9, decay_steps=10, print_iter=500):
        self.batch_size = batch_size
        self.decay_factor = decay_factor
        self.decay_steps = decay_steps
        self.print_iter = print_iter

        max_batch = int(X_train.shape[0] / self.batch_size)

        print('Start training...')
        for epoch in range(num_epochs):
            # Shuffle the training data
            indices = np.random.permutation(X_train.shape[0])
            X_train = X_train[indices]
            y_train = y_train[indices]
            # Learning rate decay
            if epoch % self.decay_steps == 0 and epoch != 0:
                self.learning_rate *= self.decay_factor
                print("=" * 50)
                print(f"Decaying learning rate to {self.learning_rate:.6f}")
                print("=" * 50)

            # Mini-batch training
            for idx_batch in range(max_batch):
                batch_images = X_train[idx_batch*self.batch_size:(idx_batch+1)*self.batch_size, :]
                batch_labels = y_train[idx_batch*self.batch_size:(idx_batch+1)*self.batch_size]\
                
                # Forward propagation
                output = self.forward(batch_images)

                # Backward propagation and update parameters
                topdiff = output - batch_labels
                self.backward(topdiff)  

                # Print loss and accuracy
                if idx_batch % self.print_iter == 0:
                    batch_train = X_train[max((idx_batch-self.print_iter+1),0) * self.batch_size : (idx_batch+1)*self.batch_size, :]
                    batch_labels = y_train[max((idx_batch-self.print_iter+1),0) * self.batch_size : (idx_batch+1)*self.batch_size]

                    # Train loss
                    train_output = self.forward(batch_train)
                    train_loss = np.mean(-np.sum(batch_labels * np.log(train_output + 1e-6), axis=1))
        
                    # Train accuracy
                    train_accuracy = self.evaluate(targets=batch_labels, X_test=batch_train)
                    
                    print(f"Epoch {epoch}, Iter {idx_batch}, Train Loss: {train_loss:.6f}, Train Accuracy: {train_accuracy:.3f}")

                    # Validation
                    test_accuracy = self.evaluate(targets=y_val, X_test=X_val)
                    print(f"Test Accuracy = {test_accuracy * 100:.2f}%")


if __name__ == "__main__":
    # 参数设置
    data_dir = "MLP-Numpy/data/cifar-10-batches-py"  # CIFAR-10 数据集的本地路径
    num_input = 32 * 32 * 3  # CIFAR-10 图像展平后的维度

    # data_dir = "MLP-Numpy/data/fashion"  # Fasion 数据集的本地路径
    # num_input = 28 * 28 * 1  # Fasion 图像展平后的维度
    
    num_output = 10  # CIFAR-10/Fasion 分类类别
    learning_rate = 0.005
    epochs = 400

    # 加载 CIFAR-10 训练数据
    X_train, y_train = load_cifar10_data(data_dir, train=True)
    y_train_one_hot = one_hot_encode(y_train, num_output)
    X_test, y_test = load_cifar10_data(data_dir, train=False)
    y_test_one_hot = one_hot_encode(y_test, num_output)

    # # 加载 Fasion 训练数据
    # X_train, y_train = load_mnist(data_dir, kind='train')
    # y_train_one_hot = one_hot_encode(y_train, num_output)
    # X_test, y_test = load_mnist(data_dir, kind='t10k')
    # y_test_one_hot = one_hot_encode(y_test, num_output)

    layers = [num_input, 512, 512, 128, 128, 32, num_output]
    mlp = MLP(layers=layers, lr=learning_rate)

    mlp.build_model()
    mlp.train(X_train, y_train_one_hot, X_test, y_test_one_hot, epochs)

    accuracy = mlp.evaluate(targets=y_test, X_test=X_test)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")