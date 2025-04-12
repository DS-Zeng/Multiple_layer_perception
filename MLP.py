import numpy as np
from modules import FC, Sigmoid, Softmax, ReLU
from utils import load_cifar10_data_with_val_split, one_hot_encode, load_mnist_with_val_split, hyperparameter_search
import pickle
import matplotlib.pyplot as plt


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

    def save_parameters(self, filepath):
        parameters = []  # 存储每个 FC 层的权重和偏置
        for layer in self.layers:
            if isinstance(layer, FC):  
                parameters.append({
                    "weights": layer.W,
                    "biases": layer.B
                })
        with open(filepath, 'wb') as f:
            pickle.dump(parameters, f) 
        print(f"Parameters successfully saved to {filepath}")
    
    def load_parameters(self, filepath):
        with open(filepath, 'rb') as f:
            parameters = pickle.load(f)  
        fc_idx = 0  # 跟踪当前 FC 层索引
        for layer in self.layers:
            if isinstance(layer, FC):  
                layer.W = parameters[fc_idx]["weights"]  
                layer.B = parameters[fc_idx]["biases"]  
                fc_idx += 1
        print(f"Parameters successfully loaded from {filepath}")

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

    def evaluate(self, X_val, y_val):  # target: one-hot encoding [batch_size, dim_output]
        output = self.forward(X_val)
        predicted_indices = np.argmax(output, axis=1)
        
        target_indices = np.argmax(y_val, axis=1) 
        correct_predictions = np.sum(predicted_indices == target_indices) 
        accuracy = correct_predictions / len(y_val) 
        return accuracy
    
    def predict(self, X_test):
        output = self.forward(X_test)
        predicted_indices = np.argmax(output, axis=1)
        return predicted_indices

    def train(self, X_train, y_train, X_val, y_val, num_epochs, batch_size=256, decay_factor=0.95, decay_steps=5, print_iter=500):
        self.batch_size = batch_size
        self.decay_factor = decay_factor
        self.decay_steps = decay_steps
        self.print_iter = print_iter

        max_batch = int(X_train.shape[0] / self.batch_size)

        # 用于保存训练过程的指标
        train_losses = []  # 存储训练集 loss
        val_losses = []    # 存储验证集 loss
        val_accuracies = []  # 存储验证集 accuracy

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
                batch_labels = y_train[idx_batch*self.batch_size:(idx_batch+1)*self.batch_size]
                
                # Forward propagation
                output = self.forward(batch_images)

                # Backward propagation and update parameters
                topdiff = output - batch_labels
                self.backward(topdiff)

                # 每次迭代记录 loss 和 accuracy
                if idx_batch % self.print_iter == 0:
                    batch_train = X_train[max((idx_batch-self.print_iter+1),0) * self.batch_size : (idx_batch+1)*self.batch_size, :]
                    batch_labels = y_train[max((idx_batch-self.print_iter+1),0) * self.batch_size : (idx_batch+1)*self.batch_size]

                    # Train loss
                    train_output = self.forward(batch_train)
                    train_loss = np.mean(-np.sum(batch_labels * np.log(train_output + 1e-6), axis=1))
                    
                    # Validation loss
                    val_output = self.forward(X_val)
                    val_loss = np.mean(-np.sum(y_val * np.log(val_output + 1e-6), axis=1))

                    # Train accuracy
                    train_accuracy = self.evaluate(batch_train, batch_labels)

                    # Validation accuracy
                    val_accuracy = self.evaluate(X_val, y_val)

                    # 记录 loss 和 accuracy
                    train_losses.append(train_loss)
                    val_losses.append(val_loss)
                    val_accuracies.append(val_accuracy)
                    
                    # 打印训练过程信息
                    print(f"Epoch {epoch}, Iter {idx_batch}, Train Loss: {train_loss:.6f}, Train Accuracy: {train_accuracy:.3f}")
                    print(f"Validation Loss: {val_loss:.6f}, Validation Accuracy: {val_accuracy * 100:.2f}%")
        
        # 训练完成后绘制曲线
        self.plot_training_curves(train_losses, val_losses, val_accuracies)
        print("Training complete.")

    def plot_training_curves(self, train_losses, val_losses, val_accuracies):
        # 绘制 Loss 曲线
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label="Train Loss")
        plt.plot(val_losses, label="Val Loss")
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.title("Loss Curve")
        plt.legend()

        # 绘制 Accuracy 曲线
        plt.subplot(1, 2, 2)
        plt.plot(val_accuracies, label="Val Accuracy", color='green')
        plt.xlabel("Iterations")
        plt.ylabel("Accuracy")
        plt.title("Validation Accuracy Curve")
        plt.legend()

        plt.tight_layout()

        # 保存图像
        plt.savefig('Cifar-10', dpi=300)  # 保存图片，默认分辨率为 300 DPI
        print(f"Training curves successfully saved.")
        plt.show() # 显示图像


if __name__ == "__main__":
    # 参数设置
    data_dir = "/SSD_DISK/users/zengzixuan/Courses/CV-PJ/Linear-Classifier/data/cifar-10-batches-py"  # CIFAR-10 数据集的本地路径
    num_input = 32 * 32 * 3  # CIFAR-10 图像展平后的维度
    # data_dir = "/SSD_DISK/users/zengzixuan/Courses/CV-PJ/Linear-Classifier/data/fashion"  # Fasion 数据集的本地路径
    # num_input = 28 * 28 * 1  # Fasion 图像展平后的维度

    num_output = 10  # CIFAR-10/Fasion 分类类别
    learning_rate = 0.005
    epochs = 250
    val_split = 0.1  # 验证集比例
    batch_size = 256

    # 加载 CIFAR-10 训练数据
    X_train, y_train, X_val, y_val = load_cifar10_data_with_val_split(data_dir, train=True, val_split=val_split)
    y_train_one_hot = one_hot_encode(y_train, num_output)
    y_val_one_hot = one_hot_encode(y_val, num_output)

    X_test, y_test = load_cifar10_data_with_val_split(data_dir, train=False)
    y_test_one_hot = one_hot_encode(y_test, num_output)

    # # 加载 Fasion 训练数据
    # # 加载训练数据，同时划分验证集
    # (X_train, y_train), (X_val, y_val) = load_mnist_with_val_split(data_dir, kind='train', val_split=val_split)
    # y_train_one_hot = one_hot_encode(y_train, num_output)
    # y_val_one_hot = one_hot_encode(y_val, num_output)

    # X_test, y_test = load_mnist_with_val_split(data_dir, kind='t10k')
    # y_test_one_hot = one_hot_encode(y_test, num_output)


    layers = [num_input, 512, num_output]
    mlp = MLP(layers=layers, lr=learning_rate)

    mlp.build_model()
    mlp.train(X_train, y_train_one_hot, X_val, y_val_one_hot, epochs)

    # Test
    predicted_indices = mlp.predict(X_test)  # 得到预测的类别索引
    correct_predictions = np.sum(predicted_indices == y_test)  # 对比真实类别索引
    accuracy = correct_predictions / len(y_test)  # 计算准确率
    print("="*20 + "Final result" + "="*20)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

    mlp.save_parameters("best_model_parameters_cifar.pkl")

    # # 定义超参数搜索空间
    # search_space = {
    #     "hidden_dims": [128, 256, 512, 1024],
    #     "lr_initial": [0.005, 0.001, 0.0005],
    #     "decay_factor": [0.95, 0.9, 0.8]
    # }

    # # 文件保存路径
    # results_filepath = "hyperparameter_search_results_cifar.pkl"
    # plot_filepath = "validation_accuracy_plot_cifar.png"

    # # 执行超参数搜索
    # hyperparameter_search(MLP, X_train, y_train_one_hot, X_val, y_val_one_hot, epochs, batch_size, search_space, results_filepath, plot_filepath)