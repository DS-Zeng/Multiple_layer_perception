import numpy as np
import os
import gzip
import matplotlib.pyplot as plt
import pickle

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        data_dict = pickle.load(fo, encoding='bytes')
    return data_dict

# 加载 CIFAR-10 数据集
def load_cifar10_data(data_dir, train=True):
    """
    Args:
        data_dir: CIFAR-10 dataset directory path
        train: Whether to load training data or test data
    
    Returns:
        X: Data matrix of shape (num_samples, 32 * 32 * 3)
        y: Labels of shape (num_samples,)
    """
    if train:
        batches = [f"data_batch_{i}" for i in range(1, 6)]
    else:
        batches = ["test_batch"]
    
    X = []
    y = []
    for batch in batches:
        data_dict = unpickle(f"{data_dir}/{batch}")
        X.append(data_dict[b'data'])  # CIFAR-10 图片是展平的 3072 维度 (32*32*3)
        y.extend(data_dict[b'labels'])
    
    X = np.concatenate(X, axis=0)
    y = np.array(y)
    
    # 归一化图像数据
    X = X / 255.0
    return X, y

def load_cifar10_data_with_val_split(data_dir, train=True, val_split=0.2):
    if train:
        batches = [f"data_batch_{i}" for i in range(1, 6)]
    else:
        batches = ["test_batch"]
    
    X = []
    y = []
    for batch in batches:
        data_dict = unpickle(f"{data_dir}/{batch}")
        X.append(data_dict[b'data'])  # CIFAR-10 32 * 32 * 3
        y.extend(data_dict[b'labels']) 
    
    X = np.concatenate(X, axis=0)
    y = np.array(y)
    
    # 归一化图像数据
    X = X / 255.0

    if train:
        # 划分训练集和验证集
        num_samples = len(y)
        num_val_samples = int(val_split * num_samples)
        
        # 随机排列数据
        indices = np.random.permutation(num_samples)
        X = X[indices]
        y = y[indices]
        
        # 划分数据
        X_val = X[:num_val_samples]
        y_val = y[:num_val_samples]
        X_train = X[num_val_samples:]
        y_train = y[num_val_samples:]
        
        return X_train, y_train, X_val, y_val
    
    else:
        return X, y

# 将标签转换为 one-hot 编码
def one_hot_encode(y, num_categories):
    """
    将分类标签转换为 one-hot 编码矩阵

    参数:
    - y: 原始标签 (num_samples,)
    - num_categories: 类别数

    返回:
    - one_hot: one-hot 编码 (num_samples, num_categories)
    """
    one_hot = np.zeros((len(y), num_categories))
    one_hot[np.arange(len(y)), y] = 1
    return one_hot # (num_samples, num_categories)

def load_mnist(path, kind='train'):

    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte.gz'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte.gz'
                               % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)

    return images/255.0, labels

import numpy as np
import os
import gzip

def load_mnist_with_val_split(path, kind='train', val_split=0.1, random_seed=42):
    """
    加载 MNIST 数据集，同时划分一部分训练集为验证集。

    参数：
    path : str
        MNIST 数据集的路径
    kind : str
        'train' 加载训练集，'t10k' 加载测试集
    val_split : float
        验证集所占训练集的比例（需要在 (0, 1) 之间）
    random_seed : int
        随机种子，保证结果可复现

    返回：
    （训练集，验证集）或（测试数据，标签）
        如果是训练集返回：
            (X_train, y_train), (X_val, y_val)
        如果是测试集返回：
            X_test, y_test
    """
    labels_path = os.path.join(path, f'{kind}-labels-idx1-ubyte.gz')
    images_path = os.path.join(path, f'{kind}-images-idx3-ubyte.gz')

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16).reshape(len(labels), 784)

    images = images / 255.0  # 将像素值归一化到 [0, 1]

    if kind == 'train':  # 如果是训练集，则进一步划分出验证集
        np.random.seed(random_seed)  
        indices = np.arange(len(labels))  
        np.random.shuffle(indices)  # 随机打乱索引

        # 计算验证集大小
        val_size = int(len(labels) * val_split)

        val_indices = indices[:val_size]  # 前 val_size 个作为验证集
        train_indices = indices[val_size:]  # 剩余的作为训练集

        X_train, y_train = images[train_indices], labels[train_indices]
        X_val, y_val = images[val_indices], labels[val_indices]

        return (X_train, y_train), (X_val, y_val)
    else:  # 如果是测试集，则直接返回
        return images, labels

def hyperparameter_search(MLP, X_train, y_train, X_val, y_val, num_epochs, batch_size, search_space, results_filepath, plot_filepath):
    """
    进行超参数搜索。
    
    Args:
    - X_train, y_train: 训练数据和标签
    - X_val, y_val: 验证数据和标签
    - num_epochs: 训练轮次
    - batch_size: mini-batch大小
    - search_space: 搜索空间的超参数字典
        {"hidden_dims": [128, 256, 512],
         "lr_initial": [0.01, 0.005, 0.001],
         "decay_factor": [0.9, 0.8, 0.7]}
    - results_filepath: 保存搜索结果到文件的路径
    - plot_filepath: 保存验证集准确率随迭代步数变化曲线的路径
    """
    # 保存结果
    search_results = []
    best_accuracy = 0.0
    best_model = None
    best_hyperparams = None

    print("Hyperparameter search started...")
    
    for hidden_dim in search_space["hidden_dims"]:
        for lr_initial in search_space["lr_initial"]:
            for decay_factor in search_space["decay_factor"]:
                print("=" * 50)
                print(f"Training with hidden_dim={hidden_dim}, lr_initial={lr_initial}, decay_factor={decay_factor}")
                
                # 构建 MLP
                layers = [X_train.shape[1], hidden_dim, y_train.shape[1]]
                mlp = MLP(layers=layers, lr=lr_initial)
                mlp.build_model()

                # 初始化
                validation_accuracies = []
                
                # 训练 MLP
                max_batch = int(X_train.shape[0] / batch_size)
                learning_rate = lr_initial
                
                for epoch in range(num_epochs):
                    # Shuffle the training data
                    indices = np.random.permutation(X_train.shape[0])
                    X_train = X_train[indices]
                    y_train = y_train[indices]
                    
                    # Learning rate decay
                    if epoch % 10 == 0 and epoch != 0:  # 根据 decay_steps 衰减
                        learning_rate *= decay_factor
                    
                    for idx_batch in range(max_batch):
                        batch_images = X_train[idx_batch * batch_size:(idx_batch+1) * batch_size, :]
                        batch_labels = y_train[idx_batch * batch_size:(idx_batch+1) * batch_size]
                        
                        # Forward propagation
                        output = mlp.forward(batch_images)
                        
                        # Backward propagation
                        topdiff = output - batch_labels
                        mlp.backward(topdiff)
                    
                    # 验证集准确率
                    val_accuracy = mlp.evaluate(X_val, y_val)
                    validation_accuracies.append(val_accuracy)
                    print(f"Epoch {epoch}, Validation Accuracy = {val_accuracy * 100:.2f}%")

                # 保存搜索结果
                search_results.append({
                    "hidden_dim": hidden_dim,
                    "lr_initial": lr_initial,
                    "decay_factor": decay_factor,
                    "validation_accuracies": validation_accuracies
                })

                # 如果验证集准确率更高，则保存这个模型
                if validation_accuracies[-1] > best_accuracy:
                    best_accuracy = validation_accuracies[-1]
                    best_model = mlp
                    best_hyperparams = {
                        "hidden_dim": hidden_dim,
                        "lr_initial": lr_initial,
                        "decay_factor": decay_factor,
                    }

    print("=" * 50)
    print(f"Best Model: {best_hyperparams}, Validation Accuracy = {best_accuracy:.2f}")
    
    # 保存超参数搜索结果到文件
    with open(results_filepath, 'wb') as f:
        pickle.dump(search_results, f)
    print(f"All search results saved to {results_filepath}")
    
    # 绘制验证集准确率随时间变化并保存
    for result in search_results:
        plt.plot(result["validation_accuracies"], label=f"hidden_dim={result['hidden_dim']}, lr={result['lr_initial']}, decay={result['decay_factor']}")
    plt.xlabel("Epochs")
    plt.ylabel("Validation Accuracy")
    plt.title("Validation Accuracy vs. Epochs")
    plt.legend()
    plt.savefig(plot_filepath)
    print(f"Validation accuracy plot saved to {plot_filepath}")
    
    # 保存最佳模型的参数
    best_model.save_parameters("best_model_parameters.pkl")
    print(f"Best model parameters saved to best_model_parameters.pkl")

def load_model_weights(pkl_file_path):
    # 加载.pkl权重文件
    with open(pkl_file_path, 'rb') as f:
        model_weights = pickle.load(f)
    return model_weights


def visualize_weights(model_weights):
    """
    可视化模型权重
    假设模型为三层MLP：输入层 -> 隐藏层 (512维) -> 输出层
    """
    
    layers = list(model_weights.keys())
    print("模型权重的层次结构：", layers)  # 查看权重的结构
    
    # 可视化隐藏层1 (输入到隐藏层的权重)
    weights_input_to_hidden = model_weights[layers[0]]['weight']  # 第一层的权重
    plt.figure(figsize=(10, 6))
    plt.title('Input to Hidden Layer Weights (Layer 1)')
    plt.imshow(weights_input_to_hidden, aspect='auto', cmap='viridis')
    plt.colorbar(label='Weight Value')
    plt.xlabel('Hidden Units')
    plt.ylabel('Input Features')
    plt.savefig('input_to_hidden_weights.png')
    
    # 可视化隐藏层权重 (隐藏层到隐藏层2，或者下一层权重)
    weights_hidden_to_output = model_weights[layers[1]]['weight']  # 第二层的权重
    plt.figure(figsize=(10, 6))
    plt.title('Hidden to Output Layer Weights (Layer 2)')
    plt.imshow(weights_hidden_to_output, aspect='auto', cmap='viridis')
    plt.colorbar(label='Weight Value')
    plt.xlabel('Output Units')
    plt.ylabel('Hidden Units')
    plt.savefig('hidden_to_output_weights.png')

