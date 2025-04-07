import numpy as np
import os
import gzip

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
