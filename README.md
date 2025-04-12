# README

This README provides step-by-step instructions for downloading the model weights, setting up the environment, and running the `mlp.py` script to train and evaluate an MLP model for CIFAR-10 classification (or FashionMNIST classification, with minor adjustments).

## 1. **Pre-requisites**

Before proceeding, ensure you have:
- An internet connection (for downloading required datasets if necessary).
- Python installed (version 3.7 or above).
- GPU support configured (optional, but recommended for faster training).

## 2. **Setup Environment**

### 2.1. **Create a Virtual Environment**
To avoid dependency conflicts, it’s recommended to set up a virtual environment.

```bash
# Create a virtual environment
python -m venv mlp_env

# Activate the environment
# Windows
mlp_env\Scripts\activate
# macOS/Linux
source mlp_env/bin/activate
```

### 2.2. Install dependencies

```bash
pip install numpy matplotlib 
```

------

## 3. **Download Pretrained Weights (Optional)**

```
Location:	https://pan.baidu.com/s/1pABeitOSwWz9sciPaDKCEQ  
Code:	dagt 
```

If you want to load a pretrained model for evaluation directly (without training), download the pretrained weights file and place it in the same directory as `mlp.py`. You can use the `load_parameters()` method to load the weights.

## 4. **Run !**

```bash
python MLP.py
```

