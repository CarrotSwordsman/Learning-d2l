# mlp_scratch_debug.py
# 确保已安装: pip install torch torchvision matplotlib

import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils import data
import time
import matplotlib.pyplot as plt

# --- 1. 辅助函数 ---

def get_device():
    """尝试获取GPU，如果没有则使用CPU"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def relu(X):
    """手动实现ReLU激活函数"""
    a = torch.zeros_like(X)
    return torch.max(X, a)

class Accumulator:
    """用于累加多个变量的和"""
    def __init__(self, n):
        self.data = [0.0] * n
    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]
    def reset(self):
        self.data = [0.0] * len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]

def accuracy(y_hat, y):
    """计算预测正确的数量"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())

def evaluate_accuracy_gpu(net, data_iter, device=None):
    """在指定数据集上评估模型的精度 (GPU兼容)"""
    if isinstance(net, nn.Module):
        net.eval()  # 设置为评估模式
        if not device:
            device = next(iter(net.parameters())).device
    
    metric = Accumulator(2)  # 正确预测数、总预测数
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(X, list):
                # BERT之类的模型会接受额外的输入
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            y = y.to(device)
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]

def train_epoch_gpu(net, train_iter, loss, updater, device):
    """单个训练轮次的循环 (GPU兼容)"""
    if isinstance(net, nn.Module):
        net.train() # 设置为训练模式
    
    metric = Accumulator(3)  # 训练损失总和、训练准确度总和、样本数
    for X, y in train_iter:
        # 将数据移动到指定设备
        X, y = X.to(device), y.to(device)
        
        # 前向传播
        y_hat = net(X)
        l = loss(y_hat, y)
        
        # 反向传播和优化
        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            l.mean().backward()
            updater.step()
        else: # 自定义优化器 (来自 d2l scratch 实现)
             l.sum().backward()
             updater(X.shape[0]) # 传入批量大小

        metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
    # 返回训练损失和训练精度
    return metric[0] / metric[2], metric[1] / metric[2]

# --- 2. 数据加载函数 ---

def load_data_fashion_mnist(batch_size, resize=None, num_workers=4):
    """下载 Fashion-MNIST 数据集并加载到内存中"""
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    transform = transforms.Compose(trans)
    
    mnist_train = datasets.FashionMNIST(
        root="./data", train=True, transform=transform, download=True)
    mnist_test = datasets.FashionMNIST(
        root="./data", train=False, transform=transform, download=True)

    return (data.DataLoader(mnist_train, batch_size, shuffle=True, num_workers=num_workers),
            data.DataLoader(mnist_test, batch_size, shuffle=False, num_workers=num_workers))

# --- 3. 核心训练函数 (封装了参数初始化、网络定义和训练循环) ---

def train_mlp_scratch(hidden_dims, lr, num_epochs, device):
    """
    使用从零实现的风格训练一个具有指定隐藏层结构的MLP。
    
    Args:
        hidden_dims (list): 每个隐藏层的大小列表, e.g., [256] or [256, 128].
        lr (float): 学习率.
        num_epochs (int): 训练轮数.
        device (torch.device): 计算设备 (CPU or GPU).
    """
    print(f"\n--- 开始训练 MLP (从零实现) ---")
    print(f"结构: {num_inputs} -> {' -> '.join(map(str, hidden_dims))} -> {num_outputs}")
    print(f"学习率: {lr}, 轮数: {num_epochs}, 设备: {device}")
    
    # a. 参数初始化 (手动)
    params = []
    current_dim = num_inputs
    weight_params = [] # 用于优化器
    bias_params = []   # 用于优化器 (可以单独设置weight decay)
    
    # 创建隐藏层参数
    for i, h_dim in enumerate(hidden_dims):
        W = nn.Parameter(torch.randn(current_dim, h_dim, device=device, requires_grad=True) * 0.01)
        b = nn.Parameter(torch.zeros(h_dim, device=device, requires_grad=True))
        params.extend([W, b])
        weight_params.append(W)
        bias_params.append(b)
        current_dim = h_dim # 更新下一层的输入维度
        
    # 创建输出层参数
    W_out = nn.Parameter(torch.randn(current_dim, num_outputs, device=device, requires_grad=True) * 0.01)
    b_out = nn.Parameter(torch.zeros(num_outputs, device=device, requires_grad=True))
    params.extend([W_out, b_out])
    weight_params.append(W_out)
    bias_params.append(b_out)
    
    print("参数数量:", sum(p.numel() for p in params if p.requires_grad))

    # b. 模型定义 (手动, 使用闭包捕获上面定义的params)
    def net(X):
        X = X.reshape((-1, num_inputs))
        # 迭代应用隐藏层 (W 和 b 成对出现)
        for i in range(0, len(params) - 2, 2): 
            W_layer = params[i]
            b_layer = params[i+1]
            X = relu(X @ W_layer + b_layer)
            
        # 应用输出层
        W_output_layer = params[-2]
        b_output_layer = params[-1]
        return X @ W_output_layer + b_output_layer

    # c. 损失函数
    loss = nn.CrossEntropyLoss(reduction='none') # reduction='none' for manual averaging if needed

    # d. 优化器 (使用上面创建的参数列表)
    updater = torch.optim.SGD(params, lr=lr) 
    # 或者分开设置 weight decay:
    # updater = torch.optim.SGD([
    #     {'params': weight_params, 'weight_decay': wd}, # 假设 wd 已定义
    #     {'params': bias_params}], lr=lr)

    # e. 训练循环
    train_losses, train_accs, test_accs = [], [], []
    start_time = time.time()
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        train_loss, train_acc = train_epoch_gpu(net, train_iter, loss, updater, device)
        test_acc = evaluate_accuracy_gpu(net, test_iter, device)
        
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)
        
        epoch_time = time.time() - epoch_start_time
        print(f'轮次 {epoch + 1}/{num_epochs}, '
              f'训练损失 {train_loss:.3f}, 训练精度 {train_acc:.3f}, '
              f'测试精度 {test_acc:.3f}, '
              f'耗时 {epoch_time:.2f} 秒')

    total_time = time.time() - start_time
    print(f"训练完成: 最终训练精度={train_accs[-1]:.4f}, 最终测试精度={test_accs[-1]:.4f}")
    print(f"总耗时: {total_time:.2f} 秒")

    # f. 绘图
    plt.figure(figsize=(6, 4))
    epochs_range = range(1, num_epochs + 1)
    plt.plot(epochs_range, train_losses, label='训练损失 (Train Loss)')
    plt.plot(epochs_range, train_accs, label='训练精度 (Train Acc)')
    plt.plot(epochs_range, test_accs, label='测试精度 (Test Acc)')
    plt.xlabel('轮次 (Epoch)')
    plt.ylabel('值')
    plt.ylim([0.0, 1.0]) # 根据需要调整Y轴范围
    plt.title(f'MLP (从零实现) - 结构: {[num_inputs] + hidden_dims + [num_outputs]}')
    plt.legend()
    plt.grid(True)
    plt.show()

# --- 4. 主执行部分 ---

if __name__ == "__main__":
    # a. 设置全局参数
    num_inputs, num_outputs = 784, 10
    batch_size = 256
    num_epochs = 10  # 训练轮数
    lr = 0.1       # 学习率

    # b. 获取设备
    device = get_device()
    print(f"将使用设备: {device}")

    # c. 加载数据
    print("正在加载 Fashion-MNIST 数据集...")
    train_iter, test_iter = load_data_fashion_mnist(batch_size, num_workers=4) # num_workers 根据机器调整
    print("数据集加载完成.")

    # d. 定义要实验的隐藏层结构列表
    # hidden_dims_list = [[32], [64], [128], [256], [512]] # 比较不同宽度
    hidden_dims_list = [[256], [256, 128]]             # 比较单层 vs 双层 (如之前讨论)
    # hidden_dims_list = [[256, 128]]                   # 只运行双层
    
    # e. 循环执行训练和评估
    for h_dims in hidden_dims_list:
        train_mlp_scratch(hidden_dims=h_dims, lr=lr, num_epochs=num_epochs, device=device)

    print("\n--- 所有实验完成 ---")