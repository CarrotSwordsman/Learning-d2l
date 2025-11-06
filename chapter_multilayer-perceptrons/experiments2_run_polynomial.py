import math
import numpy as np
import torch
from torch import nn
from d2l import torch as d2l
import matplotlib.pyplot as plt
import os

# --- 在脚本的同级目录下创建输出文件夹 ---
script_dir = os.path.dirname(os.path.abspath(__file__))
output_folder_name = "test_outputs"
output_dir_path = os.path.join(script_dir, output_folder_name)
if not os.path.exists(output_dir_path):
    os.makedirs(output_dir_path)
    print(f"Created directory: {output_dir_path}")
# --------------------------------------------------

# --- 数据生成（与Notebook中相同） ---
max_degree = 20
n_train, n_test = 100, 100
true_w = np.zeros(max_degree)
true_w[0:4] = np.array([5, 1.2, -3.4, 5.6])

# 为了可复现性，固定随机种子
np.random.seed(42)
torch.manual_seed(42)

features = np.random.normal(size=(n_train + n_test, 1))
np.random.shuffle(features)
poly_features = np.power(features, np.arange(max_degree).reshape(1, -1))
for i in range(max_degree):
    poly_features[:, i] /= math.gamma(i + 1)
labels = np.dot(poly_features, true_w)
labels += np.random.normal(scale=0.1, size=labels.shape)

true_w, features, poly_features, labels = [torch.tensor(x, dtype=
    torch.float32) for x in [true_w, features, poly_features, labels]]

# --- 辅助函数（与Notebook中相同） ---
def evaluate_loss(net, data_iter, loss):
    metric = d2l.Accumulator(2)
    net.eval()
    with torch.no_grad():
        for X, y in data_iter:
            out = net(X)
            y = y.reshape(out.shape)
            l = loss(out, y)
            metric.add(l.sum(), l.numel())
    return metric[0] / metric[1]

# --- 修改后的训练函数（不绘图，返回损失） ---
# 【【【*** 修正点 1：添加 lr 参数 ***】】】
def train_and_evaluate(train_features, test_features, train_labels, test_labels,
                       num_epochs=400, lr=0.01): # <--- 接收 lr 参数
    loss = nn.MSELoss(reduction='none') 
    input_shape = train_features.shape[-1]
    net = nn.Sequential(nn.Linear(input_shape, 1, bias=False))
    net.apply(lambda m: torch.nn.init.normal_(m.weight, std=0.01) if isinstance(m, nn.Linear) else None)
    
    batch_size = min(10, train_labels.shape[0])
    train_iter = d2l.load_array((train_features, train_labels.reshape(-1,1)),
                                batch_size)
    test_iter = d2l.load_array((test_features, test_labels.reshape(-1,1)),
                               batch_size, is_train=False)
    
    # 【【【*** 修正点 2：使用传入的 lr ***】】】
    trainer = torch.optim.SGD(net.parameters(), lr=lr) # <--- 使用 lr
    
    for epoch in range(num_epochs):
        net.train()
        for X, y in train_iter:
            l = loss(net(X), y.reshape(-1, 1))
            trainer.zero_grad()
            l.sum().backward() # 使用 .sum()
            trainer.step()
            
    train_loss = evaluate_loss(net, train_iter, loss)
    test_loss = evaluate_loss(net, test_iter, loss)
    return train_loss, test_loss

# --- 实验 2.1 & 2.2：损失 vs 模型复杂度 ---
print("Running experiment: Loss vs. Model Complexity...")
degrees = np.arange(1, max_degree + 1)
train_losses = []
test_losses = []

for d in degrees:
    # 【【【*** 修正点 3：动态调整 LR ***】】】
    # 对高阶模型使用更高的学习率，迫使它们训练
    current_lr = 0.1 if d > 4 else 0.01
    
    train_loss, test_loss = train_and_evaluate(
        poly_features[:n_train, :d], 
        poly_features[n_train:, :d],
        labels[:n_train], 
        labels[n_train:],
        lr=current_lr # 传入学习率
    )
    train_losses.append(train_loss) 
    test_losses.append(test_loss)
    
print("Plotting Loss vs. Model Complexity...")
plt.figure(figsize=(10, 5))
plt.plot(degrees - 1, train_losses, label='Train Loss', marker='o')
plt.plot(degrees - 1, test_losses, label='Test Loss', marker='o')
plt.xlabel('Polynomial Degree (Model Complexity)')
plt.ylabel('Loss (MSE)')
plt.title('Loss vs. Model Complexity (Degree)')
plt.legend()
plt.yscale('log')
plt.ylim(1e-2, 1e3) # 调高 ylim 上限
plt.xticks(np.arange(0, 20, 2))
plt.grid(True, which="both", ls="--")
save_path_1 = os.path.join(output_dir_path, 'loss_vs_complexity.png')
plt.savefig(save_path_1)
plt.close()
print(f"Saved plot to {save_path_1}")

# --- 实验 2.3：损失 vs 数据量 ---
print("Running experiment: Loss vs. Data Size...")
data_sizes = np.arange(20, n_train + 1, 5)
good_model_train_losses = []
good_model_test_losses = []
overfit_model_train_losses = []
overfit_model_test_losses = []

loss_fn = nn.MSELoss(reduction='none')

for n in data_sizes:
    # 训练 "Good" Model (Degree 3)
    train_loss_good, test_loss_good = train_and_evaluate(
        poly_features[:n, :4], 
        poly_features[n_train:, :4],
        labels[:n], 
        labels[n_train:],
        num_epochs=400,
        lr=0.01 # 简单模型使用 lr=0.01
    )
    good_model_train_losses.append(train_loss_good)
    good_model_test_losses.append(test_loss_good)
    
    # 训练 "Overfit" Model (Degree 19)
    train_loss_overfit, test_loss_overfit = train_and_evaluate(
        poly_features[:n, :],
        poly_features[n_train:, :],
        labels[:n], 
        labels[n_train:],
        num_epochs=1500,
        lr=0.1 # 【【【*** 修正点 4：复杂模型使用 lr=0.1 ***】】】
    )
    overfit_model_train_losses.append(train_loss_overfit)
    overfit_model_test_losses.append(test_loss_overfit)
    
    if n == 20 or n == 100:
        print(f"[Degree 19 Model, lr=0.1] n={n}: Train Loss={train_loss_overfit:.4f}, Test Loss={test_loss_overfit:.4f}")

print("Plotting Loss vs. Data Size...")
plt.figure(figsize=(12, 6))

# 图 (a): 正常拟合 (Degree 3)
plt.subplot(1, 2, 1)
plt.plot(data_sizes, good_model_train_losses, label='Train Loss', marker='o')
plt.plot(data_sizes, good_model_test_losses, label='Test Loss', marker='o')
plt.xlabel('Number of Training Samples')
plt.ylabel('Loss (MSE)')
plt.title('Degree 3 Model (Good Fit)')
plt.legend()
plt.yscale('log')
plt.ylim(1e-2, 1e3)
plt.grid(True, which="both", ls="--")

# 图 (b): 过拟合 (Degree 19)
plt.subplot(1, 2, 2)
plt.plot(data_sizes, overfit_model_train_losses, label='Train Loss', marker='o')
plt.plot(data_sizes, overfit_model_test_losses, label='Test Loss', marker='o')
plt.xlabel('Number of Training Samples')
plt.ylabel('Loss (MSE)')
plt.title('Degree 19 Model (Overfit)')
plt.legend()
plt.yscale('log')
plt.ylim(1e-2, 1e3) 
plt.grid(True, which="both", ls="--")

plt.tight_layout()
save_path_2 = os.path.join(output_dir_path, 'loss_vs_datasize.png')
plt.savefig(save_path_2)
plt.close()
print(f"Saved plot to {save_path_2}")

print("All experiments finished.")