import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.decomposition import PCA
from torch.utils.data import TensorDataset, DataLoader
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体为黑体
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# ======================================================
# 1. 数据加载
df = pd.read_csv("Concrete_Data_Yeh.csv")
print("="*30 + "数据集基本信息" + "="*30)
print(f"数据集形状：{df.shape}")
print("缺失值统计：", df.isnull().sum().sum())

# ======================================================
# 2. 高相关特征筛选
corr_matrix = df.corr()
corr_with_target = corr_matrix['csMPa'].sort_values(ascending=False)
print("\n" + "="*30 + "特征与强度相关性" + "="*30)
print(corr_with_target)

# 筛选阈值：绝对值高相关特征
CORR_THRESHOLD = 0.25
selected_features = corr_with_target[
    (abs(corr_with_target) > CORR_THRESHOLD) & 
    (corr_with_target.index != 'csMPa')
].index.tolist()

print("\n" + "="*30 + "筛选结果" + "="*30)
print(f"筛选高相关特征：{selected_features}")
print(f"特征数量：{len(selected_features)}")

# 绘制热力图
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', fmt='.2f')
plt.title('特征相关性热力图')
plt.tight_layout()
plt.savefig('相关性热力图.png', dpi=300)
plt.show()

# ======================================================
# 3. 主成分分析(PCA)
X = df.iloc[:, :-1].values
Y = df.iloc[:, -1].values.reshape(-1, 1)
feature_names = df.columns[:-1]
scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X_scaled)
print(f'PCA 后主成分数: {X_pca.shape[1]}')

cum_var = np.cumsum(pca.explained_variance_ratio_)
plt.figure(figsize=(6, 4))
plt.plot(np.arange(1, len(cum_var) + 1), cum_var, marker='o')
plt.axhline(0.95, color='r', linestyle='--', label='95% variance')
plt.xlabel('主成分数量')
plt.ylabel('累计解释方差')
plt.title('PCA 累积解释方差')
plt.legend()
plt.tight_layout()
plt.savefig('PCA累积解释方差.png', dpi=300)
plt.show()
# ======================================================
# 5. 数据划分与标准化
X_train, X_test, Y_train, Y_test = train_test_split(
    X_pca, Y, test_size=0.2, random_state=42
)
scaler_Y = StandardScaler()
Y_train_scaled = scaler_Y.fit_transform(Y_train)
Y_test_scaled = scaler_Y.transform(Y_test)
# ======================================================
# 6. 线性回归
lr = LinearRegression()
lr.fit(X_train, Y_train)
Y_pred_lr = lr.predict(X_test)
mse_lr = mean_squared_error(Y_test, Y_pred_lr)
r2_lr = r2_score(Y_test, Y_pred_lr)

print("\n" + "="*30 + "线性回归结果" + "="*30)
print(f"测试集MSE: {mse_lr:.2f}")
print(f"测试集R²: {r2_lr:.2f}")

# ======================================================
# 7. 优化神经网络
# 张量转换
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
Y_train_tensor = torch.tensor(Y_train_scaled, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
Y_test_tensor = torch.tensor(Y_test_scaled, dtype=torch.float32)

# 数据加载器
train_loader = DataLoader(TensorDataset(X_train_tensor, Y_train_tensor), batch_size=16, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test_tensor, Y_test_tensor), batch_size=16, shuffle=False)

class SimpleNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 64),  
            nn.ReLU(),         
            nn.Linear(64, 32),          
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        return self.layers(x)

# 初始化
model = SimpleNN(input_dim=X_pca.shape[1])
criterion = nn.MSELoss()
# L2正则化(weight_decay) + 调整学习率
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

# 训练
epochs = 500
train_loss = []
test_loss = []

print("\n" + "="*30 + "神经网络训练" + "="*30)
for epoch in range(epochs):
    model.train()
    total_train = 0
    for x, y in train_loader:
        pred = model(x)
        loss = criterion(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_train += loss.item() * x.size(0)
    
    
    # 记录损失
    train_l = total_train / len(X_train)
    train_loss.append(train_l)
    
    if (epoch+1) % 20 == 0:
        print(f"Epoch {epoch+1:3d} | 训练损失:{train_l:.4f} ")




# ======================================================
# 8. 模型评估
model.eval()
with torch.no_grad():
    y_pred_nn = scaler_Y.inverse_transform(model(X_test_tensor).numpy())
mse_nn = mean_squared_error(Y_test, y_pred_nn)
r2_nn = r2_score(Y_test, y_pred_nn)

print("\n" + "="*30 + "优化后神经网络结果" + "="*30)
print(f"测试集MSE: {mse_nn:.2f}")
print(f"测试集R²: {r2_nn:.2f}")

# ======================================================
# 9. 可视化
plt.figure(figsize=(15, 5))

plt.subplot(1,3,1)
plt.plot(train_loss, label='训练损失', c='blue')
plt.plot(test_loss, label='测试损失', c='red')
plt.xlabel('轮次')
plt.ylabel('MSE')
plt.title('训练/测试损失曲线')
plt.legend()

# 预测对比
plt.subplot(1,3,2)
plt.scatter(Y_test, y_pred_nn, alpha=0.6)
plt.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], 'r--')
plt.xlabel('真实值')
plt.ylabel('预测值')
plt.title('神经网络预测结果')

# 误差分布
plt.subplot(1,3,3)
plt.hist(Y_test - y_pred_nn, bins=20, edgecolor='black')
plt.xlabel('误差')
plt.title('误差分布')

plt.tight_layout()
plt.savefig('优化后结果.png', dpi=300)
plt.show()

# ======================================================
# 总结
print("\n" + "="*30 + "最终总结" + "="*30)
print(f" 线性回归 MSE: {mse_lr:.2f}")
print(f" 神经网络 MSE: {mse_nn:.2f}")