import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import ParameterSampler
from model import BiLSTMModel
from data_preprocessing import load_data_from_directory, preprocess_data, get_dataloaders

# 设置随机种子以保证实验可重复性
np.random.seed(42)
torch.manual_seed(42)

# 加载和预处理数据
directory_path = '../csv_output/CIRA-CIC-DoHBrw-2020'
combined_data = load_data_from_directory(directory_path)
train_dataset, val_dataset, scaler, label_encoder = preprocess_data(combined_data)

# 定义参数搜索空间
param_dist = {
    'hidden_dim': [32, 64, 128],
    'lstm_layers': [1, 2, 3],
    'learning_rate': [1e-3, 1e-4, 1e-5],
    'batch_size': [32, 64, 128]
}

# 使用ParameterSampler从参数空间中随机选择组合
n_iter_search = 10  # 设定随机搜索的迭代次数
random_search = ParameterSampler(param_dist, n_iter=n_iter_search, random_state=42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 定义损失函数
criterion = nn.CrossEntropyLoss()

# 用于保存最佳参数和性能
best_val_accuracy = 0.0
best_params = None

# 搜索最佳超参数组合
for params in random_search:
    # 初始化模型
    numerical_dims = train_dataset[0][0].shape[0]
    model = BiLSTMModel(
        numerical_dims=numerical_dims,
        hidden_dim=params['hidden_dim'],
        lstm_layers=params['lstm_layers'],
        output_dim=len(label_encoder.classes_)
    )
    model.to(device)
    
    # 定义优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'])
    
    # 创建数据加载器
    train_loader, val_loader = get_dataloaders(train_dataset, val_dataset, batch_size=params['batch_size'])
    
    # 训练模型（可以根据需要定义多个 epoch 或进行简化的训练）
    for epoch in range(10):  # 简化训练过程，测试10个epoch
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

        # 验证模型
        model.eval()
        val_correct = 0
        total = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                _, predicted = torch.max(outputs, 1)
                total += y_batch.size(0)
                val_correct += (predicted == y_batch).sum().item()
        
        val_accuracy = val_correct / total
        print(f"Epoch {epoch+1}, Validation Accuracy: {val_accuracy:.4f}")
    
    # 如果当前组合的验证集准确率高于之前的记录，则更新最佳参数
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        best_params = params

print(f"Best parameters: {best_params}, Best validation accuracy: {best_val_accuracy:.4f}")
