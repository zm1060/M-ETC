import torch
import torch.nn as nn
from data_preprocessing import load_data_from_directory, preprocess_data, get_dataloaders
from model import BiLSTMModel
import logging

# 配置logging
logging.basicConfig(
    filename='training.log',  # 日志文件名
    filemode='a',  # 'a' 代表追加模式，'w' 代表覆盖
    format='%(asctime)s - %(levelname)s - %(message)s',  # 日志格式
    level=logging.INFO  # 设置日志记录级别为INFO
)

# 加载和预处理数据
directory_path = '../csv_output/CIRA-CIC-DoHBrw-2020'
combined_data = load_data_from_directory(directory_path)

# 预处理数据
train_dataset, val_dataset, scaler, label_encoder = preprocess_data(combined_data)

# 创建数据加载器
train_loader, val_loader = get_dataloaders(train_dataset, val_dataset, batch_size=64)

# 定义模型参数
numerical_dims = train_dataset[0][0].shape[0]  # 仅保留数值特征的维度

# 初始化模型
model = BiLSTMModel(
    numerical_dims=numerical_dims,  # 数值特征维度
    hidden_dim=64,
    lstm_layers=2,
    output_dim=len(label_encoder.classes_)  # 使用 label_encoder 来获取标签类别数
)

# 将模型移动到GPU（如果可用）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)  # 调整学习率为1e-4

# 训练循环
num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch in train_loader:
        X_batch, y_batch = [b.to(device) for b in batch]
        optimizer.zero_grad()
        outputs = model(X_batch)  # 模型前向传播
        
        # 打印部分输出以调试
        if epoch == 0 and total_loss == 0:
            print(f"Sample output: {outputs[:5].cpu().data.numpy()}")
        
        loss = criterion(outputs, y_batch)  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    logging.info(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_loss:.4f}")

    # 验证模型
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in val_loader:
            X_batch, y_batch = [b.to(device) for b in batch]
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()

    val_accuracy = correct / total
    logging.info(f"Epoch {epoch+1}/{num_epochs}, Val Loss: {val_loss/len(val_loader):.4f}, Val Accuracy: {val_accuracy:.4f}")

    # 每5个epoch保存一次模型
    if epoch % 5 == 0:
        torch.save(model.state_dict(), f'model_epoch_{epoch+1}.pth')
