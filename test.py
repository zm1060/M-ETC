import os
import re
import pandas as pd
import matplotlib.pyplot as plt

# 1. 获取所有日志文件
logs_dir = './logs'  # 日志文件的目录

# 获取 logs 目录下所有的 .log 文件
log_files = [f for f in os.listdir(logs_dir) if f.endswith('.log')]

# 用于存储所有模型的训练数据
data = []

# 2. 正则表达式模式，用于匹配所需的日志数据

# 机器学习模型的正则表达式（适用于：RandomForest, XGBoost, LogisticRegression, AdaBoost, DecisionTree, NaiveBayes, LDA, ExtraTrees, CatBoost, LightGBM）
ml_train_time_pattern = r"Train time: ([\d.]+) seconds"
ml_inference_time_pattern = r"Inference time: ([\d.]+) seconds"
ml_accuracy_pattern = r"'train':.*'accuracy': ([\d.]+).*'loss': ([\d.]+)"
ml_val_accuracy_pattern = r"'val':.*'accuracy': ([\d.]+).*'loss': ([\d.]+)"

# 深度学习模型的正则表达式（例如：BiGRU, CNN, etc.）
dl_train_time_pattern = r"Train time: ([\d.]+) seconds"
dl_inference_time_pattern = r"Inference time: ([\d.]+) seconds"
dl_epoch_accuracy_pattern = r"Epoch (\d+)/\d+, Train Metrics:.*'accuracy': ([\d.]+).*'loss': ([\d.]+)"
dl_val_accuracy_pattern = r"Epoch (\d+)/\d+, Validation Metrics:.*'accuracy': ([\d.]+).*'loss': ([\d.]+)"

# 3. 遍历所有日志文件并提取数据
for log_file in log_files:
    log_path = os.path.join(logs_dir, log_file)

    with open(log_path, 'r') as file:
        log_content = file.read()

    # 1) 处理机器学习模型日志
    if any(model in log_file for model in ['RandomForest', 'XGBoost', 'LogisticRegression', 'AdaBoost', 'DecisionTree', 'NaiveBayes', 'LDA', 'ExtraTrees', 'CatBoost', 'LightGBM']):
        # 提取每一折训练和推理时间
        train_time = re.findall(ml_train_time_pattern, log_content)
        inference_time = re.findall(ml_inference_time_pattern, log_content)

        # 提取每一折的训练精度和验证精度
        accuracy_matches = re.findall(ml_accuracy_pattern, log_content)
        val_accuracy_matches = re.findall(ml_val_accuracy_pattern, log_content)

        # 如果匹配项有多个
        for fold_idx in range(5):
            train_accuracy, train_loss = accuracy_matches[fold_idx]  # 提取每一折的训练精度与损失
            val_accuracy, val_loss = val_accuracy_matches[fold_idx]  # 提取每一折的验证精度与损失

            data.append({
                'log_file': log_file,
                'fold': fold_idx + 1,  # 1到5的fold编号
                'train_accuracy': float(train_accuracy),
                'train_loss': float(train_loss),
                'val_accuracy': float(val_accuracy),
                'val_loss': float(val_loss),
                'train_time': float(train_time[fold_idx]),
                'inference_time': float(inference_time[fold_idx]),
            })

    # 2) 处理深度学习模型日志
    elif "Epoch" in log_content:
        # 提取每个 epoch 的训练和验证精度
        epochs_data = re.findall(dl_epoch_accuracy_pattern, log_content)
        val_accuracy_data = re.findall(dl_val_accuracy_pattern, log_content)
        
        # 提取训练时间和推理时间
        train_time = re.findall(dl_train_time_pattern, log_content)
        inference_time = re.findall(dl_inference_time_pattern, log_content)

        # 添加每个 epoch 的数据，每次循环为一折（fold）
        for fold_idx in range(5):  # 处理5-fold
            for epoch_idx, (epoch, train_acc, train_loss) in enumerate(epochs_data):
                val_acc = val_accuracy_data[epoch_idx][1]
                data.append({
                    'log_file': log_file,
                    'fold': fold_idx + 1,
                    'epoch': int(epoch),
                    'train_accuracy': float(train_acc),
                    'train_loss': float(train_loss),
                    'val_accuracy': float(val_acc),
                    'train_time': float(train_time[fold_idx]),
                    'inference_time': float(inference_time[fold_idx])
                })

# 4. 将数据存储到 DataFrame
df = pd.DataFrame(data)

# 打印前几行数据，确认提取正确
print(df.head())

# 5. 绘制图表：多个模型对比

# 绘制训练与验证精度对比
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
for model in df['log_file'].unique():
    model_data = df[df['log_file'] == model]
    # 对5个fold的训练和验证精度求平均
    avg_train_accuracy = model_data.groupby('fold')['train_accuracy'].mean()
    avg_val_accuracy = model_data.groupby('fold')['val_accuracy'].mean()
    plt.plot(range(1, 6), avg_train_accuracy, label=f"{model} Train Accuracy", marker='o')
    plt.plot(range(1, 6), avg_val_accuracy, label=f"{model} Val Accuracy", marker='x')

plt.xlabel('Fold')
plt.ylabel('Accuracy')
plt.title('Train vs Validation Accuracy')
plt.legend()

# 绘制训练与验证损失对比
plt.subplot(1, 2, 2)
for model in df['log_file'].unique():
    model_data = df[df['log_file'] == model]
    # 对5个fold的训练和验证损失求平均
    avg_train_loss = model_data.groupby('fold')['train_loss'].mean()
    avg_val_loss = model_data.groupby('fold')['val_loss'].mean()
    plt.plot(range(1, 6), avg_train_loss, label=f"{model} Train Loss", marker='o')
    plt.plot(range(1, 6), avg_val_loss, label=f"{model} Val Loss", marker='x')

plt.xlabel('Fold')
plt.ylabel('Loss')
plt.title('Train vs Validation Loss')
plt.legend()

plt.tight_layout()
plt.show()

# 6. 绘制训练时间与推理时间对比
plt.figure(figsize=(6, 6))
for model in df['log_file'].unique():
    model_data = df[df['log_file'] == model]
    avg_train_time = model_data.groupby('fold')['train_time'].mean()
    avg_inference_time = model_data.groupby('fold')['inference_time'].mean()
    plt.plot(range(1, 6), avg_train_time, label=f"{model} Train Time", marker='o')
    plt.plot(range(1, 6), avg_inference_time, label=f"{model} Inference Time", marker='x')

plt.xlabel('Fold')
plt.ylabel('Time (seconds)')
plt.title('Training Time vs Inference Time')
plt.legend()
plt.show()

# 7. 展示某一fold的训练过程数据
fold_to_show = 3  # 选择展示的fold（比如展示fold 1的数据）

# 获取某一fold的数据
fold_data = df[df['fold'] == fold_to_show]

# 绘制训练过程的详细图表：每个epoch的训练精度与验证精度
plt.figure(figsize=(12, 6))

# 训练精度和损失
plt.subplot(1, 2, 1)
for model in fold_data['log_file'].unique():
    model_data = fold_data[fold_data['log_file'] == model]
    plt.plot(model_data['epoch'], model_data['train_accuracy'], label=f"{model} Train Accuracy", marker='o')
    plt.plot(model_data['epoch'], model_data['val_accuracy'], label=f"{model} Val Accuracy", marker='x')

plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title(f'Fold {fold_to_show}: Train vs Val Accuracy')
plt.legend()

# 训练损失与验证损失
plt.subplot(1, 2, 2)
for model in fold_data['log_file'].unique():
    model_data = fold_data[fold_data['log_file'] == model]
    plt.plot(model_data['epoch'], model_data['train_loss'], label=f"{model} Train Loss", marker='o')
    plt.plot(model_data['epoch'], model_data['val_loss'], label=f"{model} Val Loss", marker='x')

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title(f'Fold {fold_to_show}: Train vs Val Loss')
plt.legend()

plt.tight_layout()
plt.show()
