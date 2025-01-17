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

# 1. 添加正则表达式
ml_train_time_pattern = r"Train time: ([\d.]+) seconds"
ml_inference_time_pattern = r"Inference time: ([\d.]+) seconds"
ml_accuracy_pattern = r"'train':.*'accuracy': ([\d.]+).*'loss': ([\d.]+)"
ml_val_accuracy_pattern = r"'val':.*'accuracy': ([\d.]+).*'loss': ([\d.]+)"

dl_train_time_pattern = r"Train time: ([\d.]+) seconds"
dl_inference_time_pattern = r"Inference time: ([\d.]+) seconds"
dl_epoch_accuracy_pattern = r"Epoch (\d+)/\d+, Train Metrics:.*'accuracy': ([\d.]+).*'loss': ([\d.]+)"
dl_val_accuracy_pattern = r"Epoch (\d+)/\d+, Validation Metrics:.*'accuracy': ([\d.]+).*'loss': ([\d.]+)"

# 2. 遍历日志文件提取信息
data = []
for log_file in log_files:
    log_path = os.path.join(logs_dir, log_file)

    with open(log_path, 'r') as file:
        log_content = file.read()

    # 机器学习模型日志处理
    if any(model in log_file for model in ['RandomForest', 'XGBoost', 'LogisticRegression', 'AdaBoost', 'DecisionTree', 'NaiveBayes', 'LDA', 'ExtraTrees', 'CatBoost', 'LightGBM']):
        train_time = re.findall(ml_train_time_pattern, log_content)
        inference_time = re.findall(ml_inference_time_pattern, log_content)
        accuracy_matches = re.findall(ml_accuracy_pattern, log_content)
        val_accuracy_matches = re.findall(ml_val_accuracy_pattern, log_content)

        if len(train_time) >= 5 and len(inference_time) >= 5 and len(accuracy_matches) >= 5 and len(val_accuracy_matches) >= 5:
            for fold_idx in range(5):
                train_accuracy, train_loss = accuracy_matches[fold_idx]
                val_accuracy, val_loss = val_accuracy_matches[fold_idx]

                data.append({
                    'log_file': log_file,
                    'fold': fold_idx + 1,
                    'train_accuracy': float(train_accuracy),
                    'train_loss': float(train_loss),
                    'val_accuracy': float(val_accuracy),
                    'val_loss': float(val_loss),
                    'train_time': float(train_time[fold_idx]),
                    'inference_time': float(inference_time[fold_idx]),
                    'epoch': None  # 机器学习模型没有 epoch 的概念
                })

    # 深度学习模型日志处理
    elif "Epoch" in log_content:
        epochs_data = re.findall(dl_epoch_accuracy_pattern, log_content)
        val_accuracy_data = re.findall(dl_val_accuracy_pattern, log_content)
        train_time = re.findall(dl_train_time_pattern, log_content)
        inference_time = re.findall(dl_inference_time_pattern, log_content)

        if len(train_time) >= 5 and len(inference_time) >= 5 and len(epochs_data) >= len(val_accuracy_data):
            for fold_idx in range(5):
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
        else:
            print(f"Warning: Incomplete data in {log_file}")

# 3. 转为 DataFrame
df = pd.DataFrame(data)
df.sort_values(by=["fold", "epoch"], inplace=True)
# 检查数据是否提取正确
print(df.head())

# 4. 绘制训练时间、推断时间与验证准确率对比
plt.figure(figsize=(10, 6))
for model in df['log_file'].unique():
    model_data = df[df['log_file'] == model]
    avg_train_time = model_data.groupby('fold')['train_time'].mean()
    avg_inference_time = model_data.groupby('fold')['inference_time'].mean()
    avg_val_accuracy = model_data.groupby('fold')['val_accuracy'].mean()

    # 绘制时间对比图
    plt.plot(range(1, 6), avg_train_time, label=f"{model} Train Time", marker='o')
    plt.plot(range(1, 6), avg_inference_time, label=f"{model} Inference Time", marker='x')

    # 绘制验证准确率
    plt.plot(range(1, 6), avg_val_accuracy, label=f"{model} Validation Accuracy", linestyle='--', marker='s')

plt.xlabel('Fold')
plt.ylabel('Metrics')
plt.title('Training Time, Inference Time, and Validation Accuracy')
plt.legend()
plt.savefig('training_inference_accuracy_comparison.jpg',format='jpg', dpi=300, bbox_inches='tight')
plt.show()

