import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 数据
data = {
    'log_file': [
        'AdaBoost', 'BiGRU', 'BiLSTM', 'CNN', 'CNN_BiGRU_Attention', 'CNN_BiLSTM_Attention', 
        'CNN_GRU', 'CNN_GRU_Attention', 'CNN_LSTM', 'CNN_LSTM_Attention', 'CatBoost', 
        'DNN', 'DecisionTree', 'ExtraTrees', 'GRU', 'LDA', 'LSTM', 'LightGBM', 
        'LogisticRegression', 'MLP', 'NaiveBayes', 'RNN', 'RandomForest', 'XGBoost'
    ],
    'train_time': [
        289.762, 88.132, 93.726, 64.84, 98.576, 148.536, 85.648, 129.112, 93.074, 136.986, 
        114.622, 42.532, 111.626, 259.286, 59.082, 28.016, 65.262, 24.082, 33.38, 42.742, 
        3.778, 61.79, 665.56, 20.582
    ],
    'inference_time': [
        3.538, 8.878, 10.566, 7.412, 8.754, 11.218, 8.904, 12.75, 10.51, 12.092, 
        4.284, 5.716, 0.794, 8.264, 6.62, 1.642, 6.056, 2.668, 1.654, 5.088, 
        4.194, 7.624, 4.298, 2.162
    ]
}

df = pd.DataFrame(data)

# 按训练时间排序
df = df.sort_values(by='train_time', ascending=True)

# 数据准备
bar_width = 0.4
index = np.arange(len(df))

# 绘图
plt.figure(figsize=(16, 10))

# 绘制条形图
plt.barh(index - bar_width / 2, df['train_time'], bar_width, label='Training Time', color='blue', alpha=0.7)
plt.barh(index + bar_width / 2, df['inference_time'], bar_width, label='Inference Time', color='orange', alpha=0.7)

# 添加数值标签
for i, (train, infer) in enumerate(zip(df['train_time'], df['inference_time'])):
    plt.text(train + 5, i - bar_width / 2, f'{train:.1f}', va='center', fontsize=9)
    plt.text(infer + 1, i + bar_width / 2, f'{infer:.1f}', va='center', fontsize=9)

# 设置标签和标题
plt.yticks(index, df['log_file'], fontsize=10)
plt.xlabel('Time (seconds)', fontsize=12)
# plt.title('Comparison of Training and Inference Times', fontsize=16)
plt.legend(fontsize=12)

# 网格线和布局
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()

# 保存图表
plt.savefig('training_inference_comparison.jpg', format='jpg', dpi=300)

# 显示图表
plt.show()
