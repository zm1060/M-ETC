import matplotlib.pyplot as plt
import numpy as np

# 原始数据（百分比转化为0到1范围）
original_metrics = {
    "CNN-BiGRU-Attention": [55.95 / 100, 63.77 / 100, 55.95 / 100, 48.40 / 100],  # Accuracy, Precision, Recall, F1
    "XGBoost": [52.47 / 100, 65.43 / 100, 52.47 / 100, 39.49 / 100],  # Accuracy, Precision, Recall, F1
}

# 新数据（已经是0到1范围）
new_metrics = {
    "CNN-BiGRU-Attention": {
        "10%": [0.6886889068049593, 0.7141619699098366, 0.6886889068049593, 0.6786442325746831],
        "20%": [0.7003744735424948, 0.7268500332884054, 0.7003744735424948, 0.6909090086890816],
        "30%": [0.7301300229415628, 0.7783681582294042, 0.7301300229415628, 0.7183496416447973],
    },
    "XGBoost": {
        "10%": [0.7822699384593356, 0.8147099456554305, 0.7822699384593356, 0.7767582960838699],
        "20%": [0.788056690874114, 0.8223310665942062, 0.788056690874114, 0.7825102265524766],
        "30%": [0.7912846963900502, 0.8269635321721405, 0.7912846963900502, 0.7856702598891759],
    }
}

# 指标标签
metrics_labels = ["Accuracy", "Precision", "Recall", "F1-score"]

# 计算相对提升（百分比）
improvements = {}
for model in new_metrics:
    improvements[model] = {}
    for dataset in new_metrics[model]:
        improvements[model][dataset] = []
        for i in range(4):  # 4个指标
            original = original_metrics[model][i]
            new = new_metrics[model][dataset][i]
            relative_improvement = ((new - original) / original) * 100
            improvements[model][dataset].append(relative_improvement)

def plot_improvements():
    # plt.style.use('seaborn')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    # fig.suptitle('Relative Performance Improvements (%)', fontsize=16, y=0.95)
    
    # 扁平化axes数组以便迭代
    axes = axes.ravel()
    
    # 设置数据
    datasets = ["10%", "20%", "30%"]
    x = np.arange(len(datasets))
    width = 0.35
    
    # 为每个指标创建子图
    for idx, metric in enumerate(metrics_labels):
        ax = axes[idx]
        
        # 提取当前指标的数据
        cnn_data = [improvements["CNN-BiGRU-Attention"][dataset][idx] for dataset in datasets]
        xgb_data = [improvements["XGBoost"][dataset][idx] for dataset in datasets]
        
        # 绘制条形图
        rects1 = ax.bar(x - width/2, cnn_data, width, label='CNN-BiGRU-Attention', 
                       color='#1f77b4', alpha=0.7)
        rects2 = ax.bar(x + width/2, xgb_data, width, label='XGBoost',
                       color='#2ca02c', alpha=0.7)
        
        # 设置图表属性
        ax.set_title(f'{metric}', fontsize=12)
        ax.set_xlabel('Additional Data Size')
        ax.set_ylabel('Improvement (%)')
        ax.set_xticks(x)
        ax.set_xticklabels(datasets)
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # 添加数值标签
        def autolabel(rects):
            for rect in rects:
                height = rect.get_height()
                ax.annotate(f'{height:.1f}%',
                          xy=(rect.get_x() + rect.get_width() / 2, height),
                          xytext=(0, 3),  # 3 points vertical offset
                          textcoords="offset points",
                          ha='center', va='bottom', rotation=0)
        
        autolabel(rects1)
        autolabel(rects2)
    
    plt.tight_layout()
    plt.savefig('performance_improvements_relative.jpg', format='jpg', bbox_inches='tight', dpi=300)
    plt.show()

# 调用可视化函数
plot_improvements()
