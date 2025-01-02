import re
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

# 用于存储各个模型的训练和验证指标
models_data = {}

# 读取 training.log 文件
log_file_path = 'logs/CNN_BiGRU_Attention.log'  # 根据实际文件路径修改
with open(log_file_path, 'r') as file:
    log_data = file.read()

# 正则表达式：提取模型名称
model_switches = re.findall(r"INFO - Using model type: ([\w_]+)", log_data)

# 正则表达式：提取每个epoch的所有指标
epoch_data = re.findall(
    r"Epoch (\d+)/\d+.*?Train Metrics.*?({.*?}).*?Validation Metrics.*?({.*?})",
    log_data,
    re.DOTALL,
)

# 初始化当前模型名称
current_model_name = None

# 逐行解析日志并将指标数据添加到相应的模型中
for match in epoch_data:
    epoch, train_metrics_raw, val_metrics_raw = match
    epoch = int(epoch)

    # 将训练和验证的 JSON 字符串转换为字典
    train_metrics = eval(train_metrics_raw)
    val_metrics = eval(val_metrics_raw)

    # 移除非标量数据
    train_metrics = {k: v for k, v in train_metrics.items() if isinstance(v, (int, float))}
    val_metrics = {k: v for k, v in val_metrics.items() if isinstance(v, (int, float))}

    # 检查是否有模型切换
    if model_switches:
        if current_model_name is None or epoch == 1:  # 切换到新模型
            current_model_name = model_switches.pop(0)
            if current_model_name not in models_data:
                models_data[current_model_name] = {
                    'epochs': [],
                    'metrics': {}
                }

    # 初始化模型的指标容器
    if epoch == 1:
        for metric in train_metrics.keys():
            if metric not in models_data[current_model_name]['metrics']:
                models_data[current_model_name]['metrics'][f'train_{metric}'] = []
                models_data[current_model_name]['metrics'][f'val_{metric}'] = []

    # 添加数据
    models_data[current_model_name]['epochs'].append(epoch)
    for metric, value in train_metrics.items():
        models_data[current_model_name]['metrics'][f'train_{metric}'].append(value)
    for metric, value in val_metrics.items():
        models_data[current_model_name]['metrics'][f'val_{metric}'].append(value)

# 列出所有可用指标，并过滤非标量指标（去掉 train/val 前缀）
all_metrics = set()
for model_data in models_data.values():
    for k, v in model_data['metrics'].items():
        base_metric = k.split('_', 1)[1]  # 去掉 train/val 前缀
        if all(isinstance(x, (int, float)) for x in v):
            all_metrics.add(base_metric)
all_metrics = sorted(all_metrics)  # 排序以方便选择

print("Available metrics for plotting (without train/val distinction):")
print(all_metrics)

# 用户选择要绘制的指标
# selected_metrics = input("Enter metrics to plot (comma-separated, or leave blank to plot all): ").strip()
selected_metrics = ""
if selected_metrics:
    selected_metrics = [m.strip() for m in selected_metrics.split(',')]
else:
    selected_metrics = all_metrics  # 默认绘制所有指标

# 用户选择绘图模式
print("\nChoose plotting mode:")
print("1. Both Train and Validation")
print("2. Only Validation")
# mode = input("Enter your choice (1 or 2): ").strip()
mode = 1

# 在绘图部分之前添加样式设置

# 设置matplotlib的全局样式
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'axes.grid': True,
    'grid.alpha': 0.2,  # 降低网格线的透明度
    'grid.linestyle': '--',
    'figure.facecolor': 'white',  # 设置图形背景为白色
    'axes.facecolor': 'white',    # 设置坐标轴区域背景为白色
})

# 为每个指标创建单独的图表
for metric in selected_metrics:
    # 创建新的图形
    plt.figure(figsize=(12, 7))
    ax = plt.gca()
    
    # 创建颜色循环
    colors = plt.cm.Dark2(np.linspace(0, 1, len(models_data)))
    
    # 为每个模型绘制曲线
    for model_idx, (model_name, data) in enumerate(models_data.items()):
        train_metric_key = f"train_{metric}"
        val_metric_key = f"val_{metric}"
        color = colors[model_idx]
        
        # 绘制训练曲线
        if train_metric_key in data['metrics']:
            epochs = data['epochs']
            values = data['metrics'][train_metric_key]
            if len(epochs) == len(values):
                plt.plot(epochs, values,
                        label=f"{model_name} (Train)",
                        linestyle='-',
                        color=color,
                        linewidth=2.5,
                        marker='o',
                        markersize=6,
                        markerfacecolor='white',
                        markeredgewidth=2,
                        alpha=0.8)
        
        # 绘制验证曲线
        if val_metric_key in data['metrics']:
            epochs = data['epochs']
            values = data['metrics'][val_metric_key]
            if len(epochs) == len(values):
                plt.plot(epochs, values,
                        label=f"{model_name} (Val)",
                        linestyle='--',
                        color=color,
                        linewidth=2.5,
                        marker='s',
                        markersize=6,
                        markerfacecolor='white',
                        markeredgewidth=2,
                        alpha=1.0)
    
    # 设置图表样式
    title = f'{metric.upper()} Comparison Across Models'
    plt.title(title, pad=20, fontweight='bold', fontsize=16)
    plt.xlabel('Epoch', fontweight='bold')
    plt.ylabel(metric.title(), fontweight='bold')
    
    # 优化网格
    ax.grid(True, linestyle='--', alpha=0.2)  # 降低网格线的透明度
    ax.set_axisbelow(True)  # 将网格线置于数据下方
    
    # 移除顶部和右侧边框
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # 添加图例
    legend = plt.legend(bbox_to_anchor=(1.02, 1),
                       loc='upper left',
                       borderaxespad=0,
                       frameon=True,
                       fancybox=True,
                       shadow=True,
                       fontsize=10)
    
    # 添加轻微的边框效果
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
        spine.set_color('#4a4a4a')
    
    # 优化刻度
    ax.tick_params(axis='both', which='major', labelsize=10)
    
    # 如果是准确率或损失等特定指标，设置y轴范围
    if 'acc' in metric.lower() or 'accuracy' in metric.lower():
        plt.ylim(0, 1.05)
    elif 'loss' in metric.lower():
        plt.ylim(bottom=0)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存每个指标的图表
    plt.savefig(f'training_metrics_{metric}.png',
                bbox_inches='tight',
                dpi=300,
                facecolor='white',
                edgecolor='none')
    
    plt.close()  # 关闭当前图表，避免内存占用
