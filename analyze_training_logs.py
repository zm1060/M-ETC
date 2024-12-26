import re
import matplotlib.pyplot as plt

# 用于存储各个模型的训练和验证指标
models_data = {}

# 读取 training.log 文件
log_file_path = 'logs/doh_dataset_all_model-training.log'  # 根据实际文件路径修改
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
selected_metrics = input("Enter metrics to plot (comma-separated, or leave blank to plot all): ").strip()
if selected_metrics:
    selected_metrics = [m.strip() for m in selected_metrics.split(',')]
else:
    selected_metrics = all_metrics  # 默认绘制所有指标

# 用户选择绘图模式
print("\nChoose plotting mode:")
print("1. Both Train and Validation")
print("2. Only Validation")
mode = input("Enter your choice (1 or 2): ").strip()

# 绘图：为每个模型分别绘制选定指标的曲线
plt.figure(figsize=(16, 10))

# 使用颜色映射
color_list = plt.cm.tab10.colors  # 获取颜色列表

# 为每个模型绘制曲线
for idx, (model_name, data) in enumerate(models_data.items()):
    for metric in selected_metrics:
        train_metric_key = f"train_{metric}"
        val_metric_key = f"val_{metric}"
        if train_metric_key in data['metrics'] or val_metric_key in data['metrics']:
            if mode == "1" and train_metric_key in data['metrics']:  # 绘制训练曲线
                plt.plot(
                    data['epochs'],
                    data['metrics'][train_metric_key],
                    label=f"{model_name} Train {metric.title()}",
                    linestyle='-',
                    color=color_list[idx % len(color_list)]
                )
            if val_metric_key in data['metrics']:  # 绘制验证曲线
                plt.plot(
                    data['epochs'],
                    data['metrics'][val_metric_key],
                    label=f"{model_name} Val {metric.title()}",
                    linestyle='--',
                    color=color_list[idx % len(color_list)]
                )

# 添加图例、标题和标签
plt.title('Metrics Comparison for Different Models', fontsize=16)
plt.xlabel('Epoch', fontsize=14)
plt.ylabel('Metrics', fontsize=14)
plt.legend(fontsize=12, loc='best')
plt.grid(True)

# 显示图形
plt.tight_layout()
plt.show()
