import torch
import torch.nn as nn
import torch.optim as optim
import logging
from data_processing import load_data
from model import LSTMModel
from training import train_model, evaluate_model
from hyperparameter_tuning import grid_search

# 设置日志记录
logging.basicConfig(filename='experiment_log.txt', level=logging.INFO, format='%(asctime)s - %(message)s')

# 数据集目录
datasets_dirs = {
    'Collection': '../meter/csv_output/Collection',
    'FiveWeek': '../meter/csv_output/FiveWeek',
    'RealWorld': '../meter/csv_output/RealWorld',
    'tunnel': '../meter/csv_output/tunnel'
}

# 加载数据
data_loaders = load_data(datasets_dirs)

# 确定 input_size
input_size = None
for dataset_name in data_loaders:
    if 'train' in data_loaders[dataset_name]:
        sample_input, _ = next(iter(data_loaders[dataset_name]['train']))
        input_size = sample_input.shape[1]
        break

if input_size is None:
    raise ValueError('未找到有效的数据集进行训练。')

# 超参数
param_grid = {
    'hidden_size': [64, 128],
    'num_layers': [1, 2],
    'learning_rate': [0.001, 0.0001],
    'num_epochs': [5, 10]
}

# 初始化设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 在 'Collection' 数据集上进行网格搜索
if 'Collection' in data_loaders:
    train_loader = data_loaders['Collection']['train']
    test_loader = data_loaders['Collection']['test']

    best_params, best_accuracy = grid_search(train_loader, test_loader, LSTMModel, param_grid, 3, input_size, device)

    # 记录最佳参数和准确率
    logging.info(f"最佳参数: {best_params}, 准确率: {best_accuracy:.2f}%")

# 使用最佳参数训练模型
model = LSTMModel(input_size, 
                  hidden_size=best_params['hidden_size'], 
                  num_layers=best_params['num_layers'], 
                  num_classes=3).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=best_params['learning_rate'])
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

# 训练最佳模型
train_model(model, train_loader, criterion, optimizer, scheduler, num_epochs=best_params['num_epochs'], device=device)

# 在所有数据集上评估模型
for dataset_name in data_loaders:
    if 'test' in data_loaders[dataset_name]:
        accuracy = evaluate_model(model, data_loaders[dataset_name]['test'], device)
        logging.info(f'{dataset_name} 数据集上的准确率: {accuracy:.2f}%')
        print(f'{dataset_name} 数据集上的准确率: {accuracy:.2f}%')
