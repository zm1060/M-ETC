import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
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

# # 确定 input_size
# input_size = None
# for dataset_name in data_loaders:
#     if 'train' in data_loaders[dataset_name]:
#         sample_input, _ = next(iter(data_loaders[dataset_name]['train']))
#         input_size = sample_input.shape[1]
#         break

# if input_size is None:
#     raise ValueError('未找到有效的数据集进行训练。')

# # 超参数
# param_grid = {
#     'hidden_size': [64, 128],
#     'num_layers': [1, 2],
#     'learning_rate': [0.001, 0.0001],
#     'num_epochs': [5, 10]
# }

# # 初始化设备
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# # 在 'Collection' 数据集上进行网格搜索
# if 'Collection' in data_loaders:
#     train_loader = data_loaders['Collection']['train']
#     test_loader = data_loaders['Collection']['test']

#     best_params, best_accuracy = grid_search(train_loader, test_loader, LSTMModel, param_grid, 3, input_size, device)

#     # 记录最佳参数和准确率
#     logging.info(f"最佳参数: {best_params}, 准确率: {best_accuracy:.2f}%")

# # 使用最佳参数训练模型
# model = LSTMModel(input_size, 
#                   hidden_size=best_params['hidden_size'], 
#                   num_layers=best_params['num_layers'], 
#                   num_classes=3).to(device)

# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=best_params['learning_rate'])
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

# # 训练最佳模型
# train_model(model, train_loader, criterion, optimizer, scheduler, num_epochs=best_params['num_epochs'], device=device)



# # 在所有数据集上评估模型
# for dataset_name in data_loaders:
#     if 'test' in data_loaders[dataset_name]:
#         accuracy = evaluate_model(model, data_loaders[dataset_name]['test'], device)
#         logging.info(f'{dataset_name} 数据集上的准确率: {accuracy:.2f}%')
#         print(f'{dataset_name} 数据集上的准确率: {accuracy:.2f}%')



def train_and_evaluate_rf(train_loader, test_loader):
    # 将数据转换为NumPy数组以适应RandomForestClassifier
    train_data = []
    train_labels = []
    test_data = []
    test_labels = []

    for inputs, labels in train_loader:
        train_data.append(inputs.numpy())
        train_labels.append(labels.numpy())
    
    for inputs, labels in test_loader:
        test_data.append(inputs.numpy())
        test_labels.append(labels.numpy())

    # 转换为NumPy数组格式
    train_data = np.vstack(train_data)
    train_labels = np.concatenate(train_labels)
    test_data = np.vstack(test_data)
    test_labels = np.concatenate(test_labels)

    # 创建并训练随机森林模型
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(train_data, train_labels)
    # 保存模型
    joblib.dump(rf_model, f'random_forest_model_{dataset_name}.pkl')  # 为每个数据集保存模型

    # 在测试集上进行预测
    rf_predictions = rf_model.predict(test_data)

    # 计算随机森林的评估指标
    rf_accuracy = accuracy_score(test_labels, rf_predictions)
    rf_precision = precision_score(test_labels, rf_predictions, average='weighted', zero_division=0)
    rf_recall = recall_score(test_labels, rf_predictions, average='weighted', zero_division=0)
    rf_f1 = f1_score(test_labels, rf_predictions, average='weighted', zero_division=0)

    # 记录和打印随机森林的评估结果
    logging.info(f'随机森林模型的准确率: {rf_accuracy:.2f}%')
    logging.info(f'随机森林模型的精确率: {rf_precision:.2f}')
    logging.info(f'随机森林模型的召回率: {rf_recall:.2f}')
    logging.info(f'随机森林模型的F1分数: {rf_f1:.2f}')
    print(f'随机森林模型的准确率: {rf_accuracy:.2f}%')
    print(f'随机森林模型的精确率: {rf_precision:.2f}')
    print(f'随机森林模型的召回率: {rf_recall:.2f}')
    print(f'随机森林模型的F1分数: {rf_f1:.2f}')

# --- 基准模型：随机森林 ---
# 训练和评估所有数据集
for dataset_name in data_loaders:
    if 'train' in data_loaders[dataset_name] and 'test' in data_loaders[dataset_name]:
        print(f'正在处理数据集: {dataset_name}')
        train_loader = data_loaders[dataset_name]['train']
        test_loader = data_loaders[dataset_name]['test']
        train_and_evaluate_rf(train_loader, test_loader)