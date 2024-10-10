from sklearn.model_selection import ParameterGrid
import torch
import torch.nn as nn
import torch.optim as optim
from training import train_model, evaluate_model

def grid_search(train_data_loader, test_data_loader, model_class, param_grid, num_classes, input_size, device):
    best_accuracy = 0
    best_params = None

    for params in ParameterGrid(param_grid):
        print(f"正在测试参数: {params}")
        model = model_class(input_size=input_size, hidden_size=params['hidden_size'], num_layers=params['num_layers'], num_classes=num_classes).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)  # 创建调度器

        train_model(model, train_data_loader, criterion, optimizer, scheduler=scheduler, num_epochs=params['num_epochs'], device=device)
        accuracy = evaluate_model(model, test_data_loader, device)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_params = params

    print(f"最佳参数: {best_params}, 准确率: {best_accuracy}")
    return best_params, best_accuracy