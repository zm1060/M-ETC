from sklearn.model_selection import  KFold, ParameterGrid
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import  DataLoader

from data_processing import FlowDataset
from training import evaluate_model, train_model

# 交叉验证模型
def cross_validate_model(model_class, dataset, param_grid, num_folds=5):
    kfold = KFold(n_splits=num_folds, shuffle=True, random_state=42)
    results = []

    for params in ParameterGrid(param_grid):
        fold_accuracy = []
        for train_idx, test_idx in kfold.split(dataset):
            train_dataset = FlowDataset(dataset.iloc[train_idx])
            test_dataset = FlowDataset(dataset.iloc[test_idx])
            train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

            model = model_class(input_size=train_dataset.features.shape[1], hidden_size=params['hidden_size'], num_layers=params['num_layers'], num_classes=3).to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])

            train_model(model, train_loader, criterion, optimizer, num_epochs=params['num_epochs'])
            accuracy = evaluate_model(model, test_loader)
            fold_accuracy.append(accuracy)

        avg_accuracy = sum(fold_accuracy) / len(fold_accuracy)
        results.append((params, avg_accuracy))
    
    best_params = max(results, key=lambda x: x[1])[0]
    best_accuracy = max(results, key=lambda x: x[1])[1]
    return best_params, best_accuracy