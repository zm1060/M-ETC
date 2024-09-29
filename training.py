import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import logging

# 设置日志记录
logging.basicConfig(filename='experiment_log.txt', level=logging.INFO, format='%(asctime)s - %(message)s')

def train_model(model, data_loader, criterion, optimizer, scheduler=None, num_epochs=5, device=None):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(data_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if scheduler:
                scheduler.step()  # 在每个训练步后更新学习率
            
            if i % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i}/{len(data_loader)}], Loss: {loss.item():.4f}')
        
        logging.info(f'Epoch [{epoch+1}/{num_epochs}], Average Loss: {running_loss / len(data_loader):.4f}')

def evaluate_model(model, data_loader, device):
    model.eval()
    all_labels = []
    all_predictions = []
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    accuracy = np.mean(np.array(all_predictions) == np.array(all_labels)) * 100
    precision = precision_score(all_labels, all_predictions, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_predictions, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_predictions, average='weighted', zero_division=0)
    cm = confusion_matrix(all_labels, all_predictions)

    logging.info(f'Accuracy: {accuracy:.2f}%')
    logging.info(f'Precision: {precision:.2f}')
    logging.info(f'Recall: {recall:.2f}')
    logging.info(f'F1 Score: {f1:.2f}')
    logging.info(f'Confusion Matrix:\n{cm}')

    return accuracy
