import os
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler, LabelEncoder
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from data_preprocessing import load_data_from_directory, preprocess_data, get_dataloaders
from model import BiGRUModel  # 根据需要修改为 BiGRUModel 或 BiLSTMModel

# Define a function to preprocess new test data and generate a DataLoader
def preprocess_new_data(test_data, scaler, label_encoder):
    """
    Preprocess the new test data. We use the existing scaler and label_encoder from the training phase.
    """
    features_to_remove = ['SourceIP', 'DestinationIP', 'SourcePort', 'DestinationPort', 'TimeStamp']
    test_data = test_data.drop(columns=[feat for feat in features_to_remove if feat in test_data.columns], errors='ignore')
    
    # Fill missing values in the same way as training
    test_data = test_data.fillna(0)
    
    # Separate features and labels
    X_test = test_data.drop(columns=['Label']).values
    y_test = test_data['Label'].values

    # Apply the existing scaler from training
    X_test = scaler.transform(X_test)

    # Encode labels using the existing label_encoder
    y_test = label_encoder.transform(y_test)
    
    # Convert to tensors
    X_test = torch.tensor(X_test, dtype=torch.float)
    y_test = torch.tensor(y_test, dtype=torch.long)
    
    # Create a TensorDataset for test data
    test_dataset = TensorDataset(X_test, y_test)
    
    return test_dataset

# Define a function to load the model and test on new data
def test_model(test_directory, model_path, batch_size=64):
    # Load the trained model parameters
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load the test data from the new directory
    test_data = load_data_from_directory(test_directory)

    # Preprocess the test data using the trained scaler and label encoder
    train_dataset, val_dataset, scaler, label_encoder = preprocess_data(load_data_from_directory('../csv_output/CIRA-CIC-DoHBrw-2020'))  # Ensure the preprocessing uses same training logic
    test_dataset = preprocess_new_data(test_data, scaler, label_encoder)
    
    # Create DataLoader for test data
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Load the trained model
    numerical_dims = train_dataset[0][0].shape[0]  # Input feature dimension
    model = BiGRUModel(  # 使用 BiGRUModel 而非 BiLSTMModel
        numerical_dims=numerical_dims,  
        hidden_dim=64,
        gru_layers=2,  # 如果是 BiGRU，记得调整
        output_dim=len(label_encoder.classes_)  
    )
    
    # Load the saved model weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)

    # Switch model to evaluation mode
    model.eval()

    # Initialize metrics
    total = 0
    correct = 0
    predictions = []
    labels = []

    # No gradient calculation needed during evaluation
    with torch.no_grad():
        for batch in test_loader:
            X_batch, y_batch = [b.to(device) for b in batch]
            outputs = model(X_batch)

            _, predicted = torch.max(outputs, 1)
            predictions.extend(predicted.cpu().numpy())
            labels.extend(y_batch.cpu().numpy())
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()

    # Calculate accuracy
    accuracy = correct / total

    # Calculate other evaluation metrics
    precision = precision_score(labels, predictions, average='weighted')
    recall = recall_score(labels, predictions, average='weighted')
    f1 = f1_score(labels, predictions, average='weighted')
    conf_matrix = confusion_matrix(labels, predictions)

    print(f'Test Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')
    print(f'Confusion Matrix:\n{conf_matrix}')

    return predictions, labels, accuracy, precision, recall, f1, conf_matrix

# Example of how to use the function
test_directory = '../csv_output/Collection'
model_path = 'model_epoch_16.pth'  # Adjust the model path accordingly
predictions, labels, test_accuracy, test_precision, test_recall, test_f1, confusion_matrix = test_model(test_directory, model_path)
