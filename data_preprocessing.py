# data_preprocessing.py
import os
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from torch.utils.data import TensorDataset, DataLoader

def load_data_from_directory(directory_path):
    combined_data = pd.DataFrame()

    for root, _, files in os.walk(directory_path):
        for file in files:
            if file.endswith('.csv'):
                file_path = os.path.join(root, file)
                print(f"Processing file: {file_path}")
                data = pd.read_csv(file_path)
                combined_data = pd.concat([combined_data, data], ignore_index=True)
    
    if combined_data.empty:
        raise ValueError("No CSV files found in the directory.")

    return combined_data

def preprocess_data(data, label_column='Label'):
    features_to_remove = ['SourceIP', 'DestinationIP', 'SourcePort', 'DestinationPort', 'TimeStamp']
    data = data.drop(columns=[feat for feat in features_to_remove if feat in data.columns], errors='ignore')

    print(f"Data shape after removing features: {data.shape}")

    nan_threshold = 0.9
    data = data.dropna(axis=1, thresh=int((1 - nan_threshold) * len(data)))
    data = data.fillna(0)

    print(f"Data shape after handling NaN: {data.shape}")
    
    if data.empty:
        raise ValueError("Data is empty after preprocessing. Please check your dataset.")

    label_encoder = LabelEncoder()
    if label_column not in data.columns:
        raise ValueError(f"Label column '{label_column}' not found in the dataset.")
    
    data['Label'] = label_encoder.fit_transform(data[label_column])


    X = data.drop(columns=[label_column]).values
    y = data['Label'].values

    if X.shape[0] != y.shape[0]:
        raise ValueError(f"Feature matrix and labels have inconsistent samples. X.shape: {X.shape}, y.shape: {y.shape}")

    print(f"Feature matrix shape: {X.shape}")

    # scaler = StandardScaler()
    scaler = MinMaxScaler()

    X = scaler.fit_transform(X)

    return X, y, scaler, label_encoder

def get_dataloaders(X, y, test_size=0.2, batch_size=64, sample_size=None):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)

    if sample_size:
        if isinstance(sample_size, float) and 0 < sample_size < 1:
            sample_size = int(len(X_train) * sample_size)
        elif isinstance(sample_size, int) and sample_size < len(X_train):
            sample_size = sample_size
        else:
            raise ValueError("sample_size must be a proportion (0 < sample_size < 1) or an integer less than the training set size")

        indices = np.random.choice(len(X_train), sample_size, replace=False)
        X_train, y_train = X_train[indices], y_train[indices]

    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    return train_loader, val_loader

def get_test_loader(X, y, batch_size=64):
    test_dataset = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long))
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    return test_loader
