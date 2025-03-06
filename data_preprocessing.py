# data_preprocessing.py
import logging
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

def get_dataloaders(X, y, batch_size=64, sample_size=None, test_size=None):
    """
    Creates PyTorch DataLoaders for training and validation sets.
    If test_size is provided, splits the data into train and validation sets.
    If test_size is None, assumes the data is already split (e.g., by k-fold).

    Parameters:
    - X (numpy.ndarray): Feature data
    - y (numpy.ndarray): Label data
    - batch_size (int): Number of samples per batch
    - sample_size (float or int): If specified, limits the training set to a subset
    - test_size (float): If provided, splits data into train and validation sets

    Returns:
    - train_loader (DataLoader): DataLoader for the training set
    - val_loader (DataLoader): DataLoader for the validation set (if test_size is provided)
    """
    if test_size is not None:
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
    else:
        # If test_size is None, assume data is already split
        X_train, y_train = X, y
        X_val, y_val = None, None

    if sample_size:
        if isinstance(sample_size, float) and 0 < sample_size < 1:
            sample_size = int(len(X_train) * sample_size)
        elif isinstance(sample_size, int) and sample_size < len(X_train):
            sample_size = sample_size
        else:
            raise ValueError(
                "sample_size must be a proportion (0 < sample_size < 1) "
                "or an integer less than the training set size"
            )

        indices = np.random.choice(len(X_train), sample_size, replace=False)
        X_train, y_train = X_train[indices], y_train[indices]

    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long)
    )
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )

    if X_val is not None and y_val is not None:
        val_dataset = TensorDataset(
            torch.tensor(X_val, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.long)
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size
        )
        return train_loader, val_loader
    else:
        return train_loader

def get_fine_tune_loader(X_fine_train, y_fine_train, X_fine_val, y_fine_val, sample_size, batch_size=64):
    """
    Creates DataLoaders for the fine-tuning phase by sampling the training data and manually handling the validation data.

    Parameters:
    - X_fine_train (numpy.ndarray): Training feature data.
    - y_fine_train (numpy.ndarray): Training label data.
    - X_fine_val (numpy.ndarray): Validation feature data.
    - y_fine_val (numpy.ndarray): Validation label data.
    - sample_size (float or int): Sampling size.
    - batch_size (int): Number of samples per batch for both DataLoaders.

    Returns:
    - fine_tune_train_loader (DataLoader): DataLoader for the training set.
    - fine_tune_val_loader (DataLoader): DataLoader for the validation set.
    """
    # Sampling the training data
    X_fine_train, y_fine_train = sample_data(X_fine_train, y_fine_train, sample_size)

    # Create the training dataset and DataLoader
    train_dataset = TensorDataset(
        torch.tensor(X_fine_train, dtype=torch.float32),
        torch.tensor(y_fine_train, dtype=torch.long)
    )
    fine_tune_train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True  # Shuffle for better training
    )

    # Create the validation dataset and DataLoader
    val_dataset = TensorDataset(
        torch.tensor(X_fine_val, dtype=torch.float32),
        torch.tensor(y_fine_val, dtype=torch.long)
    )
    fine_tune_val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False  # No shuffle for validation
    )

    return fine_tune_train_loader, fine_tune_val_loader

def sample_data(X, y, sample_size):
    """
    Samples the training data based on the specified sample size.

    Parameters:
    - X (numpy.ndarray): Feature data.
    - y (numpy.ndarray): Label data.
    - sample_size (float or int): 
        - If float (0 < sample_size < 1), represents the proportion of the data to sample.
        - If int, represents the absolute number of samples to retain.

    Returns:
    - X_sampled (numpy.ndarray): Sampled feature data.
    - y_sampled (numpy.ndarray): Sampled label data.
    """
    if sample_size:
        if isinstance(sample_size, float) and 0 < sample_size < 1:
            sample_size_calculated = int(len(X) * sample_size)
        elif isinstance(sample_size, int) and sample_size < len(X):
            sample_size_calculated = sample_size
        else:
            raise ValueError("sample_size must be a float between 0 and 1 or an integer less than the size of the dataset.")
        
        sampled_indices = np.random.choice(len(X), sample_size_calculated, replace=False)
        X_sampled = X[sampled_indices]
        y_sampled = y[sampled_indices]
        return X_sampled, y_sampled
    return X, y

def get_test_loader(X, y, batch_size=64):
    test_dataset = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long))
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    return test_loader
