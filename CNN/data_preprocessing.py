# data_preprocessing.py
import os
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import TensorDataset, DataLoader

def load_data_from_directory(directory_path):
    """
    Load and preprocess data from all CSV files in a directory (including subdirectories).
    """
    # Initialize an empty DataFrame to store combined data
    combined_data = pd.DataFrame()

    # Traverse through the directory and subdirectories
    for root, _, files in os.walk(directory_path):
        for file in files:
            if file.endswith('.csv'):
                # Construct the full file path
                file_path = os.path.join(root, file)
                print(f"Processing file: {file_path}")
                # Load the CSV into a DataFrame
                data = pd.read_csv(file_path)
                combined_data = pd.concat([combined_data, data], ignore_index=True)
    
    if combined_data.empty:
        raise ValueError("No CSV files found in the directory.")

    return combined_data

def preprocess_data(data):
    """
    Preprocess the combined DataFrame and prepare it for the model.
    Remove 'SourceIP', 'DestinationIP', 'SourcePort', 'DestinationPort' features.
    """
    # 移除过拟合的特征
    features_to_remove = ['SourceIP', 'DestinationIP', 'SourcePort', 'DestinationPort', 'TimeStamp']
    data = data.drop(columns=[feat for feat in features_to_remove if feat in data.columns], errors='ignore')

    # 打印数据的大小，确保数据集不为空
    print(f"Data shape after removing features: {data.shape}")
    
    # 打印每列的 NaN 比例
    print("NaN values per column before processing:")
    print(data.isnull().mean())

    # 处理 NaN 值
    # 1. 如果某些列的 NaN 比例过高，可以选择删除这些列
    nan_threshold = 0.9  # 如果超过 90% 的值为 NaN，则删除该列
    data = data.dropna(axis=1, thresh=int((1 - nan_threshold) * len(data)))

    # 2. 对剩余的 NaN 值进行填充（这里选择填充为 0，也可以用其他方法）
    data = data.fillna(0)

    # 再次打印处理 NaN 后的数据大小
    print(f"Data shape after handling NaN: {data.shape}")
    
    if data.empty:
        raise ValueError("Data is empty after preprocessing. Please check your dataset.")

    # 编码标签列
    label_encoder = LabelEncoder()
    data['Label'] = label_encoder.fit_transform(data['Label'])

    # 分离特征和标签
    X = data.drop(columns=['Label']).values
    y = data['Label'].values

    # 打印 X 的大小，确保特征矩阵不为空
    print(f"Feature matrix shape: {X.shape}")

    # 数值特征标准化
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # 将数据拆分为训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=42)

    # 转换为张量
    X_train = torch.tensor(X_train, dtype=torch.float)
    X_val = torch.tensor(X_val, dtype=torch.float)
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_val = torch.tensor(y_val, dtype=torch.long)

    # 创建数据集
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)

    # 返回 train_dataset, val_dataset, scaler, 以及 仅用于 'Label' 的 label_encoder
    return train_dataset, val_dataset, scaler, label_encoder


def get_dataloaders(train_dataset, val_dataset, batch_size=64):
    """
    Generate DataLoader objects for training and validation datasets.
    """
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    return train_loader, val_loader
