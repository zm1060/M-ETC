import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import torch
from torch.utils.data import Dataset, DataLoader

class FlowDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe.drop(columns=['SourceIP', 'DestinationIP', 'SourcePort', 'DestinationPort', 'TimeStamp'], errors='ignore')
        self.features = self.dataframe.iloc[:, :-1].values
        self.label_encoder = LabelEncoder()
        self.labels = self.label_encoder.fit_transform(dataframe['Label'].values)
        self.scaler = StandardScaler()
        self.features = self.scaler.fit_transform(self.features)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        features = torch.tensor(self.features[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return features, label

def load_data(datasets_dirs):
    data_loaders = {}
    for dataset_name, dir_path in datasets_dirs.items():
        all_files = []
        print(f'正在加载数据集: {dataset_name}...')
        for root, dirs, files in os.walk(dir_path):
            for file in files:
                if file.endswith('.csv'):
                    file_path = os.path.join(root, file)
                    print(f'正在读取文件: {file_path}')
                    try:
                        df = pd.read_csv(file_path, engine='c', on_bad_lines='skip')
                        if not df.empty:
                            all_files.append(df)
                            print(f'成功加载文件: {file_path}，行数: {len(df)}')
                    except pd.errors.EmptyDataError:
                        print(f'空的CSV文件已跳过: {file_path}')
        if all_files:
            df = pd.concat(all_files, ignore_index=True)
            df.dropna(inplace=True)
            if 'Label' in df.columns:
                print(f'数据集 {dataset_name} 包含有效数据，行数: {len(df)}')
                train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
                train_loader = DataLoader(FlowDataset(train_df), batch_size=64, shuffle=True)
                test_loader = DataLoader(FlowDataset(test_df), batch_size=64, shuffle=False)
                data_loaders[dataset_name] = {'train': train_loader, 'test': test_loader}
            else:
                print(f'数据集 {dataset_name} 中未找到标签列')
        else:
            print(f'数据集 {dataset_name} 中没有有效的文件')
    return data_loaders
