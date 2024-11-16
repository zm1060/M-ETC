# model.py
import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, input_dim, cnn_out_channels, output_dim):
        super(CNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=cnn_out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(0.3)
        )
        self.fc = nn.Sequential(
            nn.Linear((input_dim // 2) * cnn_out_channels, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        x = x.unsqueeze(1)  # (batch_size, 1, seq_length)
        x = self.cnn(x)      # CNN for feature exatraction
        x = x.view(x.size(0), -1)  # Flatten for FC layer
        output = self.fc(x)
        return output

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attention_weights = nn.Linear(hidden_dim * 2, 1)

    def forward(self, lstm_out):
        attention_scores = self.attention_weights(lstm_out).squeeze(-1)
        attention_weights = torch.softmax(attention_scores, dim=1)
        context_vector = torch.sum(lstm_out * attention_weights.unsqueeze(-1), dim=1)
        return context_vector, attention_weights

class CNN_BiLSTM_Attention_Model(nn.Module):
    def __init__(self, input_dim, cnn_out_channels, lstm_hidden_dim, lstm_layers, output_dim):
        super(CNN_BiLSTM_Attention_Model, self).__init__()
        
        # CNN部分
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=cnn_out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(0.3)
        )
        
        # BiLSTM部分
        self.lstm = nn.LSTM(
            input_size=cnn_out_channels,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True
        )
        
        # Attention层
        self.attention = Attention(lstm_hidden_dim)
        
        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, output_dim)
        )
    
    def forward(self, x):
        x = x.unsqueeze(1)  # (batch_size, 1, seq_length)
        x = self.cnn(x)
        x = x.transpose(1, 2)  # 转换为 (batch_size, seq_length, features) 以供LSTM使用
        lstm_out, _ = self.lstm(x)
        
        # 加入注意力机制
        context_vector, attention_weights = self.attention(lstm_out)
        
        output = self.fc(context_vector)
        return output

class CNN_BiGRU_Attention_Model(nn.Module):
    def __init__(self, input_dim, cnn_out_channels, gru_hidden_dim, gru_layers, output_dim):
        super(CNN_BiGRU_Attention_Model, self).__init__()
        
        # CNN部分
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=cnn_out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(0.3)
        )
        
        # BiGRU部分
        self.gru = nn.GRU(
            input_size=cnn_out_channels,
            hidden_size=gru_hidden_dim,
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=True
        )
        
        # Attention层
        self.attention = Attention(gru_hidden_dim)
        
        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(gru_hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, output_dim)
        )
    
    def forward(self, x):
        x = x.unsqueeze(1)  # (batch_size, 1, seq_length)
        x = self.cnn(x)      # CNN特征提取
        x = x.transpose(1, 2)  # 转换为 (batch_size, seq_length, features) 以供GRU使用
        gru_out, _ = self.gru(x)
        
        # 加入注意力机制
        context_vector, attention_weights = self.attention(gru_out)
        
        output = self.fc(context_vector)
        return output

class BiLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout1=0.2, dropout2=0):
        super(BiLSTM, self).__init__()
        # First LSTM layer
        self.lstm1 = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout1 = nn.Dropout(dropout1) if dropout1 else nn.Identity()

        # Second LSTM layer
        self.lstm2 = nn.LSTM(hidden_dim * 2, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout2 = nn.Dropout(dropout2) if dropout2 else nn.Identity()

        # Fully connected layer
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        # LSTM and dropout layers
        lstm_out1, _ = self.lstm1(x)
        lstm_out1 = self.dropout1(lstm_out1)

        lstm_out2, _ = self.lstm2(lstm_out1)
        lstm_out2 = self.dropout2(lstm_out2)

        # If lstm_out2 is 3D, use the last time step
        if lstm_out2.ndimension() == 3:
            output = self.fc(lstm_out2[:, -1, :])  # Last time step
        else:
            # If lstm_out2 is 2D, pass it directly to the fully connected layer
            output = self.fc(lstm_out2)
        
        return output

class BiGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout1=0.2, dropout2=0):
        super(BiGRU, self).__init__()
        # First GRU layer
        self.gru1 = nn.GRU(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout1 = nn.Dropout(dropout1) if dropout1 else nn.Identity()

        # Second GRU layer
        self.gru2 = nn.GRU(hidden_dim * 2, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout2 = nn.Dropout(dropout2) if dropout2 else nn.Identity()

        # Fully connected layer
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        """
        Forward pass for BiGRU.
        Handles cases where the input tensor dimensions may change dynamically.
        """
        gru_out1, _ = self.gru1(x)  # Shape: (batch_size, seq_length, hidden_dim*2)
        gru_out1 = self.dropout1(gru_out1)

        gru_out2, _ = self.gru2(gru_out1)  # Shape: (batch_size, seq_length, hidden_dim*2)
        gru_out2 = self.dropout2(gru_out2)

        if gru_out2.ndimension() == 3:  # Check if the output has time steps
            output = self.fc(gru_out2[:, -1, :])  # Use the last time step
        elif gru_out2.ndimension() == 2:  # Edge case: Input is already reduced (batch_size, hidden_dim*2)
            output = self.fc(gru_out2)
        else:
            raise ValueError(f"Unexpected tensor dimensions: {gru_out2.shape}")

        return output
    

class CNN_BiGRU(nn.Module):
    def __init__(self, input_dim, cnn_out_channels, gru_hidden_dim, gru_layers, output_dim):
        super(CNN_BiGRU, self).__init__()
        
        # CNN部分
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=cnn_out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(0.3)
        )
        
        # BiGRU部分
        self.gru = nn.GRU(
            input_size=cnn_out_channels,
            hidden_size=gru_hidden_dim,
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=True
        )
        
        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(gru_hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, output_dim)
        )
    
    def forward(self, x):
        x = x.unsqueeze(1)  # (batch_size, 1, seq_length)
        x = self.cnn(x)      # CNN特征提取
        x = x.transpose(1, 2)  # 转换为 (batch_size, seq_length, features) 以供GRU使用
        gru_out, _ = self.gru(x)
        output = self.fc(gru_out[:, -1, :])  # 使用最后时间步的特征
        return output
    

class CNN_BiLSTM(nn.Module):
    def __init__(self, input_dim, cnn_out_channels, lstm_hidden_dim, lstm_layers, output_dim):
        super(CNN_BiLSTM, self).__init__()
        
        # CNN部分
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=cnn_out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(0.3)
        )
        
        # BiLSTM部分
        self.lstm = nn.LSTM(
            input_size=cnn_out_channels,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True
        )
        
        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, output_dim)
        )
    
    def forward(self, x):
        x = x.unsqueeze(1)  # 添加通道维度: (batch_size, 1, seq_length)
        x = self.cnn(x)      # CNN特征提取
        x = x.transpose(1, 2)  # 转换为 (batch_size, seq_length // 2, cnn_out_channels) 以供LSTM使用
        lstm_out, _ = self.lstm(x)  # LSTM提取序列信息
        output = self.fc(lstm_out[:, -1, :])  # 使用最后时间步的特征
        return output