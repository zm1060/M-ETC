# model.py
import torch.nn as nn

class BiLSTMModel(nn.Module):
    def __init__(self, numerical_dims, hidden_dim, lstm_layers, output_dim):
        super(BiLSTMModel, self).__init__()
        
        # BiLSTM
        self.lstm = nn.LSTM(
            input_size=numerical_dims, 
            hidden_size=hidden_dim, 
            num_layers=lstm_layers, 
            batch_first=True, 
            bidirectional=True
        )
        
        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),  # BiLSTM是双向的，因此是 hidden_dim * 2
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, output_dim)  # 输出层
        )
        
    def forward(self, X_num):
        X_num = X_num.unsqueeze(1)  # 扩展维度以匹配 (batch_size, seq_length, input_size)
        lstm_out, _ = self.lstm(X_num)  # (batch_size, seq_length, hidden_dim * 2)
        lstm_out = lstm_out[:, -1, :]  # 取最后一个时间步的输出
        out = self.fc(lstm_out)
        return out
