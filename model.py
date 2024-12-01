# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attention_weights = nn.Linear(hidden_dim * 2, 1)

    def forward(self, lstm_out):
        attention_scores = self.attention_weights(lstm_out).squeeze(-1)
        attention_weights = torch.softmax(attention_scores, dim=1)
        context_vector = torch.sum(lstm_out * attention_weights.unsqueeze(-1), dim=1)
        return context_vector, attention_weights

class CNN_Model(nn.Module):
    def __init__(self, input_dim, cnn_out_channels, output_dim):
        super(CNN_Model, self).__init__()
        
        # CNN with residual connection
        self.cnn = nn.Sequential(
            nn.Conv1d(1, cnn_out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_out_channels),
            nn.ReLU(),
            nn.Conv1d(cnn_out_channels, cnn_out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_out_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(0.3)
        )
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear((input_dim // 2) * cnn_out_channels, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, output_dim)
        )
    
    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension
        x = self.cnn(x)
        x = x.view(x.size(0), -1)  # Flatten for FC layer
        output = self.fc(x)
        return output

class LSTM_Model(nn.Module):
    def __init__(self, input_dim, lstm_hidden_dim, lstm_layers, output_dim):
        super(LSTM_Model, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_dim,  # This is correct, as input_size is for LSTM layer
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=False  # Single-direction LSTM
        )

        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        if lstm_out.dim() == 2:
            last_hidden_state = lstm_out  # If 2D output, take it directly
        else:
            last_hidden_state = lstm_out[:, -1, :]  # Else take the last hidden state
        output = self.fc(last_hidden_state)
        return output

class GRU_Model(nn.Module):
    def __init__(self, input_dim, gru_hidden_dim, gru_layers, output_dim):
        super(GRU_Model, self).__init__()
        
        # Define GRU layer
        self.gru = nn.GRU(
            input_size=input_dim,  # Input dimension for GRU
            hidden_size=gru_hidden_dim,  # Hidden state dimension
            num_layers=gru_layers,  # Number of GRU layers
            batch_first=True,  # Input tensor shape: (batch_size, seq_len, input_dim)
            bidirectional=False  # Set to True if you want a bidirectional GRU
        )

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(gru_hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
    
    def forward(self, x):
        # Forward pass through GRU
        gru_out, _ = self.gru(x)
        
        # If the output is 2D, just use it directly as the last hidden state
        if gru_out.dim() == 2:
            last_hidden_state = gru_out  # (batch_size, hidden_dim)
        else:
            last_hidden_state = gru_out[:, -1, :]  # (batch_size, hidden_dim) for sequence-based output
        
        # Pass the last hidden state through fully connected layers
        output = self.fc(last_hidden_state)
        return output

class BiLSTM_Model(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout1=0.2, dropout2=0):
        super(BiLSTM_Model, self).__init__()
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

class BiGRU_Model(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout1=0.2, dropout2=0):
        super(BiGRU_Model, self).__init__()
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
    

class CNN_BiGRU_Model(nn.Module):
    def __init__(self, input_dim, cnn_out_channels, gru_hidden_dim, gru_layers, output_dim, dropout=0.3, gru_dropout=0.3):
        super(CNN_BiGRU_Model, self).__init__()
        
        # CNN Layer
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=cnn_out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_out_channels),
            nn.ReLU(),
            nn.Conv1d(cnn_out_channels, cnn_out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_out_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(dropout)
        )
        
        # BiGRU Layer
        self.gru = nn.GRU(
            input_size=cnn_out_channels,
            hidden_size=gru_hidden_dim,
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=True,
            dropout=gru_dropout
        )
        
        # Fully connected layer
        self.fc = nn.Sequential(
            nn.Linear(gru_hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, output_dim)
        )
    
    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.cnn(x)
        x = x.transpose(1, 2)
        
        gru_out, _ = self.gru(x)
        output = self.fc(gru_out[:, -1, :])  # Using last time step
        return output

    

class CNN_BiLSTM_Model(nn.Module):
    def __init__(self, input_dim, cnn_out_channels, lstm_hidden_dim, lstm_layers, output_dim):
        super(CNN_BiLSTM_Model, self).__init__()
        
        # CNN with residual connection
        self.cnn = nn.Sequential(
            nn.Conv1d(1, cnn_out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_out_channels),
            nn.ReLU(),
            nn.Conv1d(cnn_out_channels, cnn_out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_out_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(0.3)
        )
        
        # BiLSTM with LayerNorm
        self.lstm = nn.LSTM(
            input_size=cnn_out_channels,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True
        )
        self.layer_norm = nn.LayerNorm(lstm_hidden_dim * 2)

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, output_dim)
        )
    
    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension
        cnn_out = self.cnn(x)
        cnn_out = cnn_out.transpose(1, 2)  # (batch_size, seq_length, features)
        
        lstm_out, _ = self.lstm(cnn_out)
        lstm_out = self.layer_norm(lstm_out)

        output = self.fc(lstm_out[:, -1, :])  # Use the last time step
        return output

class CAttention(nn.Module):
    def __init__(self, hidden_size):
        super(CAttention, self).__init__()
        self.attn = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: (batch_size, seq_length, cnn_out_channels)
        attn_weights = torch.softmax(self.attn(x), dim=1)  # (batch_size, seq_length, 1)
        context_vector = torch.sum(attn_weights * x, dim=1)  # (batch_size, cnn_out_channels)
        return context_vector, attn_weights

class CNN_Attention_Model(nn.Module):
    def __init__(self, input_dim, cnn_out_channels, output_dim):
        super(CNN_Attention_Model, self).__init__()
        
        # CNN with residual connection
        self.cnn = nn.Sequential(
            nn.Conv1d(1, cnn_out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_out_channels),
            nn.ReLU(),
            nn.Conv1d(cnn_out_channels, cnn_out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_out_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(0.3)
        )

        # Attention mechanism
        self.attention = CAttention(cnn_out_channels)

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(cnn_out_channels, 128),  # The expected input is cnn_out_channels
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, output_dim)
        )
    
    def forward(self, x):
        # Add channel dimension for Conv1d (batch_size, 1, seq_length)
        x = x.unsqueeze(1)  # (batch_size, 1, seq_length)
        
        # CNN part
        cnn_out = self.cnn(x)  # Output shape: (batch_size, cnn_out_channels, seq_length)
        
        # Transpose to (batch_size, seq_length, cnn_out_channels)
        cnn_out = cnn_out.transpose(1, 2)
        
        # Apply attention mechanism (context_vector shape: (batch_size, cnn_out_channels))
        context_vector, _ = self.attention(cnn_out)

        # Flatten context vector for fully connected layer input
        context_vector = context_vector.view(context_vector.size(0), -1)  # Flatten to (batch_size, cnn_out_channels)

        # Fully connected layers
        output = self.fc(context_vector)  # Output shape: (batch_size, output_dim)
        return output

# LAttention: LSTM的注意力机制
class LAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(LAttention, self).__init__()
        # 使用一个线性层来计算注意力分数
        self.attention_weights = nn.Linear(hidden_dim * 2, 1)

    def forward(self, lstm_out):
        # 计算每个时间步的注意力分数 (batch_size, seq_len, 1)
        attention_scores = self.attention_weights(lstm_out)
        
        # 如果注意力分数是三维张量（即 (batch_size, seq_len, 1)），就去掉最后一维
        if attention_scores.dim() == 3:
            attention_scores = attention_scores.squeeze(-1)  # (batch_size, seq_len)
        
        # 使用缩放技巧以增强数值稳定性
        attention_scores = attention_scores / (lstm_out.size(-1) ** 0.5)  # 缩放

        # 使用 softmax 得到注意力权重 (batch_size, seq_len)
        attention_weights = torch.softmax(attention_scores, dim=1)
        
        # 使用加权求和得到上下文向量 (batch_size, hidden_dim * 2)
        context_vector = torch.sum(lstm_out * attention_weights.unsqueeze(-1), dim=1)
        
        return context_vector, attention_weights

# BiLSTM_Attention_Model: BiLSTM模型 + Attention
class BiLSTM_Attention_Model(nn.Module):
    def __init__(self, input_dim, lstm_hidden_dim, lstm_layers, output_dim, dropout=0.3):
        super(BiLSTM_Attention_Model, self).__init__()
        # BiLSTM 层
        self.lstm = nn.LSTM(
            input_size=input_dim,        # 输入特征的维度
            hidden_size=lstm_hidden_dim, # 隐藏层的维度
            num_layers=lstm_layers,      # LSTM 层数
            batch_first=True,            # 输入输出格式为 (batch_size, seq_len, input_size)
            bidirectional=True           # 双向 LSTM
        )
        
        # LayerNorm 层进行标准化
        self.layer_norm = nn.LayerNorm(lstm_hidden_dim * 2)
        
        # Dropout 层防止过拟合
        self.dropout = nn.Dropout(dropout)
        
        # 注意力机制层
        self.attention = LAttention(lstm_hidden_dim)
        
        # 输出层
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden_dim * 2, 128),  # 将 BiLSTM 的输出连接到一个线性层
            nn.ReLU(),                            # 激活函数
            nn.Dropout(dropout),                  # Dropout 防止过拟合
            nn.Linear(128, output_dim)            # 最终的输出层，输出类别数
        )

    def forward(self, x):
        # 获取 BiLSTM 的输出 (batch_size, seq_len, hidden_dim * 2)
        lstm_out, _ = self.lstm(x)
        
        # LayerNorm 进行归一化
        lstm_out = self.layer_norm(lstm_out)
        
        # Dropout
        lstm_out = self.dropout(lstm_out)
        
        # 获取上下文向量和注意力权重
        context_vector, attention_weights = self.attention(lstm_out)
        
        # 通过全连接层输出最终结果
        output = self.fc(context_vector)
        
        return output


# GAttention: GRU的注意力机制
class GAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(GAttention, self).__init__()
        # 使用一个线性层来计算注意力分数
        self.attention_weights = nn.Linear(hidden_dim * 2, 1)

    def forward(self, gru_out):
        # 计算每个时间步的注意力分数 (batch_size, seq_len, 1)
        attention_scores = self.attention_weights(gru_out)
        
        # 如果注意力分数是三维张量（即 (batch_size, seq_len, 1)），就去掉最后一维
        if attention_scores.dim() == 3:
            attention_scores = attention_scores.squeeze(-1)  # (batch_size, seq_len)
        
        # 使用缩放技巧以增强数值稳定性
        attention_scores = attention_scores / (gru_out.size(-1) ** 0.5)  # 缩放

        # 使用 softmax 得到注意力权重 (batch_size, seq_len)
        attention_weights = torch.softmax(attention_scores, dim=1)
        
        # 使用加权求和得到上下文向量 (batch_size, hidden_dim * 2)
        context_vector = torch.sum(gru_out * attention_weights.unsqueeze(-1), dim=1)
        
        return context_vector, attention_weights


# BiGRU_Attention_Model: BiGRU模型 + Attention
class BiGRU_Attention_Model(nn.Module):
    def __init__(self, input_dim, gru_hidden_dim, gru_layers, output_dim, dropout=0.3, gru_dropout=0.3):
        super(BiGRU_Attention_Model, self).__init__()
        # BiGRU 层
        self.gru = nn.GRU(
            input_size=input_dim,         # 输入特征的维度
            hidden_size=gru_hidden_dim,   # 隐藏层的维度
            num_layers=gru_layers,        # GRU 层数
            batch_first=True,             # 输入输出格式为 (batch_size, seq_len, input_size)
            bidirectional=True,           # 双向 GRU
            dropout=gru_dropout           # Dropout 防止过拟合
        )
        
        # LayerNorm 层进行标准化
        self.layer_norm = nn.LayerNorm(gru_hidden_dim * 2)
        
        # Dropout 层
        self.dropout = nn.Dropout(dropout)
        
        # 注意力机制层
        self.attention = GAttention(gru_hidden_dim)
        
        # 输出层
        self.fc = nn.Sequential(
            nn.Linear(gru_hidden_dim * 2, 128),  # 将 BiGRU 的输出连接到一个线性层
            nn.ReLU(),                            # 激活函数
            nn.Dropout(dropout),                  # Dropout 防止过拟合
            nn.BatchNorm1d(128),                  # BatchNorm 增强稳定性
            nn.Linear(128, output_dim)            # 最终的输出层，输出类别数
        )

    def forward(self, x):
        # 获取 BiGRU 的输出 (batch_size, seq_len, hidden_dim * 2)
        gru_out, _ = self.gru(x)
        
        # LayerNorm 进行归一化
        gru_out = self.layer_norm(gru_out)
        
        # Dropout
        gru_out = self.dropout(gru_out)
        
        # 获取上下文向量和注意力权重
        context_vector, attention_weights = self.attention(gru_out)
        
        # 通过全连接层输出最终结果
        output = self.fc(context_vector)
        return output

class CNN_BiLSTM_Attention_Model(nn.Module):
    def __init__(self, input_dim, cnn_out_channels, lstm_hidden_dim, lstm_layers, output_dim):
        super(CNN_BiLSTM_Attention_Model, self).__init__()
        
        # CNN with residual connection
        self.cnn = nn.Sequential(
            nn.Conv1d(1, cnn_out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_out_channels),
            nn.ReLU(),
            nn.Conv1d(cnn_out_channels, cnn_out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_out_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(0.3)
        )
        
        # BiLSTM with LayerNorm
        self.lstm = nn.LSTM(
            input_size=cnn_out_channels,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True
        )
        self.layer_norm = nn.LayerNorm(lstm_hidden_dim * 2)

        # Attention
        self.attention = Attention(lstm_hidden_dim)
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, output_dim)
        )
    
    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension
        cnn_out = self.cnn(x)
        cnn_out = cnn_out.transpose(1, 2)  # (batch_size, seq_length, features)
        
        lstm_out, _ = self.lstm(cnn_out)
        lstm_out = self.layer_norm(lstm_out)

        context_vector, _ = self.attention(lstm_out)  # Apply attention
        output = self.fc(context_vector)
        return output


class CNN_BiGRU_Attention_Model(nn.Module):
    def __init__(self, input_dim, cnn_out_channels, gru_hidden_dim, gru_layers, output_dim, dropout=0.3, gru_dropout=0.3):
        super(CNN_BiGRU_Attention_Model, self).__init__()
        
        # CNN Layer with residual connection
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=cnn_out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_out_channels),
            nn.ReLU(),
            nn.Conv1d(cnn_out_channels, cnn_out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_out_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),  # Add max pooling to reduce sequence length
            nn.Dropout(dropout)
        )
        
        # BiGRU Layer with LayerNorm
        self.gru = nn.GRU(
            input_size=cnn_out_channels,  # CNN output size
            hidden_size=gru_hidden_dim,  # GRU hidden size
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=True,
            dropout=gru_dropout  # Dropout for GRU layers
        )
        self.layer_norm = nn.LayerNorm(gru_hidden_dim * 2)  # Apply LayerNorm after GRU
        
        # Attention Layer
        self.attention = Attention(gru_hidden_dim)
        
        # Fully Connected Layer with BatchNorm
        self.fc = nn.Sequential(
            nn.Linear(gru_hidden_dim * 2, 128),  # BiGRU output size is 2 * gru_hidden_dim
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.BatchNorm1d(128),  # BatchNorm after the first linear layer
            nn.Linear(128, output_dim)
        )

        # Initialize layers (help with convergence)
        self._initialize_weights()

    def _initialize_weights(self):
        # Initialize weights for CNN and GRU layers
        for name, param in self.named_parameters():
            if 'weight' in name:
                if isinstance(self._modules[name.split('.')[0]], nn.Conv1d):
                    nn.init.kaiming_normal_(param, nonlinearity='relu')
                elif isinstance(self._modules[name.split('.')[0]], nn.GRU):
                    nn.init.xavier_normal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def forward(self, x):
        # Add an extra channel dimension for CNN input (batch_size, 1, seq_length)
        x = x.unsqueeze(1)  # (batch_size, 1, seq_length)
        
        # CNN feature extraction
        x = self.cnn(x)  # Apply CNN layers
        x = x.transpose(1, 2)  # Convert to shape (batch_size, seq_len, cnn_out_channels) for GRU input

        # BiGRU for sequence processing
        gru_out, _ = self.gru(x)  # (batch_size, seq_len, 2 * gru_hidden_dim)
        gru_out = self.layer_norm(gru_out)  # Apply LayerNorm

        # Attention Mechanism
        context_vector, attention_weights = self.attention(gru_out)  # (batch_size, 2 * gru_hidden_dim)

        # Final classification (fully connected layer)
        output = self.fc(context_vector)  # (batch_size, output_dim)
        
        return output
