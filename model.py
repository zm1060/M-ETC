# model.py
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

class Attention(nn.Module):
    """
    Implements an attention mechanism that assigns different importance to time steps.
    """
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attention_weights = nn.Linear(hidden_dim * 2, 1)

    def forward(self, lstm_out):
        """
        Forward pass for the attention mechanism.
        
        Parameters:
            lstm_out (torch.Tensor): Output from LSTM layers (batch_size, seq_length, hidden_dim*2).
        
        Returns:
            tuple: (context_vector, attention_weights)
        """
        attention_scores = self.attention_weights(lstm_out).squeeze(-1)
        attention_weights = torch.softmax(attention_scores, dim=1)
        context_vector = torch.sum(lstm_out * attention_weights.unsqueeze(-1), dim=1)
        return context_vector, attention_weights

class CNN_BiLSTM_Attention_Model(nn.Module):
    """
    A hybrid CNN-BiLSTM model with an attention layer.
    """
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
        """
        Forward pass for the CNN-BiLSTM-Attention model.
        
        Parameters:
            x (torch.Tensor): Input tensor (batch_size, seq_length, num_features).
        
        Returns:
            torch.Tensor: Output logits (batch_size, output_dim).
        """
        x = x.unsqueeze(1)  # (batch_size, 1, seq_length)
        x = self.cnn(x)      # CNN特征提取
        x = x.transpose(1, 2)  # 转换为 (batch_size, seq_length, features) 以供LSTM使用
        lstm_out, _ = self.lstm(x)
        
        # 加入注意力机制
        context_vector, attention_weights = self.attention(lstm_out)
        
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

    
def get_baseline_models(input_dim, output_dim):
    """
    Initializes different models for comparison.
    """
    models = {
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
        'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='mlogloss'),
        'BiLSTM': BiLSTM(input_dim=input_dim, hidden_dim=64, output_dim=output_dim),
        'CNN_BiLSTM_Attention': CNN_BiLSTM_Attention_Model(input_dim=input_dim, cnn_out_channels=64, lstm_hidden_dim=64, lstm_layers=2, output_dim=output_dim)
    }
    return models