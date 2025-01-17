# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN_Model(nn.Module):
    def __init__(self, input_dim, cnn_out_channels, output_dim):
        super(CNN_Model, self).__init__()

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

        self.fc = nn.Sequential(
            nn.Linear((input_dim // 2) * cnn_out_channels, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        output = self.fc(x)
        return output

class RNN_Model(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1):
        super(RNN_Model, self).__init__()
        self.rnn = nn.RNN(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        out, _ = self.rnn(x)
        if out.dim() == 3:
            out = out[:, -1, :]
        output = self.fc(out)
        return output

class DNN_Model(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DNN_Model, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        output = self.fc(x)
        return output
    
class MLP_Model(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_hidden_layers=2, dropout=0.3):
        super(MLP_Model, self).__init__()

        layers = [
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        ]

        for _ in range(num_hidden_layers - 1):
            layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])

        layers.append(nn.Linear(hidden_dim, output_dim))
        self.fc = nn.Sequential(*layers)

    def forward(self, x):
        return self.fc(x)

    
class Transformer_Model(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, output_dim):
        super(Transformer_Model, self).__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True),
            num_layers=num_layers
        )
        self.fc = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        x = self.embedding(x)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)
        output = self.fc(x)
        return output


class LSTM_Model(nn.Module):
    def __init__(self, input_dim, lstm_hidden_dim, lstm_layers, output_dim, dropout=0.3):
        super(LSTM_Model, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0
        )
        lstm_output_dim = lstm_hidden_dim
        self.fc = nn.Sequential(
            nn.Linear(lstm_output_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        if hasattr(self.lstm, 'flatten_parameters'):
            self.lstm.flatten_parameters()
        lstm_out, _ = self.lstm(x)
        if lstm_out.dim() == 3:
            last_hidden_state = lstm_out[:, -1, :]
        else:
            last_hidden_state = lstm_out
        output = self.fc(last_hidden_state)
        return output

class GRU_Model(nn.Module):
    def __init__(self, input_dim, gru_hidden_dim, gru_layers, output_dim, dropout=0.3):
        super(GRU_Model, self).__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=gru_hidden_dim,
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=False,
            dropout=dropout if gru_layers > 1 else 0
        )
        fc_input_dim = gru_hidden_dim * 1
        self.fc = nn.Sequential(
            nn.Linear(fc_input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        gru_out, _ = self.gru(x)
        if gru_out.dim() == 2:
            last_hidden_state = gru_out
        else:
            last_hidden_state = gru_out[:, -1, :]
        output = self.fc(last_hidden_state)
        return output


class BiLSTM_Model(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, dropout=0.2):
        super(BiLSTM_Model, self).__init__()
        self.lstm_layers = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        for i in range(num_layers):
            input_size = input_dim if i == 0 else hidden_dim * 2
            self.lstm_layers.append(nn.LSTM(input_size, hidden_dim, batch_first=True, bidirectional=True))
            self.dropouts.append(nn.Dropout(dropout) if dropout else nn.Identity())
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        for lstm, dropout in zip(self.lstm_layers, self.dropouts):
            lstm.flatten_parameters()
            x, _ = lstm(x)
            x = dropout(x)

        if x.ndimension() == 3:
            x = x[:, -1, :]
        elif x.ndimension() == 2:
            pass
        else:
            raise ValueError(f"Unexpected tensor dimensions: {x.shape}")

        output = self.fc(x)
        return output



class BiGRU_Model(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, dropout=0.2):
        super(BiGRU_Model, self).__init__()
        self.gru_layers = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        for i in range(num_layers):
            input_size = input_dim if i == 0 else hidden_dim * 2
            self.gru_layers.append(nn.GRU(input_size, hidden_dim, batch_first=True, bidirectional=True))
            self.dropouts.append(nn.Dropout(dropout) if dropout else nn.Identity())
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        for gru, dropout in zip(self.gru_layers, self.dropouts):
            gru.flatten_parameters()
            x, _ = gru(x)
            x = dropout(x)
        if x.ndimension() == 3:
            x = x[:, -1, :]
        elif x.ndimension() == 2:
            pass
        else:
            raise ValueError(f"Unexpected tensor dimensions: {x.shape}")

        output = self.fc(x)
        return output

class CNN_GRU_Model(nn.Module):
    def __init__(self, input_dim, cnn_out_channels, gru_hidden_dim, gru_layers, output_dim, dropout=0.3, gru_dropout=0.3):
        super(CNN_GRU_Model, self).__init__()

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

        self.gru = nn.GRU(
            input_size=cnn_out_channels,
            hidden_size=gru_hidden_dim,
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=False,
            dropout=gru_dropout
        )

        self.fc = nn.Sequential(
            nn.Linear(gru_hidden_dim * 1, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.cnn(x)
        x = x.transpose(1, 2)
        gru_out, _ = self.gru(x)
        output = self.fc(gru_out[:, -1, :])
        return output

class CNN_LSTM_Model(nn.Module):
    def __init__(self, input_dim, cnn_out_channels, lstm_hidden_dim, lstm_layers, output_dim):
        super(CNN_LSTM_Model, self).__init__()

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

        self.lstm = nn.LSTM(
            input_size=cnn_out_channels,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=False
        )
        self.layer_norm = nn.LayerNorm(lstm_hidden_dim * 1)
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden_dim * 1, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        cnn_out = self.cnn(x)
        cnn_out = cnn_out.transpose(1, 2)
        self.lstm.flatten_parameters()
        lstm_out, _ = self.lstm(cnn_out)
        lstm_out = self.layer_norm(lstm_out)
        pooled_out = torch.mean(lstm_out, dim=1)
        output = self.fc(pooled_out)
        return output

class CNN_BiGRU_Model(nn.Module):
    def __init__(self, input_dim, cnn_out_channels, gru_hidden_dim, gru_layers, output_dim, dropout=0.3, gru_dropout=0.3):
        super(CNN_BiGRU_Model, self).__init__()

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
        self.gru = nn.GRU(
            input_size=cnn_out_channels,
            hidden_size=gru_hidden_dim,
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=True,
            dropout=gru_dropout
        )
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
        output = self.fc(gru_out[:, -1, :])
        return output

class CNN_BiLSTM_Model(nn.Module):
    def __init__(self, input_dim, cnn_out_channels, lstm_hidden_dim, lstm_layers, output_dim):
        super(CNN_BiLSTM_Model, self).__init__()

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
        self.lstm = nn.LSTM(
            input_size=cnn_out_channels,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True
        )
        self.layer_norm = nn.LayerNorm(lstm_hidden_dim * 2)
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        cnn_out = self.cnn(x)
        cnn_out = cnn_out.transpose(1, 2)
        self.lstm.flatten_parameters()
        lstm_out, _ = self.lstm(cnn_out)
        lstm_out = self.layer_norm(lstm_out)
        pooled_out = torch.mean(lstm_out, dim=1)
        output = self.fc(pooled_out)
        return output


class CAttention(nn.Module):
    def __init__(self, hidden_size):
        super(CAttention, self).__init__()
        self.attn = nn.Linear(hidden_size, 1)

    def forward(self, x):
        attn_weights = torch.softmax(self.attn(x), dim=1)
        context_vector = torch.sum(attn_weights * x, dim=1)
        return context_vector, attn_weights


class CNN_Attention_Model(nn.Module):
    def __init__(self, input_dim, cnn_out_channels, output_dim):
        super(CNN_Attention_Model, self).__init__()

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

        self.attention = CAttention(cnn_out_channels)

        self.fc = nn.Sequential(
            nn.Linear(cnn_out_channels, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        x = x.unsqueeze(1) 
        cnn_out = self.cnn(x)
        cnn_out = cnn_out.transpose(1, 2)
        context_vector, _ = self.attention(cnn_out)
        context_vector = context_vector.view(context_vector.size(0), -1)
        output = self.fc(context_vector)
        return output


class LAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(LAttention, self).__init__()
        self.attention_weights = nn.Linear(hidden_dim * 2, 1)

    def forward(self, lstm_out):
        attention_scores = self.attention_weights(lstm_out)
        if attention_scores.dim() == 3:
            attention_scores = attention_scores.squeeze(-1)
        attention_scores = attention_scores / (lstm_out.size(-1) ** 0.5)
        attention_weights = torch.softmax(attention_scores, dim=1)
        context_vector = torch.sum(lstm_out * attention_weights.unsqueeze(-1), dim=1)

        return context_vector, attention_weights


class BiLSTM_Attention_Model(nn.Module):
    def __init__(self, input_dim, lstm_hidden_dim, lstm_layers, output_dim, dropout=0.3):
        super(BiLSTM_Attention_Model, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True
        )

        self.layer_norm = nn.LayerNorm(lstm_hidden_dim * 2)
        self.dropout = nn.Dropout(dropout)
        self.attention = LAttention(lstm_hidden_dim)
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        self.lstm.flatten_parameters()
        lstm_out, _ = self.lstm(x)
        lstm_out = self.layer_norm(lstm_out)
        lstm_out = self.dropout(lstm_out)
        context_vector, attention_weights = self.attention(lstm_out)
        output = self.fc(context_vector)

        return output


class GAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(GAttention, self).__init__()
        self.attention_weights = nn.Linear(hidden_dim * 2, 1)

    def forward(self, gru_out):
        attention_scores = self.attention_weights(gru_out)
        if attention_scores.dim() == 3:
            attention_scores = attention_scores.squeeze(-1)
        attention_scores = attention_scores / (gru_out.size(-1) ** 0.5)
        attention_weights = torch.softmax(attention_scores, dim=1)
        context_vector = torch.sum(gru_out * attention_weights.unsqueeze(-1), dim=1)

        return context_vector, attention_weights


class BiGRU_Attention_Model(nn.Module):
    def __init__(self, input_dim, gru_hidden_dim, gru_layers, output_dim, dropout=0.3, gru_dropout=0.3):
        super(BiGRU_Attention_Model, self).__init__()
        self.gru = nn.GRU(
            input_size=input_dim,         
            hidden_size=gru_hidden_dim,   
            num_layers=gru_layers,        
            batch_first=True,           
            bidirectional=True,           
            dropout=gru_dropout           
        )
        self.layer_norm = nn.LayerNorm(gru_hidden_dim * 2)
        self.dropout = nn.Dropout(dropout)
        self.attention = GAttention(gru_hidden_dim)
        self.fc = nn.Sequential(
            nn.Linear(gru_hidden_dim * 2, 128),  
            nn.ReLU(),                          
            nn.Dropout(dropout),                  
            nn.BatchNorm1d(128),                  
            nn.Linear(128, output_dim)          
        )

    def forward(self, x):
        gru_out, _ = self.gru(x)
        gru_out = self.layer_norm(gru_out)
        gru_out = self.dropout(gru_out)
        context_vector, attention_weights = self.attention(gru_out)
        output = self.fc(context_vector)
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

class ConvAttention(nn.Module):
    def __init__(self, hidden_dim, kernel_size=3):
        super(ConvAttention, self).__init__()
        self.conv = nn.Conv1d(hidden_dim * 2, hidden_dim * 2, kernel_size, padding=kernel_size//2)
        self.attention_weights = nn.Linear(hidden_dim * 2, 1)
    def forward(self, lstm_out):
        conv_out = self.conv(lstm_out.transpose(1, 2)).transpose(1, 2)
        attention_scores = self.attention_weights(conv_out).squeeze(-1)
        attention_weights = torch.softmax(attention_scores, dim=1)
        context_vector = torch.sum(lstm_out * attention_weights.unsqueeze(-1), dim=1)
        return context_vector, attention_weights

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads=8):
        super(MultiHeadAttention, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim=hidden_dim * 2, num_heads=num_heads, batch_first=True)
    def forward(self, lstm_out):
        attn_output, attn_weights = self.multihead_attn(lstm_out, lstm_out, lstm_out)
        context_vector = torch.mean(attn_output, dim=1)
        return context_vector, attn_weights

class CNN_BiLSTM_Attention_Model(nn.Module):
    def __init__(self, input_dim, cnn_out_channels, lstm_hidden_dim, lstm_layers, output_dim):
        super(CNN_BiLSTM_Attention_Model, self).__init__()
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
        self.lstm = nn.LSTM(
            input_size=cnn_out_channels,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True
        )
        self.layer_norm = nn.LayerNorm(lstm_hidden_dim * 2)
        self.attention = MultiHeadAttention(lstm_hidden_dim, num_heads=8)
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        cnn_out = self.cnn(x)
        cnn_out = cnn_out.transpose(1, 2)
        self.lstm.flatten_parameters()
        lstm_out, _ = self.lstm(cnn_out)
        lstm_out = self.layer_norm(lstm_out)
        context_vector, _ = self.attention(lstm_out)
        output = self.fc(context_vector)
        return output



class CNN_BiGRU_Attention_Model(nn.Module):
    def __init__(self, input_dim, cnn_out_channels, gru_hidden_dim, gru_layers, output_dim, dropout=0.3, gru_dropout=0.3):
        super(CNN_BiGRU_Attention_Model, self).__init__()

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
        self.gru = nn.GRU(
            input_size=cnn_out_channels,
            hidden_size=gru_hidden_dim,
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=True,
            dropout=gru_dropout
        )
        self.layer_norm = nn.LayerNorm(gru_hidden_dim * 2)
        self.attention = MultiHeadAttention(gru_hidden_dim, num_heads=8)
        self.fc = nn.Sequential(
            nn.Linear(gru_hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.BatchNorm1d(128),
            nn.Linear(128, output_dim)
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                if isinstance(self._modules[name.split('.')[0]], nn.Conv1d):
                    nn.init.kaiming_normal_(param, nonlinearity='relu')
                elif isinstance(self._modules[name.split('.')[0]], nn.GRU):
                    nn.init.xavier_normal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.cnn(x)
        x = x.transpose(1, 2)
        gru_out, _ = self.gru(x)
        gru_out = self.layer_norm(gru_out)
        context_vector, attention_weights = self.attention(gru_out)
        output = self.fc(context_vector)
        return output

class MultiHeadAttentionHalf(nn.Module):
    def __init__(self, hidden_dim, num_heads=8):
        super(MultiHeadAttentionHalf, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim=hidden_dim * 1, num_heads=num_heads, batch_first=True)

    def forward(self, lstm_out):
        attn_output, attn_weights = self.multihead_attn(lstm_out, lstm_out, lstm_out)
        context_vector = torch.mean(attn_output, dim=1)
        return context_vector, attn_weights

class CNN_LSTM_Attention_Model(nn.Module):
    def __init__(self, input_dim, cnn_out_channels, lstm_hidden_dim, lstm_layers, output_dim):
        super(CNN_LSTM_Attention_Model, self).__init__()
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
        self.lstm = nn.LSTM(
            input_size=cnn_out_channels,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=False
        )
        self.layer_norm = nn.LayerNorm(lstm_hidden_dim * 1)
        self.attention = MultiHeadAttentionHalf(lstm_hidden_dim, num_heads=4)
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden_dim * 1, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        cnn_out = self.cnn(x)
        cnn_out = cnn_out.transpose(1, 2)
        self.lstm.flatten_parameters()
        lstm_out, _ = self.lstm(cnn_out)
        lstm_out = self.layer_norm(lstm_out)
        context_vector, _ = self.attention(lstm_out)
        output = self.fc(context_vector)
        return output



class CNN_GRU_Attention_Model(nn.Module):
    def __init__(self, input_dim, cnn_out_channels, gru_hidden_dim, gru_layers, output_dim, dropout=0.3, gru_dropout=0.3):
        super(CNN_GRU_Attention_Model, self).__init__()

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
        self.gru = nn.GRU(
            input_size=cnn_out_channels,
            hidden_size=gru_hidden_dim,
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=False,
            dropout=gru_dropout
        )
        self.layer_norm = nn.LayerNorm(gru_hidden_dim * 1)
        self.attention = MultiHeadAttentionHalf(gru_hidden_dim, num_heads=4)
        self.fc = nn.Sequential(
            nn.Linear(gru_hidden_dim * 1, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.BatchNorm1d(128),
            nn.Linear(128, output_dim)
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                if isinstance(self._modules[name.split('.')[0]], nn.Conv1d):
                    nn.init.kaiming_normal_(param, nonlinearity='relu')
                elif isinstance(self._modules[name.split('.')[0]], nn.GRU):
                    nn.init.xavier_normal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.cnn(x)
        x = x.transpose(1, 2)
        gru_out, _ = self.gru(x)
        gru_out = self.layer_norm(gru_out)
        context_vector, attention_weights = self.attention(gru_out)
        output = self.fc(context_vector)
        return output
