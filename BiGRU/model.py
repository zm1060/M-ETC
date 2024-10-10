import torch.nn as nn

class BiGRUModel(nn.Module):
    def __init__(self, numerical_dims, hidden_dim, gru_layers, output_dim):
        super(BiGRUModel, self).__init__()
        
        # BiGRU layer
        self.gru = nn.GRU(
            input_size=numerical_dims, 
            hidden_size=hidden_dim, 
            num_layers=gru_layers, 
            batch_first=True, 
            bidirectional=True
        )
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),  # BiGRU is bidirectional, so hidden_dim * 2
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, output_dim)  # Output layer
        )
        
    def forward(self, X_num):
        X_num = X_num.unsqueeze(1)  # Add sequence dimension (batch_size, seq_length, input_size)
        gru_out, _ = self.gru(X_num)  # (batch_size, seq_length, hidden_dim * 2)
        gru_out = gru_out[:, -1, :]  # Use the output from the last time step
        out = self.fc(gru_out)
        return out
