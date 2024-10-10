# model.py
import torch.nn as nn

class MLPMixerModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, patch_size, num_blocks, output_dim):
        super(MLPMixerModel, self).__init__()
        self.mlp1 = nn.Linear(input_dim, hidden_dim)
        self.mixer_blocks = nn.Sequential(
            *[nn.Sequential(
                nn.Linear(hidden_dim, patch_size),
                nn.ReLU(),
                nn.Linear(patch_size, hidden_dim)
            ) for _ in range(num_blocks)]
        )
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.mlp1(x)
        x = self.mixer_blocks(x)
        return self.fc(x)
