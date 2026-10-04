import torch.nn as nn
from configs import configs

class FFNN(nn.Module):

    def __init__(self, configs: configs):
        super().__init__()
        self.fc1 = nn.Linear(configs.hidden_dim, 4 * configs.hidden_dim)
        self.fc2 = nn.Linear(4 * configs.hidden_dim, configs.hidden_dim)
        self.dropout = nn.Dropout(configs.ffnn_dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x