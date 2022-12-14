import torch
import torch.nn as nn

class FCL(nn.Module):
    def __init__(self):
        super(FCL, self).__init__()
        self.fc1 = torch.nn.Linear(1 * 28 * 28, 500, bias=True)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(500, 500, bias=True)
        self.fc3 = torch.nn.Linear(500, 10, bias=True)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        out = self.fc3(x)
        return out
