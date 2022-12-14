import torch
import torch.nn as nn

class FCL_Regression(nn.Module):
    def __init__(self):
        super(FCL_Regression, self).__init__()
        self.fc1 = torch.nn.Linear(1 * 28 * 28, 500, bias=True)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(500, 500, bias=True)
        self.fc3 = torch.nn.Linear(500, 10, bias=True)
        self.fc4 = torch.nn.Linear(10, 1, bias=True)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        out = self.fc4(x)
        return out
