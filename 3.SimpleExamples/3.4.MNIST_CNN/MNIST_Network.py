import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.keep_prob = 0.5

        # Input : [28 x 28 x 1]
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1), # [14 x 14 x 32]
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2)) # [7 x 7 x 32]

        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), # [7 x 7 x 64]
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2)) # [4 x 4 x 64]

        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), # [7 x 7 x 128]
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=1)) # [4 x 4 x 128]

        self.fc1 = torch.nn.Linear(4 * 4 * 128, 625, bias=True) # [4 x 4 x 128 to 625]
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        self.layer4 = torch.nn.Sequential(
            self.fc1, # [625]
            torch.nn.ReLU(),
            torch.nn.Dropout(p=1 - self.keep_prob))

        self.fc2 = torch.nn.Linear(625, 10, bias=True) # [625 to 10]
        torch.nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.view(x.size(0), -1)
        x = self.layer4(x)
        out = self.fc2(x)
        return out

