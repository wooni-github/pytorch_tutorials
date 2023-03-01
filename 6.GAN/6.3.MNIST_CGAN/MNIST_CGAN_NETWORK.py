import torch.nn as nn

latent_size = 100

def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)

class CGAN_Discriminator(nn.Module):
    def __init__(self, image_size=784, hidden_size=256, latent_size=100, condition_size = 10):
        super(CGAN_Discriminator, self).__init__()
        self.hidden_size = hidden_size
        self.image_size = image_size
        self.latent_size = latent_size
        self.condition_size = condition_size

        self.linear1 = nn.Linear(self.image_size + self.condition_size, self.hidden_size)
        self.linear2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.linear3 = nn.Linear(self.hidden_size, 1)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.leaky_relu(self.linear1(x))
        x = self.leaky_relu(self.linear2(x))
        x = self.sigmoid(self.linear3(x))
        return x

class CGAN_Generator(nn.Module):
    def __init__(self, image_size=784, hidden_size=256, latent_size=100, condition_size = 10):
        super(CGAN_Generator, self).__init__()
        self.hidden_size = hidden_size
        self.image_size = image_size
        self.latent_size = latent_size
        self.condition_size = condition_size

        self.linear1 = nn.Linear(self.latent_size + self.condition_size, self.hidden_size)
        self.linear2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.linear3 = nn.Linear(self.hidden_size, self.image_size)
        self.Tanh = nn.Tanh()

        self.relu = nn.ReLU(0.2)

    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.Tanh(self.linear3(x))
        return x





