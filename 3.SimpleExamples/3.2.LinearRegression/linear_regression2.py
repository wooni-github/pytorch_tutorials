import torch
from random import *
import matplotlib.pyplot as plt
import numpy as np

n = 30
x = [-5 + 10 / n * i for i in range(n)]
y = [2 * (xx - 1) ** 2 + 1 for xx in x]
# y = 2(x-1)^2 + 1 = 2x*x - 4*x + 3 = ax*x + b*x + c
print('y = 2x^2 - 4x + 3')

x_train = torch.FloatTensor([[each_x ** 2, each_x, 1] for each_x in x])
y_train = torch.FloatTensor(y)

W = torch.zeros(3, requires_grad=True)

optimizer = torch.optim.SGD([W], lr=0.0001)

epochs = 5000
for epoch in range(1, epochs + 1):
    hypothesis = x_train.matmul(W)

    loss = torch.mean((hypothesis - y_train) ** 2)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    a, b, c = W.data.numpy()
    if epoch == 10:
        plt.plot(x, [a * xx ** 2 + b * xx + c for xx in x], '-g', linewidth=3, label='Epoch 100')
    elif epoch == 100:
        plt.plot(x, [a * xx ** 2 + b * xx + c for xx in x], '-y', linewidth=3, label='Epoch 100')
    elif epoch == 300:
        plt.plot(x, [a * xx ** 2 + b * xx + c for xx in x], '-m', linewidth=3, label='Epoch 300')
    if epoch % 1000 == 0:
        print("epoch: {:4d}, y = {:.4f}x^2 + {:.4f}x + {:.4f} Cost {:.6f}".format(epoch, a, b, c, loss.data))

plt.plot(x, [a * xx ** 2 + b * xx + c for xx in x], '-r', linewidth=3, label='Epoch 5000')

plt.plot(x, y, 'o')
plt.grid()

plt.legend(loc='upper left')

plt.show()
