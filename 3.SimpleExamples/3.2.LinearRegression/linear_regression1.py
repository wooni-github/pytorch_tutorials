import torch
from random import *
import matplotlib.pyplot as plt
import numpy as np

n = 30
x_list = [uniform(-5, 5) for _ in range(n)]
y_list = [2 * x + 3 + uniform(-0.5, 0.5) for x in x_list]
# y = 2x + 3 + noise = Wx + b
print('y = 2x + 3')
x_list_torch = torch.FloatTensor(x_list)
y_list_torch = torch.FloatTensor(y_list)

W = torch.zeros(1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)

optimizer = torch.optim.SGD([W, b], lr=0.015)  # 나중에 W, b같은 변수가 아닌 Network 자체를 설계하면 Network.parameters()를 변수로 넣어줌.

x = np.linspace(min(x_list), max(x_list), 100)

np_epoch = 100
for epoch in range(1, np_epoch + 1):
    h = x_list_torch * W + b  # 예측값은 Wx + b

    cost = torch.mean((h - y_list_torch) ** 2)  # Mean Square Error를 사용
    if epoch % 20 == 0:
        print("Epoch : {:4d}, y = {:.4f}x+{:.4f} Cost {:.6f}".format(epoch, W.item(), b.item(), cost.item()))
    if epoch == 2:
        y = W.item() * x + b.item()
        plt.plot(x, y, '-g', label='Epoch 1')
    elif epoch == 5:
        y = W.item() * x + b.item()
        plt.plot(x, y, '-y', label='Epoch 5')
    elif epoch == 50:
        y = W.item() * x + b.item()
        plt.plot(x, y, '-m', label='Epoch 50')

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

y = W.item() * x + b.item()
plt.plot(x, y, '-r', label='Epoch 100')
plt.legend(loc='upper left')
plt.plot(x_list, y_list, 'o')
plt.grid()
plt.show()
