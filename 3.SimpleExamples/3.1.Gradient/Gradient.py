import torch

print('### Gradient Descent 예제\n')

x = torch.ones(2, 2, requires_grad=True)
print('x', x)

y = x + 1
print('\ny = x + 1\n', y)

z = 2*y**2
print('\nz = 2*y^2 = 2(x+1)^2\n', z)

res = z.mean()
print('\nres = z.mean()\n', res)
# d(res)/dx_i 구하기

# res = (z1 + z2 + z3 + z4) /4
# z_i = 2y_i **2
# z_i = 2(x+1) **2
# dz_i/dx = 4(x+1) * (x+1)' = 4(x+1)

# res = {2(x_1+1)^(2) + 2(x_2+1)^(2) + 2(x_3+1)^(2) + 2(x_4+1)^(2)}/4.0
# d(res)/dx_1 = {4(x_1+1) + 0 + 0 + 0}/4.0 = x_1 + 1
# d(res)/dx_i = (x_i +1)

res.backward()
print('\nx.grad = d(res)/d(x_i)', x.grad) # 실제로는 아래의 nn이나 functional을 사용함

import torch.nn as nn
import torch.nn.functional as F
