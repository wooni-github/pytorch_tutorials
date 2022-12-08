# 참고 : https://github.com/wooni-github/pytorch_tutorials/blob/main/1.Tensors/1.Tensors.md

import torch
import numpy as np

print('\n### 파이토치 텐서 기본 연산 \n')

print('================================================== 텐서 생성 ==================================================\n')
print('torch.empty(5, 4) # [5 x 4] 크기, 임의의 값으로 초기화')
print(torch.empty(5, 4), end='\n')
print('torch.ones(3, 3) # [3 x 3] 크기, 1로 초기화')
print(torch.ones(3, 3), end='\n')
print('torch.zeros(2) # [0, 2] 크기, 0으로 초기화')
print(torch.zeros(2), end='\n')
print('torch.rand(3, 5) # [3 x 5] 크기, [0, 1] 사이의 랜덤값으로 초기화')
print(torch.rand(3, 5), end='\n\n')

print('================================================== 텐서 변환 ==================================================\n')
l1 = [1, 2, 3] # list
l2 = np.array([4, 5, 6]) # numpy array
print('l1 = [1, 2, 3] # python list\n', l1, end='\n')
print('l2 = np.array([4, 5, 6]) # numpy array\n', l2, end='\n')
print('torch.tensor(l1) # python list -> tensor로 변환 후 반환')
print(torch.tensor(l1), end='\n')
print('torch.tensor(l2) # numpy array -> tensor로 변환 후 반환 * torch.int32로 반환된 것에 주의')
print(torch.tensor(l2), end='\n')
print('** list, numpy에서 변환시 dtype 주의할것 !! ')
print('torch.tensor(l2, dtype=torch.float32) # numpy array -> tensor(float32) 로 변환 후 반환')
print(torch.tensor(l2, dtype=torch.float32), end='\n')

l3 = [[1, 2], [3, 4], [5, 6]]
print('l3 = [[1, 2], [3, 4], [5, 6]]\n', l3, end='\n')
print('torch.tensor(l3).size()')
print(torch.tensor(l3).size(), end='\n')
print('type(torch.tensor(l3))')
print(type(torch.tensor(l3)), end='\n\n')

print('================================================== 텐서 연산 (곱, 행렬곱) ==================================================\n')
x = torch.rand(4, 4)
y = torch.rand(4, 4)
print('x');print(x, end='\n')
print('y');print(y, end='\n')

print('x*y # 행렬 요소별 곱셈') # 행렬 각 요소별 곱셈
print(x*y, end='\n')
print('x.numpy()*y.numpy() # 행렬 요소별 곱셈')
print(x.numpy()*y.numpy(), end='\n')

print('torch.matmul(x, y) # 행렬곱') # 행렬 곱셈
print(torch.matmul(x, y), end='\n')
print('np.matmul(x.numpy(), y.numpy()) # 행렬곱')
print(np.matmul(x.numpy(), y.numpy()), end='\n\n')

print('================================================== 텐서 연산 (합) ==================================================\n')
print('x + y == torch.add(x + y) == y.add(x) 모두 x + y를 반환함')
print('x + y # 합 반환');print(x + y, end='\n')
print('torch.add(x, y) # 합 반환');print(torch.add(x, y), end='\n')
print('y.add(x) # 합 반환');print(y.add(x), end='\n')

print('y.add_(x) # y = y + x 반환하면서 y 변환됨')
print('y before');print(y, end='\n')
print('y.add_(x) # y + x 반환')
print(y.add_(x), end='\n')
print('y after # y가 y + x 가 됨')
print(y, end='\n\n')

print('================================================== 텐서 접근 ==================================================\n')
print('y[0, 0] # 특정 텐서에 접근', y[0, 0])
print('y[0, 0].numpy() # 텐서의 값 반환', y[0, 0].numpy())
z = torch.Tensor([1])
print('z = torch.Tensor([1])')
print('z.item() # 텐서의 값 반환 (스칼라 한정 / 1개의 값)', z.item())
print('y.item() # 에러', end='\n')

print('y[0, 0] = 4.44 # 텐서 값 변환')
y[0, 0] = 4.44;print()

print('y[0, 0] # 특정 텐서에 접근', y[0, 0])
print('y[0, 0].numpy() 텐서의 값 반환', y[0, 0].numpy())


