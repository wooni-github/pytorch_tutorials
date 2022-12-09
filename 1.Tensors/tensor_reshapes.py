import torch

print('\n### 파이토치 텐서 기본 연산 \n')

x = torch.rand(4, 4)
y = torch.rand(4, 4)
print('x = torch.rand(4, 4)');print(x,end = '\n')
print('y = torch.rand(4, 4)');print(y,end = '\n\n')

print('================================================== 텐서 shape 변환 ==================================================\n')
print('y.reshape(16) # [1 x 16]으로 변환');print(y.reshape(16),end = '\n')
print('y.view(16) # [1 x 16]으로 변환');print(y.view(16),end = '\n\n')

print('# ? 는 자동으로 계산해서 채워지는 값. 즉 약수로 값을 넣어줘야겠지.')
print('y.reshape(-1) # [1 x ?]으로 변환');print(y.reshape(-1),end = '\n')
print('y.view(-1) # [1 x ?]으로 변환');print(y.view(-1),end = '\n\n')

print('y.reshape(8, 2) # [8 x 2]로 변환');print(y.reshape(8, 2),end = '\n')
print('y.reshape(8, -1) # [8 x ?]로 변환');print(y.reshape(8, -1),end = '\n\n')

print('y.reshape(7, -1) # 오류') # 오류
print('y.reshape(2, 2, 4)');print(y.reshape(2, 2, 4),end = '\n')
print('y.reshape(2, 2, -1)');print(y.reshape(2, 2, -1),end = '\n\n')

print('================================================== reshape vs view ==================================================\n')
print('reshape :')
print('contiguous한 텐서 : view와 동일하게 작동 ( Input 텐서를 참조하고, 참조한 텐서를 reshape함 )')
print('non-contigous한 텐서 : Input 텐서의 shape를 변환한 복사본을 반환함',end = '\n')

print('view :')
print('contiguous한 텐서 : reshape와 동일')
print('non-contiugous한 텐서 : 에러 발생',end = '\n')

print('contiguous : torch에서 메모리상에 순차적으로 데이터가 들어있는 경우를 의미함',end = '\n')

a = torch.Tensor([[1, 2], [3, 4]])
print('a = torch.Tensor([[1, 2], [3, 4]])');print(a,end = '\n')

print('a # 배열이므로 메모리상에 1, 2, 3, 4 순으로 저장되어 있음 = a의 값은 배열 순서대로 1, 2, 3, 4 -> contiguous')
print('torch.Tensor.is_contiguous(a)', torch.Tensor.is_contiguous(a),end = '\n')
print('a.t()');print(a.t())
print('a.t() # a의 transpose 배열은 값이 1, 3, 2, 4 순이므로 메모리상에 순서대로 저장되어 있지 않음 non-contiguous')
print('torch.Tensor.is_contiguous(a.t())', torch.Tensor.is_contiguous(a.t()),end = '\n')

print('b = a.t() -> non-contiguous -> reshape하면 복사본이 반환, view는 에러 발생')
b = a.t()
print('torch.Tensor.is_contiguous(b)', torch.Tensor.is_contiguous(b),end = '\n')
c = b.reshape(-1)
print('c = b.reshape(-1)');print('c\n', c);print('c는 b의 복사본의 reshape를 참조함',end = '\n')
b[0][0] = 100
print('b[0][0] = 100 으로 값 변경',end = '\n')
print('b : [0, 0]이 100으로 변경 (참조)\n', b,end = '\n')
print('c : [0]이 그대로 (복사본의 참조)\n', c,end = '\n')

d = a.reshape(-1)
a[0][0] = 10
print('a[0][0] = 10 으로 값 변경')
print('d = a.reshape(-1)\n', d);print('d는 a의 reshape를 참조함',end = '\n')
print('a : [0, 0]이 10으로 변경(참조)\n', a,end = '\n')
print('d : [0]이 10으로 변경(참조)\n', d,end = '\n')
print('d = a.t().view(-1) : non-contigous 하므로 error')

print('================================================== concatenation, transpose, permutation ==================================================')
print('================================================== torch.cat (ex1) ==================================================\n')

x = torch.Tensor([[[0, 1, 2], [3, 4, 5], [6, 7, 8]],
    [[9, 10, 11], [12, 13, 14], [15, 16, 17]],
    [[18, 19, 20], [21, 22, 23], [24, 25, 26]]])
y = torch.Tensor([27, 28, 29])

print('x\n', x)
print('y\n', y,end = '\n\n')
print('x.view(-1)\n', x.view(-1),end = '\n\n')
print('x.view(-1).shape', x.view(-1).shape,end = '\n\n')
print('y.shape', y.shape,end = '\n\n')

print('x shape [27],  y shape [3] 을 0번째 채널로 concat -> [30]', end ='\n\n')
print('torch.cat((x.view(-1), y), 0)\n', torch.cat((x.view(-1), y), 0), end = '\n\n')
print('torch.cat((x.view(-1), y), 0).shape', torch.cat((x.view(-1), y), 0).shape,end = '\n\n')

print('================================================== torch.cat (ex2) ==================================================\n')
x1 = torch.Tensor([[[1, 2, 3]]])
x2 = torch.Tensor([[[7, 8, 9]]])
a = torch.cat((x1, x2), 0)
b = torch.cat((x1, x2), 1)
c = torch.cat((x1, x2), 2)
print('x1', x1)
print('x2', x2,end = '\n')

print('x1.shape : ', x1.shape)
print('x2.shape : ', x2.shape,end = '\n\n')

print('a = torch.cat((x1, x2), 0) : shape : ', a.shape,end = '\n')
print('a\n', a, end = '\n\n')

print('b = torch.cat((x1, x2), 1) : shape : ', b.shape,end = '\n')
print('b\n', b, end = '\n\n')

print('c = torch.cat((x1, x2), 2) : shape : ', c.shape,end = '\n')
print('c\n', c, end = '\n\n')

print('================================================== torch.permute ==================================================\n')
x = torch.Tensor([[[1, 2, 3], [4, 5, 6]]])
print('x = torch.Tensor([[[1, 2, 3], [4, 5, 6]]])\n', x, end = '\n\n')
print('x.shape : ', x.shape,end = '\n\n')

y = x.permute(1, 0, 2)
print('y = x.permute(1, 0, 2) # x shape : [0, 1, 2] 순으로 index를 매겼을 때 [1, 0, 2]순으로 변환하여 반환함 (정확하게는 참조하여 reshape하여 보여줄 뿐)',end = '\n')
print(y, end = '\n\n')
print('y.shape', y.shape, end = '\n\n\n')

print('***텐서 값 변환 x[0][0][0] = 100')
x[0][0][0] = 100
print('x\n', x, end = '\n')
print('y\n', y, end = '\n\n')

print('================================================== torch.transpose ==================================================\n')
print('z = torch.transpose(x, 0, 2) # x의 0번과 2번 차원을 교환하여 z로 반환')
z = torch.transpose(x, 0, 2)
print('x.shape', x.shape, end = '\n\n')
print('z\n', z, end = '\n')

print('z.shape', z.shape)

