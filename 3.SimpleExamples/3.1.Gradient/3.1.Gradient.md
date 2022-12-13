
<br>

## Gradient를 이용한 미분값 구하기

예제코드 [pytorch_tutorials/3.SimpleExamples/3.1.Gradient/Gradient.py](https://github.com/wooni-github/pytorch_tutorials/blob/main/3.SimpleExamples/3.1.Gradient/Gradient.py)

파이토치를 이용하여 미분값을 구해보는 예제입니다.

실제로 사용하는 경우는 거의 없을테지만, 알고는 있자! 라는 의미의 예제입니다.

<br>

```python
x = torch.ones(2, 2, requires_grad=True) 
```

우선 `[2 x 2]` 형태의 텐서를 준비합니다.

여기서 주의할 점은, 파이토치에서 텐서를 그냥 선언하게 되면 위 코드와 다르게 `requires_grad=False`가 기본으로 설정됩니다.

미분값을 구하기 위해선 이를 `True`로 설정해줍니다.

이후 $y = x + 1$ 과 $z = 2y^2 = 2(x+1)^2$ 를 코드로 구현하고, 최종적으로 `res`는 `z`의 평균을 할당합니다.

```python
y = x + 1
z = 2*y**2
res = z.mean()
```

이 때 $res = (z_1 + z_2 + z_3 + z_4) / 4$ 이며 ,

$z_i = 2y_i ^ 2 = 2(x_i + 1)^2$ 이 되고,

$dz_i/dx_i = 4(x_i + 1)$ 이, 

$res = {2(x_1+1)^2 + 2(x_2+1)^2 + 2(x_3+1)^2 + 2(x_4+1)^2}/4$ 입니다.

$dres/dx_i = [4(x_i+1) + 0 + 0 + 0]/4 = x_i + 1$이 되겠죠. $x_i = 1$로 통일되어 있으므로 결국 $dres/dx_i = 2$입니다.

```python
res.backward()
print(x.grad)
```

후에 본격적인 네트워크들을 구현할 때 나오겠지만, `res.backward()`를 통해 역전파(Back propagation)를 `res`에 대해 수행해줍니다.

이후 `x.grad`를 출력해보면 우리가 원하는 $dres/dx_i$인 `[[2, 2], [2, 2]]`가 나오는 것을 확인할 수 있습니다.



