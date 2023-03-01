



# 파이토치 텐서 기본 사용법

파이토치에서 사용하는 텐서의 기본적인 연산에 관한 내용입니다.

<br>

## 1. 텐서 생성


```python
print(  torch.empty(5, 4) )
print(  torch.ones(3, 3)  )
print(  torch.zeros(2)  )
print(  torch.rand(3, 5)  )
```

코드|내용
---|---|
```torch.empty(5, 4)``` | [5x4] 크기의 임의의 값으로 초기화 된 텐서 생성 |
```torch.ones(3, 3)``` | [3x3] 크기의 1.0으로 초기화 된 텐서 생성 |
```torch.zeros(2)``` | [2] 크기의 0.0으로 초기화 된 텐서 생성 |
```torch.rand(3, 5)``` | [0~1]사이의 임의의 값으로 초기화 된  [3x5]크기의 텐서 생성|


<br>


## 2. 텐서 변환

```python
l1 = [1, 2, 3] # list
l2 = np.array([4, 5, 6]) # numpy array
print(  torch.tensor(l1)  )
print(  torch.tensor(l2)  )
print(  torch.tensor(l2, dtype=torch.float32) )

l3 = [[1, 2], [3, 4], [5, 6]]
print(  torch.tensor(l3).size() )
print(  type(torch.tensor(l3))  )
```

코드|내용
---|---|
`torch.tensor(l1)`| 파이썬 리스트 혹은 numpy 배열로부터 텐서 생성
`torch.tensor(l2, dtype=torch.float32)`|타입을 지정한 텐서 생성
`torch.tensor(l3).size()`|텐서 크기 반환
`type(torch.tensor(l3))`|텐서 타입 반환

**코드를 직접 작성할 때, 형변환에 주의할것!!**


<br>


## 3. 텐서 연산 (곱, 행렬곱, 합)

```python
x = torch.rand(4, 4)
y = torch.rand(4, 4)

# 요소별 곱
print(  x*y )
print(  x.numpy()*y.numpy() )

# 행렬곱
print(  torch.matmul(x, y)  )
print(  np.matmul(x.numpy(), y.numpy()) )
```

코드|내용
---|---|
`x * y`|요소별 곱 반환
`x.numpy() * y.numpy()`|요소별 곱 numpy에서 계산 후 반환
`torch.matmul(x, y)`|행렬곱 반환
`np.matmul(x.numpy(), y.numpy())`|행렬곱 numpy에서 계산 후 반환

<br>

```python
print(  x + y ) # x + y 반환
print(  torch.add(x, y) ) # x + y 반환
print(  y.add(x)  ) # x + y 반환
print(  y ) # y는 변하지 않음

print(  y.add_(x) ) # y = x + y 반환
print(  y ) # y가 변함
```

코드|내용
---|---|
`x+y`| x + y 반환
`torch.add(x, y)`| x + y 반환
`y.add(x)`| x + y 반환
`y.add_(x)`| y = x + y 반환 **(y가 변경됨)**


<br>


## 4. 텐서 값 접근 및 변경

```python
print(  y[0, 0] )
print(  y[0, 0].numpy() )
z = torch.Tensor([1])
print(  z.item()  )
# print(  y.item()  )
y[0, 0] = 4.44
print(y[0, 0]) # 4.44
```

코드|내용
---|---|
`y[0, 0]`|텐서 y의 [0, 0]에 접근
`y[0, 0].numpy()`|텐서 y의 [0, 0]에 접근하여 값을 numpy로 반환
`z.item()`|텐서 z 스칼라 값 반환 **단독 값으로 구성된 스칼라 텐서만 ```item()``` 사용 가능**
`y[0, 0]=4.44`|텐서의 값 변경


<br>


# 파이토치 텐서 Reshape관련 

## 5. 텐서의 형태 변환 : reshape(), .view()

```python
x = torch.rand(4, 4)
y = torch.rand(4, 4)

print(  y.reshape(16) )
print(  y.view(16)  )
print(  y.reshape(-1) )
print(  y.view(-1)  ) 

print(  y.reshape(8, 2) )
print(  y.reshape(8, -1)  )

# print(  y.reshape(7, -1)  ) # 오류

print(  y.reshape(2, 2, 4)  )
print(  y.reshape(2, 2, -1) )
```

대다수의 경우에 `.reshape()`와 `.view`는 동일하게 사용됩니다.

메서드의 매개변수로 이루어진 형태로 텐서를 변형하여 반환합니다. (복사가 아닌 참조)

또한 매개변수에 -1을 주면 토치에서 알아서 값을 추정해서 사용합니다. (물론 약수를 넣어야겠죠)

<br>

메서드|contiguous|non-contiguous|
---|---|---|
`reshape()` | 입력 텐서를 **참조**한 채로 변환하고 반환  | 입력 텐서의 **복사본**을 변환하여 반환 |
`view()` | 입력 텐서를 **참조**한 채로 변환하고 반환 | **X** |

기본적으로 배열을 선언하면, 메모리상에 각 요소의 값이 순차적으로 저장됩니다.

이 때, 어떤 데이터(배열형태의)의 요소가 메모리상에 순서대로 저장되어 있다면 **contiguous**, 그렇지 않다면 **non-contiguous**라고 정의됩니다.


<br>

```python
a = torch.Tensor([[1, 2], [3, 4]])
print(  torch.Tensor.is_contiguous(a) ) # True
print(  torch.Tensor.is_contiguous(a.t()) ) # False
```


`a` 라는 배열`[[1, 2], [3, 4]]`을 선언하면 메모리상에는 `[1, 2, 3, 4]`가 순서대로 저장되어 있겠죠. 
`torch.Tensor.is_contiguous()` 로 `a`를 확인해보면 `True` 가 반환됩니다.

`a.t()` 는 `[[1, 3], [2, 4]]`가 되는데, 파이썬은 결국에 모든 객체를 참조하는 것을 기본으로 설계되어 있기에, `a.t()`의 요소의 메모리상의 위치를 살펴보면, 뒤죽박죽이 되어 있겠죠.
`torch.Tensor.is_contiguous()` 로 살펴보면 `False`가 반환됩니다.

<br>

```python
b = a.t()
c = b.reshape(-1)
b[0][0] = 100
print(b) # [[100, 2], [3, 4]] # 변경됨
print(c) # [[1, 2], [3, 4]] # 그대로
# d = a.t().view(-1) # 에러
```

위 표에 적힌 것 처럼, `non-contiguous`한 텐서에 대해서는 `reshape`가 복사본을 반환합니다. 즉, `b`를 수정해도 `c`가 변하지 않죠.
또한, `non-contiguous`하므로 `view()`는 사용할 수 없습니다.

<br>

```python
d = a.reshape(-1)
a[0][0] = 10
print(a) # [[10, 2], [3, 4]] # 변경됨
print(d) # [[10, 2], [3, 4]] # 변경됨
```

`contiguous`한 텐서에 대해서는 `reshape`가 입력 텐서를 참조하므로, 입력 텐서를 수정하면 반환 텐서도 같이 수정됩니다. 


<br>


## 6. 텐서 합치기 (concatenation) : torch.cat())

* Example 1
```python
x = torch.Tensor([[[0, 1, 2], [3, 4, 5], [6, 7, 8]],
    [[9, 10, 11], [12, 13, 14], [15, 16, 17]],
    [[18, 19, 20], [21, 22, 23], [24, 25, 26]]])
y = torch.Tensor([27, 28, 29])

print(  x.view(-1).shape  ) # [27]
print(  y.shape  ) # [3]

print(  torch.cat((x.view(-1), y), 0  )
print(  torch.cat((x.view(-1), y), 0).shape )
```

<br>

* Example 2 

```python
x1 = torch.Tensor([[[1, 2, 3]]]) # [1 x 1 x 3]
x2 = torch.Tensor([[[7, 8, 9]]]) # [1 x 1 x 3]
a = torch.cat((x1, x2), 0)
b = torch.cat((x1, x2), 1)
c = torch.cat((x1, x2), 2)
print(  a.shape ) # [2 x 1 x 3]
print(  b.shape ) # [1 x 2 x 3]
print(  c.shape ) # [1 x 1 x 6]
```

메서드|설명|
---|---|
`torch.cat([텐서1, 텐서2], dim = 차원)` | 텐서1과 텐서2를 '차원' 기준으로 합침|

<br>

`cat`의 사용 예는 Example 1, 2를 통해 쉽게 살펴볼 수 있습니다. `dim`으로 지정된 차원 기준으로 텐서를 합쳐 반환합니다.

여전히 `reshape, view`와 마찬가지로 값을 참조하기 때문에 원본텐서 `x1, x2`의 값을 바꾸면 동일하게 변경됩니다.

Resnet등의 skip-connection이나 feature concatenation에 매우 많이 사용되는 메서드죠!!


<br>


## 7. 행렬 차원 바꾸기, 차원 순서 바꾸기 (Permute), 전치 (Transpose)

* Example 3

```python
x = torch.Tensor([[[1, 2, 3], [4, 5, 6]]])
print(  x )
print(  x.shape ) # [1 x 2 x 3]

y = x.permute(1, 0, 2)
print(  y )
print(  y.shape )  # [2 x 1 x 3]
```
<br>

* Example 4
``` python
z = torch.transpose(x, 0, 2)
print(  z )
print(  z.shape ) # [3 x 2 x 1]
```

메서드|설명|
---|---|
`텐서.permute(dims = 1, 2, 3, ...)` | 텐서를 dims 기준으로 차원 순서를 변경합니다|
`torch.transpose(입력 텐서, 변경 차원1, 변경 차원2)` | 입력 텐서의 변경 차원1과 2를 서로 변경합니다|

Example 3을 보면 쉽게 이해가 가실거에요!

`x`는 `[1 x 2 x 3]` 형태의 텐서입니다. 0번째 차원부터 살펴보면, 0번째 차원 : 1, 1번째 차원 : 2, 2번째 차원 : 3의 크기를 갖고있습니다.

이를 `.permute(1, 0, 2)` 적용하면, 0번째 차원에 (기존의)1번째 차원, 1번째 차원에 (기존의) 0번째 차원, 2번째 차원에 (기존의) 2번째 차원을 넣겠다는 의미입니다.


<br>

Example 4는 더 쉽죠. `torch.transpose`에 지정된 텐서의 입력차원 2개를 서로 바꿔주겠다는 의미입니다.


`Permute`와 `Transpose`는 보통, openCV등에서 입력받은 이미지 [h x w x c] 데이터를 파이토치가 사용할 수 있도록 [c x h x w]로 변경할 때 주로 사용됩니다.
