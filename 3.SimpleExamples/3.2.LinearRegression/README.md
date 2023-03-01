
<br>

## [Linear regression](3.2.LinearRegression/linear_regression1.py) $y = ax + b$

$y = ax + b$ 에서 `[x, y]`쌍을 데이터로 주었을 때 `a, b`를 추정해내는 **선형회귀 (Linear regression)** 예제입니다.

![linear_regression1_image](linear_regression1.png)

파란색 점으로 데이터가 주어졌을 때, 학습을 진행할 수록 (Epoch가 늘어날수록), 점들을 대표할만한 선분으로 근사해 가는 것을 확인할 수 있습니다.

<br>

파이토치 코드로 이를 구현해보면, 

```python
n = 30
x_list = [uniform(-5, 5) for _ in range(n)]
y_list = [2 * x + 3 + uniform(-0.5, 0.5) for x in x_list]
x_list_torch = torch.FloatTensor(x_list)
y_list_torch = torch.FloatTensor(y_list)
```

우선 30개의 점 데이터를 생성합니다. 

`x`는 -5부터 5까지에서 랜덤한 값을,
`y`는 $y = 2x + 3 + noise$ 로 값을 할당합니다. 이후 파이토치의 텐서로 변환합니다.


<br>

```python
W = torch.zeros(1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)

optimizer = torch.optim.SGD([W, b], lr=0.015)
```

보통 선형회귀 문제에서, 예제에서는 `a, b`라고 작성하였지만, 코드상에서는 `Wx + b`형태로 구현합니다.

앞선 예제와 마찬가지로 `requires_grad = True`를 할당하여 역전파할수 있도록 변수들을 설정하고,

학습을 진행할 `optimizer`에 이들을 넣어줍니다. 다양한 `optimizer`가 존재하지만, 예제에서는 `SGD`를 사용하였습니다. 


<br>

```python
for epoch in range(1, np_epoch + 1):
    h = x_list_torch * W + b  # 예측값은 Wx + b

    cost = torch.mean((h - y_list_torch) ** 2)  # Mean Square Error를 사용

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()
    
y = W.item() * x + b.item()
```

이후 학습을 진행합니다. 

파이토치에서는 거의 무조건이라고 할 수 있을정도로 정해진 형태로 학습을 (코드상에서) 수행합니다.

<br>

```
1. 입력값으로부터 예측값 계산 (**H**ypothesis)
2. [입력값, 예측값]을 이용하여 손실함수로부터 loss 계산
3. optimizer.zero_grad()
4. cost.backward()
5. optimizer.step()
```
(대다수의 네트워크는 위와같이 이루어져 있어, 예제에서는 굳이 상세히 다루지 않겠습니다)

예제에서는 예측값과 참값의 차의 제곱을 평균한 **평균제곱오차** (**M**ean **S**quare **E**rror)를 사용했습니다.
 
코드를 작성하고 마지막으로 `W.item()`, `b.item()`으로 값을 확인해보면 $y = 2x + 3$에 근사한 결과를 확인해 볼 수 있습니다.

<br>

```
GT : y = 2x + 3 + (noise)
Epoch :   20, y = 2.1401x+1.3444 Cost 2.548604
Epoch :   40, y = 2.0831x+2.0455 Cost 0.878996
Epoch :   60, y = 2.0498x+2.4427 Cost 0.342886
Epoch :   80, y = 2.0310x+2.6678 Cost 0.170739
Epoch :  100, y = 2.0203x+2.7954 Cost 0.115461
```

<br>

## [Linear regression](linear_regression2.py) $y = ax^2 + bx + c$

이번에는 2차함수인 $y = 2x^2 - 4x + 3$를 선형회귀로 풀어보는 예제입니다.

![linear_regression2_image](linear_regression2.png)

마찬가지로 `[x, y]` 로 구성된 데이터를 입력으로 주었을 때, 학습이 진행될 수록 $y = ax^2 + bx + c$ 로 모델링한 변수 `[a, b, c]`를 데이터에 근사하게 찾아내는 모습을 볼 수 있습니다.

<br>

```python
x_train = torch.FloatTensor([[each_x ** 2, each_x, 1] for each_x in x])
y_train = torch.FloatTensor(y)

W = torch.zeros(3, requires_grad=True)
optimizer = torch.optim.SGD([W], lr=0.0001)
```

코드상으로 달라진 부분은 오직 이 부분 뿐입니다.

**선형**회귀 이기 때문에 데이터를 2차함수에 맞게 $[x^2, x, 1]$로 준비합니다.

또한, `a, b, c`에 해당하는 `W`를 1차원 (크기 3) 벡터로 설정하고 학습을 진행합니다.

<br>

```python
GT : y = 2x^2 - 4x + 3
epoch: 1000, y = 2.2042x^2 + -3.2121x + 0.3496 Cost 7.790498
epoch: 2000, y = 2.1683x^2 + -3.8202x + 0.5652 Cost 2.815152
epoch: 3000, y = 2.1505x^2 + -3.9402x + 0.7697 Cost 2.219505
epoch: 4000, y = 2.1371x^2 + -3.9655x + 0.9583 Cost 1.854214
epoch: 5000, y = 2.1253x^2 + -3.9723x + 1.1312 Cost 1.553194
```

* 이 예제는 2차함수도 선형회귀로 풀 수**도**있다. 라는 의미의 예제입니다.
* 진정한 의미에서 2차함수를 근사하는 예제는 아니라는거죠!
* $[x^2, x, 1]$로 $x$로만 구성된 변수를 2개의 변수 $[x^2, x]$으로 나누어 Multiple Linear Regression으로 변환해서 풀은 것입니다.
* 당연하게도 현실의 예제에서는 $x$와 $x^2$이 서로 독립이지 않기 때문에 (dependent함) Linear Regression으로는 해결할 수 없습니다.
* 2차함수의 비교적 간단한 예제이기에 통했을 뿐, 딥러닝적인 관점으로 봤을 때는 이러한 Polynomial regression문제에서는 MLP를 쓰는 것이 보다 적합하다고 생각됩니다. (a, b, c를 추정해내는 모델링은 아니지만요)


