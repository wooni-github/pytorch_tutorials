
<br>

## MNIST image classification : **M**ulti-**L**ayer **P**erceptron (**MLP**) ~ **F**ully **C**onnected **L**ayers (**FCL**)

[학습] 예제코드 [pytorch_tutorials/3.SimpleExamples/3.3.MNIST_MLP/MNIST_MLP_Train.py](https://github.com/wooni-github/pytorch_tutorials/blob/main/3.SimpleExamples/3.3.MNIST_MLP/MNIST_MLP_Train.py)

[추론] 예제코드 [pytorch_tutorials/3.SimpleExamples/3.3.MNIST_MLP/MNIST_MLP_Test.py](https://github.com/wooni-github/pytorch_tutorials/blob/main/3.SimpleExamples/3.3.MNIST_MLP/MNIST_MLP_Test.py)

<br>


`[W 28 x H 28 x C 1]` 크기의 digit 0~9까지 10개의 클래스를 갖는 데이터셋 MNIST에 대한 MLP 예제입니다.

이미지 분류 (Image Classification) 하면 가장 나오는 예제죠!

이번 예제에서는 다중 퍼셉트론(**M**ulti **L**ayer **P**erceptron)를 이용합니다.


<br>

코드는 3개의 스크립트로 이루어져 있습니다.

* `MNIST_Network.py` : 분류를 위한 네트워크 구조를 담은 스크립트
* `MNIST_MLP_Train.py` : 학습을 진행하고 네트워크의 최종 weights를 저장하는 스크립트
* `MNIST_MLP_Test.py` : 저장한 weights를 로드하여 추론하는 테스트용 스크립트
 
 3개입니다.

<br>

--- 
**Network**

우선 `MNIST_Network.py`를 살펴보죠.
```python
class FCL(nn.Module):
    def __init__(self):
        super(FCL, self).__init__()
        self.fc1 = torch.nn.Linear(1 * 28 * 28, 500, bias=True)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(500, 500, bias=True)
        self.fc3 = torch.nn.Linear(500, 10, bias=True)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        out = self.fc3(x)
        return out
```

파이토치에서 대다수의 네트워크는 `클래스`로 구현합니다. 간단한 네트워크 구조는 클래스로 구현할 이유는 없지만, 클래스로 하지 않으면 알아보기가 어렵고, 반복되는 구조를 메서드로 간단하게 구현할 수 있는 클래스가 훨씬 편하기 때문이죠.

<br>

구조는 크게 아래와 같은 것들을 기억하면 됩니다 (매번 동일하니 외울 필요는 없고, 복붙을 사용합시다~!)

<br>

- 클래스는 nn.Module을 상속

`class FCL(nn.Module):`
`FCL`이라는 클래스를 선언하고 `nn.Module`을 상속받습니다.

<br>

- 생성자, `foward` 메서드 구현
- 생성자에서 `super(self).__init__()`으로 다시 상속
```python
class FCL(nn.Module):
    def __init__(self):
        super(FCL, self).__init__()
    def forward(self, x):
        pass
```

<br>

- 생성자에서는 초기 입력값을 매개변수로 받음 (분류하고자 하는 클래스 수 등)
- 생성자에서 네트워크의 모듈별 구현 (ex, conv1 = conv -> relu -> conv -> relu -> mp)

```python
self.fc1 = torch.nn.Linear(1 * 28 * 28, 500, bias=True)
self.relu = torch.nn.ReLU()
self.fc2 = torch.nn.Linear(500, 500, bias=True)
self.fc3 = torch.nn.Linear(500, 10, bias=True)
```

굳이 생성자에 넣을 필요는 없습니다만, 편리상 생성자에 다수의 메서드를 구현합니다.
예제에서는 인라인 1줄로 구현하였지만, 함수형태로 구현하여도 문제없습니다.

<br>

- `fc1`은 MNIST데이터셋의 입력 이미지 1장 `[W 28 x H 28 x C 1]`을 MLP의 입력으로 사용하기 위해 `Linear/Flatten/Dense`형으로 1D-Vector로 변환된 데이터가 들어올 때,

- `[Input channel 784 - Output channel 500]`로 **F**ully **C**onnected **L**ayer. 로 이어주는 구조입니다.

- `relu`는 활성화 함수로 `ReLU`를 사용하라는 의미,

- `fc2`는 `[Input channel 500 - Output channel 500]` FCL

- `fc3`은 `[Input channel 500 - Output channel 10]` FCL 입니다. 마지막 채널이 10인 이유는 0~9까지 숫자가 10개이기 때문이죠.


<br>

생각해보면 `fc1, relu, fc2, relu, fc3`로 간단하게 코드로 작성하면 되지 않아? 싶겠지만,

네트워크 구조가 복잡해지면 `fc1`등의 모듈을 여러 차례 반복해서 쓰게 됩니다. 심지어 fc1의 구조가 단순한 `Linear`뿐만 아닌 `conv-relu-conv-relu-conv-relu-pool`등으로 복잡하게 구성될 수 있고,
이를 메서드를 호출하는 것 만으로 반복적으로 사용할 수 있기 때문에 굳이 이렇게 구현하는 것 입니다.

<br>

- `foward` 메서드에서 학습/추론시에 사용되는 입력값(`x`)을 받고 네트워크의 `Input -> Model(Input) -> return output` 구조를 구현, 최종 네트워크의 출력값 반환 (당연히 여러 개 반환 가능)
```python
def forward(self, x):
    x = torch.flatten(x, 1)
    x = self.relu(self.fc1(x))
    x = self.relu(self.fc2(x))
    out = self.fc3(x)
    return out
```

`foward` 에 실질적인 네트워크 구조를 작성합니다.
우선, `flatten`을 이용해 `[W 28 x H 28 x C 1]`로 구성된 3차원 MNIST 이미지 데이터를 1차원으로 펴줍니다 `[784]`의 형태를 갖는 1차원 데이터가 되고, MLP의 입력 형태에 맞게 변환해줍니다.
이후 벡터(텐서)를 `fc1 -> relu -> fc2 -> relu -> fc3` 순으로 적용하고 최종 결과 `out`을 반환합니다.

<br>

---

**Training**

이후 `MNIST_MLP_Train.py` 입니다.

전체적인 구조는 [Linear regression](https://github.com/wooni-github/pytorch_tutorials/blob/main/3.SimpleExamples/3.2.LinearRegression/3.2.LinearRegression.md) 예제와 동일하지만, 일부 변경점이 있습니다.

```python
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")
```
우선 이전 예제와 다르게, 본격적으로 네트워크 구조가 복잡해지기에 `cpu`로만 학습/추론하기에는 무리가 있겠죠. 올바르게 `CUDA`를 설치하셨다면 `DEVICE`가 `cuda`로 할당됩니다.
이후 이를 `DEVICE`변수에 담아주고, 학습시에 사용합니다

* 파이토치의 연산은 반드시 같은 디바이스별로 진행해야 합니다.
* 보통 `이미지, GT(Class, Value 등), 네트워크` 3가지를 동일한 디바이스에(보통 `cuda`) 할당합니다.
* 결과 확인시에는 (보통) `cpu`에서 확인합니다. 

```python
model = FCL().to(DEVICE)
```
로드한 모델(`FCL`)과

```python
data, target = data.to(DEVICE), target.to(DEVICE) # Data -> Device
```
각 배치별 생성한 네트워크의 입력으로 들어가는 `data`와 손실함수의 `loss`값 계산을 위한 `GT`값을 동일한 `DEVICE`로 할당합니다.

<br>

```python
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data',
                   train=True,
                   download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=BATCH_SIZE, shuffle=True) # Training 단계에서는 shuffle 수행
```

앞서 [2. DataLoader](https://github.com/wooni-github/pytorch_tutorials/blob/main/2.DataLoader/2.DataLoader.md) 에서 살펴봤던 것 처럼,
데이터로더에서 `train, transform, batch_size, shuffle`을 설정해줍니다.

<br>

```python
criterion = nn.functional.cross_entropy
```

이미지 분류 예제이므로 손실함수로 크로스 엔트로피를 설정해줍니다.

<br>

```python
torch.save(model.state_dict(), 'FIRST.pth')  # 전체 모델 저장
```

학습을 진행하고 모델의 weights를 저장해주면 끝입니다.


<br>

```python
summary(model, input_size=(1, 28, 28))
```

<br>

한 가지 살펴볼 것은, `summary`를 이용해 모델의 구조를 아래와 같이 살펴볼 수 있다는 거죠.
```

----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Linear-1                  [-1, 500]         392,500
              ReLU-2                  [-1, 500]               0
            Linear-3                  [-1, 500]         250,500
              ReLU-4                  [-1, 500]               0
            Linear-5                   [-1, 10]           5,010
================================================================
Total params: 648,010
Trainable params: 648,010
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.02
Params size (MB): 2.47
Estimated Total Size (MB): 2.49
--------------------------------------------------------------

```

<br>

---

Training

```
Iteration [0/18760 = 0.0%], loss 2.32
Iteration [2000/18760 = 10.661%], loss 0.245
Iteration [4000/18760 = 21.322%], loss 0.071
Iteration [6000/18760 = 31.983%], loss 0.137
Iteration [8000/18760 = 42.644%], loss 0.053
Iteration [10000/18760 = 53.305%], loss 0.012
Iteration [12000/18760 = 63.966%], loss 0.009
Iteration [14000/18760 = 74.627%], loss 0.07
Iteration [16000/18760 = 85.288%], loss 0.012
Iteration [18000/18760 = 95.949%], loss 0.005
Save training results as : LAST.pth
```

학습 결과 비교를 위해 아무것도 학습하지 않은 랜덤값으로 초기화된 `FIRST.pth`, 학습이 25% 정도 진행되었을 때의 `MIDDLE.pth`, 전체 epoch중 가장 loss가 작았을 때의
weigths를 저장한 `LAST.pth` 3개를 저장하도록 하였습니다.
<br>

---
이후 `MNIST_MLP_Test.py` 입니다.

테스트시에도 사실 학습시와, `Linear Regression`예제와 크게 다를 바가 없습니다.

몇 가지 핵심 내용만 살펴보면

```python
model = FCL()
model = model.to(DEVICE) # Model -> Device
model.load_state_dict(torch.load(pth))
model.eval() # eval을 설정해줘야 dropout, batch_normalization 등을 해제함
```

- 학습과 동일하게 모델 클래스를 가져와 `DEVICE`에 올려줄 것.
- `.pth`파일을 `load_state_dict`로 로드할 것.
- `model.eval()`을 통해 네트워크를 추론용으로 설정하 것.

만 주의해주면 됩니다. 여기까지만 수행하면, 아무 문제 없이 추론이 가능합니다.

(* 저는 주로 웹캠 등의 실시간 영상에서의 작업을 수행하므로 추론시 `BATCH_SIZE`를 1로 설정하였지만, 필수적인 것은 아닙니다. 데이터셋을 검증용으로 확인하고자 할 때 등은 BATCH_SIZE를 자신의 개발환경에 맞게 최대로 설정하여 빠른 시간 내에 결과를 확인하는 것이 좋겠죠.)


```python
target = int(target.item())
pred = output.max(1, keepdim=True)[1].item() # .max를 수행하면 [제일 확률이 높은 값, 인덱스]를 반환함
```

추론하면 네트워크 구조상 `0~9`까지의 총 10개의 클래스에 속하는 확률을 `output`변수에 담아 반환합니다.
이중에 가장 높은 확률을 갖는 클래스가 무엇인지를 `output.max`를 통해 확인해줍니다.

<br>

---
**Inference**

![MLP_image](MNIST_MLP.png)

테스트 데이터 결과 5개를 보면 원하는 대로 `0~9`의 숫자를 잘 분류한 것을 확인할 수 있네요.

```
FIRST.pth  : Accuracy 8.06 %
MIDDLE.pth  : Accuracy 96.77 %
LAST.pth  : Accuracy 98.14 %
```

각 `.pth`별 결과를 확인해 봐도 꽤 괜찮은 수준입니다.

아무것도 학습하지 않은 랜덤값으로 초기화된 `FIRST.pth`는 기댓값인 `10%`에는 못미치지만 랜덤값이기에 당연하고,

의외로 `5 Epoch`만으로도 `96.77%`를 획득한게 신기하기도 하네요.


