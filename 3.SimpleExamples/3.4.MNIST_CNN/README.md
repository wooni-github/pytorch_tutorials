
<br>

## MNIST image classification : **C**onvolutional **N**eural **N**etwork (**CNN**)

[학습] 예제코드 [pytorch_tutorials/3.SimpleExamples/3.4.MNIST_CNN/MNIST_CNN_Train.py](https://github.com/wooni-github/pytorch_tutorials/blob/main/3.SimpleExamples/3.4.MNIST_CNN/MNIST_CNN_Train.py)

[추론] 예제코드 [pytorch_tutorials/3.SimpleExamples/3.4.MNIST_CNN/MNIST_CNN_Test.py](https://github.com/wooni-github/pytorch_tutorials/blob/main/3.SimpleExamples/3.4.MNIST_CNN/MNIST_CNN_Test.py)

앞선 `MLP`로 이미지 분류를 수행했던 예제를 합성곱 신경망 (**C**onvolutional **N**eural **N**etwork)로 바꿔 학습/추론하는 예제입니다.

--- 

**Network & Training**

```python
model = CNN().to(DEVICE)
```

네트워크 구조가 `MLP`에서 `CNN`으로 변경된 `MNIST_Network.py`를 살펴보죠.

핵심적인 부분만 살펴보면,

<br>

```python
self.keep_prob = 0.5
```

학습시 dropout을 적용할 변수를 할당합니다.

<br>

```python
self.layer1 = torch.nn.Sequential(
    torch.nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
    torch.nn.ReLU(),
    torch.nn.MaxPool2d(kernel_size=2, stride=2))
```        

Convolutional layer를 구현하는 부분입니다. `torch.nn.Sequential`은 내부의 것들을 순차적으로 적용하라는 의미입니다.

`conv -> ReLU -> Max pool`을 적용하겠다는 의미네요. `layer1, 2, 3`은 비슷하므로 설명은 생략하겠습니다. 

다만 in-out channel수에 유의해주세요!

<br>

```python
self.fc1 = torch.nn.Linear(4 * 4 * 128, 625, bias=True)

self.layer4 = torch.nn.Sequential(
    self.fc1,
    torch.nn.ReLU(),
    torch.nn.Dropout(p=1 - self.keep_prob))

self.fc2 = torch.nn.Linear(625, 10, bias=True)
```

`conv1, conv2, conv3`까지 적용하고 나면 feature map의 크기가 `4 x 4x 128`형태가 됩니다. 이를 일반적인 `CNN`기반 이미지 분류에 사용하려면
`FCL`을 이용해 1-D형태의 텐서로 펴 주어야 합니다.
 
`layer4`의 순서대로 `fc1`으로 피쳐맵 펴주기, `ReLU`, `Dropout`을 적용하고, 마지막으로 `fc2`로 총 10개의 클래스로 분류하기 위해 `FCL`을 한차례 더 적용해줍니다.  

<br>

```python
orch.nn.init.xavier_uniform_(self.fc1.weight)
```

중간에 있는 `torch.nn.init.xavier_uniform_`은 네트워크의 weights를 임의로 초기화 하지 말고, 자비에 초기화 하라는 의미입니다.

<br>

```python
def forward(self, x):
    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = x.view(x.size(0), -1)
    x = self.layer4(x)
    out = self.fc2(x)
    return out
```

`forward`에서 실질적인 네트워크의 흐름을 살펴보면

`input image x`에 `conv1, conv2, conv3`로 피쳐맵을 뽑아내고, 이를 `x.view`를 이용하여 `Flatten`해줍니다.

`Flatten`한 결과를 `layer4`에 넣어 `FCL`을 통과하게 하고 마지막으로 `fc`를 통해 총 10개의 클래스별 예측값을 반환하도록 합니다.

<br>

---

**Inference**

![CNN_image](MNIST_CNN.png)

```
FIRST.pth  : Accuracy 9.92 %
MIDDLE.pth  : Accuracy 98.78 %
LAST.pth  : Accuracy 99.39 %
```

결과를 확인해 보니 큰 차이는 아니지만, `MLP`보다는 약간이나마 `CNN`이 좋은 결과를 주네요!

물론, 서로 충분한 형태로 네트워크를 구성한 것은 아니기에 직접적인 비교는 어렵지만요!