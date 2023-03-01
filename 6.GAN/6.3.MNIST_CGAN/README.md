
<br>

## GAN : MNIST CGAN (Conditional GAN)

- [Training 예제](MNIST_CGAN_TRAIN.py)

- [Inference 예제](MNIST_CGAN_TEST.py)

<br>

이번에는 `Conditional GAN` 입니다. 

기본적으로 앞선 `MLP`, `CNN` 을 활용한 `GAN`은 임의의 `latent vector`로 부터 0~9의 `MNIST` 이미지를 생성하였습니다.

다만 `latent vector`의 어떤 차원의 값이 0~9 라는 숫자를 생성하는지는 여전히 모르는 상태죠.

`CGAN`은 이러한 한계점을 극복하고자, 학습/추론시 `0~9`라는 파라미터를 통해 그에 해당하는 이미지를 생성하도록 해줍니다.

마찬가지로 논문으로 보면 어라.. 이거 어떻게 코드로 구현하지? 싶지만, 막상 코드로 구현해보면 어렵지 않습니다.

핵심이 되는 내용은, 네트워크의 입력에 `조건`을 추가한다는 것입니다.

<br>
---
**NETWORK**

`MNIST_CGAN_NETWORK.py`를 살펴보겠습니다. 좀 더 간단한 설명을 위해 `6.1.MNIST_MLP_GAN`에서 사용한 `MLP`를 이용한 네트워크를 사용했습니다.

<br>

**Discriminator**

```python
class CGAN_Discriminator(nn.Module):
    def __init__(self, image_size=784, hidden_size=256, latent_size=100, condition_size = 10):
        super(CGAN_Discriminator, self).__init__()
        self.hidden_size = hidden_size
        self.image_size = image_size
        self.latent_size = latent_size
        self.condition_size = condition_size

        self.linear1 = nn.Linear(self.image_size + self.condition_size, self.hidden_size)
        self.linear2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.linear3 = nn.Linear(self.hidden_size, 1)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.leaky_relu(self.linear1(x))
        x = self.leaky_relu(self.linear2(x))
        x = self.sigmoid(self.linear3(x))
        return x
```  

단 한군데 다른 점이 보입니다. 

```python
# def __init__(self, image_size=784, hidden_size=256, latent_size=100):
def __init__(self, image_size=784, hidden_size=256, latent_size=100, condition_size = 10):
```

네트워크의 입력 부분에 `latent`이외에 `condition_size`라는 크기 10의 벡터를 입력으로 받네요.

10으로부터 바로 떠올릴 수 있는건 0~9로 이루어진 MNIST의 라벨이죠.

0~9로 표현된 라벨을 `[0, 0, 0, ... 1, 0]` 형태로 이루어진 one-hot 인코딩한 결과를 네트워크에서 사용한다는것입니다.

조금 더 상세하게 `CGAN`의 `D`를 설명하자면, real/fake 이미지가 real/fake인지만 판별하던 기존의 `GAN`에 한가지 조건을 추가해서, 
0~9로 라벨이 설정되어있는 real/fake 이미지가 0~9의 GT를 갖는 real/fake이미지 인지 구별하는 것입니다.

`D`의 네트워크 코드쪽에서 본다면, MNIST의 `784 (이미지크기) + 10(라벨)` 크기의 데이터를 입력으로 받아 real/fake인지 출력으로 내보냅니다. 

<br>

**Generator**

```python
class CGAN_Generator(nn.Module):
    def __init__(self, image_size=784, hidden_size=256, latent_size=100, condition_size = 10):
        super(CGAN_Generator, self).__init__()
        self.hidden_size = hidden_size
        self.image_size = image_size
        self.latent_size = latent_size
        self.condition_size = condition_size

        self.linear1 = nn.Linear(self.latent_size + self.condition_size, self.hidden_size)
        self.linear2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.linear3 = nn.Linear(self.hidden_size, self.image_size)
        self.Tanh = nn.Tanh()

        self.relu = nn.ReLU(0.2)

    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.Tanh(self.linear3(x))
        return x
```  

앞서 `G`는 `laternt vector` 로부터 `[W 28 x H 28 x C 1]` 크기의 이미지를 생성하였죠.

`D`와 비슷하게 `latent vector` 뿐만 아니라 `condition_size`를 입력으로 동시에 받아, 0~9 라는 라벨로부터 0~9에 해당하는 fake 이미지를 생성하도록 하는게 CGAN 입니다.

코드상으로는 `D`와 사실 크게 다를 것이 없습니다. `latent vector + 10(라벨)`을 입력으로 받고, 출력으로 fake image를 생성하는 직관적으로 이해가 가능한 코드입니다.

--- 
**Training**

네트워크에 들어가는 입력, 출력의 shape만 변경해주면 `MLP` 예제와 거의 동일합니다. 

<br>

```python
for i, (real_images, label) in enumerate(data_loader):
    real_images = real_images.reshape(BATCH_SIZE, -1).to(DEVICE)

    ...

    label_encoded = F.one_hot(label, num_classes=10).to(DEVICE) # 0~9로 표현되는 10개의 클래스 정보를 [1, 0, 0, ... 0] 의 벡터로 one-hot 인코딩
    real_images_concat = torch.cat((real_images, label_encoded), 1) # concat

    real_score = D(real_images_concat)
```

`MLP GAN`과 약간 달라집니다. 제너레이터에서 기존에는 label정보를 받아오지 않았는데, `CGAN`에서는 조건으로 활용해야 하기에 `label`을 받아옵니다.

이후 `label`을 one_hot 인코딩으로 크기 10의 벡터로 만들어주고, `torch.concat`을 통해 `[784]` 크기로 펴준 MNIST 이미지와 함께 더해주고 `D`를 학습합니다.


<br>

```python
# Generator 학습
z = torch.randn(BATCH_SIZE, latent_size).to(DEVICE)
z_concat = torch.cat((z, label_encoded), 1)
fake_images = G(z_concat)
fake_images_concat2 = torch.cat((fake_images, label_encoded), 1)

fake_score2 = D(fake_images_concat2)
```

`G`를 학습하는 경우에도 비슷하게 구성합니다.

동일하게 `latent vector z`로부터 `G(z)`를 통해 `가짜 이미지`를 생성하는 기존 `MLP GAN`에 `label`정보를 조건으로 추가한 상태로 `D(G(z))`를 학습합니다.


<br>

--- 
**Inference**

```python
label = torch.tensor(random.randint(0, 9)) # 0~9 사이의 임의의 숫자를 생성
label_encoded = F.one_hot(label, num_classes=10).to(DEVICE) # one-hot 인코딩
label_encoded = torch.unsqueeze(label_encoded, dim = 0)
z_concat = torch.cat((z, label_encoded), 1) # 0~9 사이의 임의의 숫자 + latent vector를 G의 입력으로 사용

fake_images = G(z_concat) # G를 통해 이미지 생성
```

추론시에도 거의 비슷한 작업을 수행합니다.

0~9 사이의 생성하고자 하는 숫자를 선택하고 (코드에서는 랜덤으로), one_hot 인코딩, 이후 `latent_vector`과 `concat`을 해주고 `G`에 넣어주면 됩니다.

<br>

!['MNIST_DCGAN`](MNIST_CGAN.png)

GT로 표시된 라벨에 맞는 숫자 이미지가 생성된 것을 확인할 수 있습니다.
