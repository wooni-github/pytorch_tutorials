
<br>

# 파이토치 데이터로더 (DataLoader) 사용법

파이토치에서 사용하는 데이터로더 (Dataloader)의 기본 사용법에 관한 내용입니다.

데이터로더는 파이토치에서 일반적으로 사용되는 Batch별 데이터를 생성해주는 클래스입니다.

굳이 사용하지 않고 직접 자신만의 방법으로 Batch별 데이터를 네트워크의 입력으로 사용해도 되지만, 굳이 편리한 방법을 놔두고 다른 방법을 사용할 필요는 없죠!

<br>

## [1. DataLoader](DataLoader.py) 

우선 예제 데이터셋으로 파이토치에서 코드만으로 바로 다운로드 및 사용할 수 있는 CIFAR-10 데이터셋을 다운로드 해 줍니다.

```python
import torchvision
import torchvision.transforms as tr # Data를 불러오면서 바로 전처리를 하게 할 수 있도록 해줌
from torch.utils.data import DataLoader, Dataset # DataLoader : BatchSize형태로 만들어줌. Dataset : 튜닝에 사용

transf = tr.Compose([tr.Resize((8, 8)), tr.ToTensor()]) # 전처리. 순서대로 작업함 (8,8) resize -> tensor로 전환
# transf = tr.Compose([tr.Resize(8), tr.ToTensor()]) # 이렇게 해도 됩니다!

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transf)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transf)
```

`transform`은 꼭 사용할 필요는 없지만, 일반적인 네트워크들을 사용한다면 반드시 사용하게 됩니다.

`transform`은 나중에 설명하겠지만, 로드한 데이터셋의 이미지를 변형해주는 의미에서의 `transform`입니다.

crop, resize, rotate, flip, normalization 등의 다양한 이미지 연산을 코드 한 줄로 할 수 있는 파이토치의 매우 큰 장점이죠. 

데이터셋은 `train = True` 혹은 `train = False`로 구분하여 학습용, 검증용으로 별도 아운로드해주고, 다운로드 이후 `transform = tranf`로 transform을 적용해줍니다.

**CIFAR-10 데이터셋**은

* 학습 데이터셋 : `[W 32 x H 32 x C 3]`의 컬러 이미지 50000장
* 검증 데이터셋 : `[W 32 x H 32 x C 3]`의 컬러 이미지 10000장

을 제공합니다. 코드에서의 `transf`는 `[W 8 x H 8]`로 이미지 크기를 변경하고, 파이토치의 텐서로 변경하라는 의미입니다. 다수의 `transform`을 적용하기 위해 `tr.Compose`를 사용하였습니다.

```python
print(trainset[0][0].size())
```

를 확인해보면 `[C 3 x H 8 x W 8]` 로 출력됩니다. OpenCV등에서 `[H x W x C]`순으로 되어있는 이미지를 파이토치에서 사용하기 위해 자동적으로 이전 1.Tensors에서 봤던 것 처럼 `[C H W]`로 변환하여 자동으로 로드되었네요.

 
```python
trainloader = DataLoader(trainset, batch_size=50, shuffle=True)
testloader = DataLoader(testset, batch_size=50, shuffle=False)

print('len(trainloader)', len(trainloader))

dataiter = iter(trainloader)
images, labels = dataiter.next()
print('images.size() : [batch x channel x height x width]', images.size())

for i in range(100): # for max iteration
    images, labels = dataiter.next()
    print('iteration', i, images.shape)
```

본격적으로 데이터셋을 사용하기 위해서 `DataLoader`에 `trainset, testset`을 넣어줍니다.

`batch_size = 50`을 통해 배치별 생성할 데이터 크기를 정해주고, `shuffle`을 정해줍니다.

보통 학습할 때는 데이터를 무작위로 섞어주고 (`shuffle = True`), 추론할 때는 입력 받은 데이터 순서대로 확인합니다.

이후에는 파이썬의 이터레이터 `iter`와 `.next()`를 통해 배치별 데이터를 꺼내줍니다.

데이터는 당연히 `[B x C x H x W]` 로 앞선 `[C X H x W]`로 구성된 하나의 데이터를 배치 갯수만큼 누적한 것입니다.



<br>

## [2. ImageFolder](DataLoader_ImageFolder.py)

파이토치에서 제공하는 또다른 매우 편리한 `ImageFolder`입니다

```sh
├─example_dataset
│  ├─ants
│  │      0013035.jpg
│  │      5650366_e22b7e1065.jpg
│  │      6240329_72c01e663e.jpg
│  │      6240338_93729615ec.jpg
│  │      6743948_2b8c096dda.jpg
│  └─bees
│          16838648_415acd9e3f.jpg
│          17209602_fe5a5a746f.jpg
│          21399619_3e61e5bb6f.jpg
│          29494643_e3410f0d37.jpg
│          36900412_92b81831ad.jpg
│          39672681_1302d204d1.jpg
│          39747887_42df2855ee.jpg
│          85112639_6e860b0469.jpg
│          509247772_2db2d01374.jpg
```

예제 데이터를 이렇게 준비했을 때, `ImageFolder`는 자동으로 폴더명을 이미지의 클래스(라벨)이라고 설정하고, 폴더 내의 이미지를 로드하여 앞선 `DataLoader`에 로드해줍니다.

예제 데이터는 ants 이미지 5장, bees 이미지 9장으로 구성된 총 14장의 이미지를 넣어두었습니다.


 ```python
transf = tr.Compose([tr.CenterCrop(100), tr.Resize(16), tr.ToTensor()])
# 이미지별 사이즈가 상이하기 때문에 Crop이나 Resize없이 batch형태로 만들면 에러 발생

trainset = torchvision.datasets.ImageFolder(root='./example_dataset', transform=transf)
trainloader = DataLoader(trainset, batch_size=2, shuffle=False)
print('len(trainset)', len(trainset)) # 5 + 9 = 14개의 이미지

dataiter = iter(trainloader)
images, labels = dataiter.next()
print('images.size() : [batch x channel x height x width]', images.size())
```

여기서 한가지 주의할 점은, `transform`입니다. 데이터셋으로 서로 다른 크기를 갖는 이미지를 준비하였기 때문에, batch별 데이터를 그대로 생성하게 되면 batch 내의 데이터가 서로 다른 크기를 갖게 됩니다. 따라서 예제 데이터를 사용할 때 Resize혹은 Crop등을 사용하여 데이터의 크기를 맞춰줍니다.


<br>

## [3. Custom DataLoader](custom_DataLoader.py)

이번에는 파이토치를 이용해서 본격적인 딥러닝을 사용하기 위해 반드시 숙지해야 할, 자신만의 `DataLoader`를 사용하는 방법입니다.

`ImageFolder`같은 편리한 툴이 있지만, 사실 모든 데이터셋이 그에 맞게 데이터를 제공하는 것이 아니기에, 자신만의 `DataLoader`를 이용하는 건 불가피하다고 할 수 있죠.


```python
train_images = np.random.randint(256, size=(20, 32, 32, 3))
train_labels = np.random.randint(2, size=(20, 1)) 

print('train_images.shape [Batch x Height x Width x Channel] : ', train_images.shape)
print('train_labels.shape [Batch x Label] : ', train_labels.shape)
```

우선 예제가 될만한 데이터셋을 생성해줍니다. `numpy`를 이용해서 `0~255`값을 갖는 `[H 32 x W 32 x C 3]`의 데이터 `training_images` `20개`와 `0 또는 1`의 값을 갖는 라벨 `train_labels` `20개`를 생성해줍니다.  


```python
class TensorData(Dataset): # from torch.utils.data import DataLoader, Dataset 의 Dataset을 상속받음
    def __init__(self, x_data, y_data):
        self.x_data = torch.FloatTensor(x_data)
        self.x_data = self.x_data.permute(0, 3, 1, 2) # [Batch x Channel x Width x Height]로 바꿔줌
        self.y_data = torch.LongTensor(y_data) # Long타입으로
        self.len = self.y_data.shape[0] # 데이터갯수

    def __getitem__(self, index):
        # getitem을 통해 batch데이터 생성 (이 예제에선 입력 이미지, 결과 클래스)
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        # len을 통해 전체 데이터 수 반환
        return self.len
```

기본적으로는 custom `DataLoader`는 3개의 메서드 init, getitem, len를 갖도록 설계합니다.


메서드|내용|
---|---|
__init__|생성자. 데이터셋의 기본 설정. transform을 정의해주기도 함|
__getitem__|`DataLoader`가 동작할 때 1개의 데이터에 접근할 때 **반환할 값** 정의|
__len__|전체 데이터 갯수 반환|

위 예제 코드에서는 `__init__`에서 x데이터(이미지)와 y데이터(라벨)을 입력으로 받습니다.

이후 입력 np로 정의된 입력 데이터를 torch의 FloatTensor로 바꿔주고, `[B x H x W x C]`로 되어있는 형식을 `[B x C x H x W]`로 `permute`를 이용해 바꿔줍니다.

또한 y데이터를 `LongTessor`(임의로)로 바꿔주고, 멤버변수에 len을 할당해줍니다.

<br>

`__getitem__`에서는 1개의 데이터에 대한 정의를 해줍니다. 예제 데이터는 [이미지, 라벨]로 학습을 수행할 지도학습(Supervised learning)이므로, 한 번의 데이터 조회시 `[이미지]`와 `[라벨]`을 반환해주어야합니다.

`__getitem__`의 매개변수로 `index`가 있는데, 이와 연관하여 `index`번째 x데이터와 y데이터를 반환하도록 합니다.

<br>

`__len__`은 말씀드린 것 처럼 전체 데이터의 갯수를 반환합니다.

여기까지 설정해준다면 기본적인 `DataLoader`클래스를 구성 완료입니다.

<br>

## 4. Transform

### [4.1. custom Transform](custom_DataLoader_transform.py)

```python
class MyDataset(Dataset):
    def __init__(self, x_data, y_data, transform=None):
        self.x_data = x_data
        self.y_data = y_data
        self.transform = transform
        self.len = len(y_data)

    def __getitem__(self, index):
        sample = self.x_data[index], self.y_data[index]

        if self.transform: # transform 할당시 적용
            sample = self.transform(sample)
        return sample

    def __len__(self):
        return self.len        
```

이번에는 데이터로더에 `custom transform`을 수행해 보겠습니다.

이 경우, 생성자에 `transform`을 매개변수로 받는 것이 보통이고, `__getitem__` 메서드에서 생성한 `index`별 데이터에 `transform`을 적용해주도록 하는 것이 일반적입니다.

```python
class ToTensorMine:
    def __call__(self, sample):
        # Pytorch tensor형태로 데이터셋 변환
        inputs, labels = sample
        inputs = torch.FloatTensor(inputs)
        inputs = inputs.permute(2, 0, 1) # [ H W C ] => [ C, H ,W ]
        return inputs, torch.LongTensor(labels)

class LinearTensor:
    def __init__(self, slope = 1, bias = 0):
        self.slope = slope
        self.bias = bias

    def __call__(self, sample):
        inputs, labels = sample
        inputs = self.slope*inputs + self.bias
        return inputs, labels

trans = tr.Compose([ToTensorMine(), LinearTensor(2, 5)]) # 두 개의 다른 transform을 묶을 때 Compose 활용
# trans = tr.Compose([tr.ToTensor, LinearTensor(2, 5)]) # tr.ToTensor같은 transform자체의 것들은 이미지가 PIL형태여야 함. 우리꺼는 numpy 형태이고.
ds1 = MyDataset(train_images, train_labels, transform=trans)
```

앞서 말씀드린 대로, `tr.Compose`는 여러 개의 `transform`을 연속해서 적용하고자 할 때 사용합니다.

예제에서는 직접 생성한 `ToTensorMine`과 `LinearTensor`을 연속해서 적용한다는 의미죠.

`custom transform`을 적용할 때에는 예제처럼 클래스를 선언하고, `__call__`메서드에 매개변수로 받은 입력 데이터에 `transform`이 적용된 결과를 반환할 수 있도록 합니다.

`ToTensorMine`의 경우, 입력 데이터 `[이미지, 라벨]`을 받아서 이미지는 `FloatTensor`로 변환하고 `.permute`수행, 

라벨은 `LongTensor`로 반환하도록 하고 있네요.

`LinearTensor`같은 경우에는 이미지의 픽셀 값에 (RGB) `slope`를 곱하고 `bias`를 더하도록 하고 있네요 (아무런 의미도 없습니다! normalization에 사용할 수 있지만, 굳이 이렇게 할 이유가 없습니다)


<br>

### [4.2. custom TransformPIL](custom_DataLoader_transformPIL.py)

```python
class MyTransform:
    def __call__(self, sample):
        inputs, labels = sample
        inputs = torch.FloatTensor(inputs) # [H x W x C]
        inputs = inputs.permute(2, 0, 1) # [C x H x W]
        # 이렇게 하면 tr.ToPILImages등의 transform관련 함수들 사용 가능
        # tr.Resize(128)은 tr.Resize((128, 128))와 동일

        transf = tr.Compose([tr.ToPILImage(), tr.Resize(128), tr.ToTensor(), tr.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        final_output = transf(inputs)

        labels = torch.FloatTensor(labels)
        return final_output, labels
```
 
 이번에는 조금 특이한 예제입니다. (실제로 사용할 일은 없을거에요)
 
 `custom transform`에 입력 이미지를 받아 `FloatTensor`로 변환하고 `permute`를 적용하여 `[C x H x W]`의 파이토치가 사용할 수 있는 형태로 이미지를 변환한 상태입니다.
 
 이후에 `transf`로 정의된 것을 살펴보면 조금 독특하죠.
 
 우선 `ToPILImage`는 파이토치의 `[C x H x W]`로 구성된 이미지를 다시 원본 (PIL)이 사용할 수 있도록 반대로 `[H x w x C]`로 변환해주는 `transform`입니다.
 
 예제니까 한번 넣어봤을 뿐, 실제로는 거의 사용될 일은 없겠죠 (학습에는 파이토치 텐서가 들어가야 하니). 가시화용으로 한번씩 등장하기는 하지만, 주의깊게 살펴볼 내용은 아닙니다.
 
 오히려 뒤쪽이 매우 빈번하게 사용되는 것들입니다. `Resize`는 이미지 크기를 변경하라는 의미이고, `ToTensor`는 파이토치의 텐서형으로 변경하라는 의미, `Normalize`는 이미지의 normalize를 수행하라는 의미입니다.
 
 이미지 normalize는 CNN에서 정말 일반적으로 사용되는 방법이기에 반드시 기억해 줍시다! 이번 예제에서는 3채널 RGB이미지를 사용하였기에, 3개의 평균값, 3개의 표준편차값을 매개변수로 전달합니다.
 
 MNIST같은 gray-scale 이미지에는 각각 1개의 값만 전달합니다. 이후의 예제에서 등장할 파이토치의 pretrained 네트워크의 전이학습에도, 올바른 이미지를 입력으로 주기 위해서 필수적으로 사용됩니다.
  

