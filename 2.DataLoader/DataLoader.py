import torchvision
import torchvision.transforms as tr # Data를 불러오면서 바로 전처리를 하게 할 수 있도록 해줌
from torch.utils.data import DataLoader, Dataset # DataLoader : BatchSize형태로 만들어줌. Dataset : 튜닝에 사용

print('### Dataloader 활용법 (Torch 기본 제공 데이터 CIFAR10 데이터)\n')

transf = tr.Compose([tr.Resize((8, 8)), tr.ToTensor()]) # 전처리. 순서대로 작업함 (8,8) resize -> tensor로 전환
# transf = tr.Compose([tr.Resize(8), tr.ToTensor()]) # 이렇게 해도 됨

# CIFAR10 dataset
# Train : [50000 images with 32 x 32 x 3]
# Test : [10000 images with 32 x 32 x 3]

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transf)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transf)

print(trainset[0][0].size()) # [ 3 x 8 x 8 ] = [channel(CIFAR : RGB) x height(transformed) x width(transformed)]

trainloader = DataLoader(trainset, batch_size=50, shuffle=True) # train 에서는 보통 shuffle True
testloader = DataLoader(testset, batch_size=50, shuffle=False) # test 에서는 false

print('len(trainloader)', len(trainloader)) # 1000 => 전체 50000 training images => 50 batches ==> 1000개의 batch별 이미지

dataiter = iter(trainloader)
images, labels = dataiter.next()
print('images.size() : [batch x channel x height x width]', images.size()) # [ 50 x 3 x 8 x 8 ]

for i in range(100): # for max iteration
    images, labels = dataiter.next()
    print('iteration', i, images.shape)
