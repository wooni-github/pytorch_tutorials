import torchvision
import torchvision.transforms as tr # Data를 불러오면서 바로 전처리를 하게 할 수 있도록 해줌
from torch.utils.data import DataLoader, Dataset # DataLoader : BatchSize형태로 만들어줌. Dataset : 튜닝에 사용

print('### ImageFolder example : 이미지를 클래스별 폴더에 넣었을 때 자동으로 labeling을 수행하여 로드\n')
# example_dataset/ants/ : 5개의 이미지 -> 자동으로 클래스 0으로 설정
# example_dataset/bees/ : 8개의 이미지 -> 클래스 1로 설정

# 로 구분해서 폴더에 넣어두었을 경우에 자동으로 폴더별 labeling
transf = tr.Compose([tr.CenterCrop(100), tr.Resize(16), tr.ToTensor()])
# 이미지별 사이즈가 상이하기 때문에 Crop이나 Resize없이 batch형태로 만들면 에러 발생

trainset = torchvision.datasets.ImageFolder(root='./example_dataset', transform=transf)
trainloader = DataLoader(trainset, batch_size=2, shuffle=False)
print('len(trainset)', len(trainset)) # 5 + 9 = 14개의 이미지

dataiter = iter(trainloader)
images, labels = dataiter.next()
print('images.size() : [batch x channel x height x width]', images.size())
