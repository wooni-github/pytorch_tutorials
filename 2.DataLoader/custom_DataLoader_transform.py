import torch
import torchvision
import torchvision.transforms as tr # Data를 불러오면서 바로 전처리를 하게 할 수 있도록 해줌
from torch.utils.data import DataLoader, Dataset # DataLoader : BatchSize형태로 만들어줌. Dataset : 튜닝에 사용
import numpy as np

print('### 자신만의 Dataloder + Transform을 이용하는 예제 #1\n')

train_images = np.random.randint(256, size=(20, 32, 32, 3))
train_labels = np.random.randint(2, size=(20, 1))

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

class ToTensorMine:
    def __call__(self, sample):
        # Pytorch tensor형태로 데이터셋 변환
        inputs, labels = sample
        inputs = torch.FloatTensor(inputs)
        inputs = inputs.permute(2, 0, 1) # [ W H C ] => [ C, W ,H ]
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
train_loader1 = DataLoader(ds1, batch_size=10, shuffle=True)
dataiter = iter(train_loader1)
images, labels = dataiter.next()

print('images.size() : ', images.size()) # [10 x 3 x 32 x 32]
# print(images[0])