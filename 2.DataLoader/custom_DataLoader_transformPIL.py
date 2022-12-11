import torch
import torchvision
import torchvision.transforms as tr # Data를 불러오면서 바로 전처리를 하게 할 수 있도록 해줌
from torch.utils.data import DataLoader, Dataset # DataLoader : BatchSize형태로 만들어줌. Dataset : 튜닝에 사용
import numpy as np

print('### 자신만의 Dataloder + Transform을 이용하는 예제 #2 (PIL 이미지) \n')

train_images = np.random.randint(256, size=(20, 32, 32, 3))
train_labels = np.random.randint(2, size=(20, 1))

# print(train_images.shape, train_labels.shape) # [batch x width x height x channel], [batch x label]

class MyDataset(Dataset): # from torch.utils.data import DataLoader, Dataset 의 Dataset을 상속받음
    def __init__(self, x_data, y_data, transform=None):
        self.x_data = x_data
        self.y_data = y_data
        self.transform = transform
        self.len = len(y_data)

    def __getitem__(self, index):
        sample = self.x_data[index], self.y_data[index]

        if self.transform:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        return self.len

class MyTransform:
    def __call__(self, sample):
        inputs, labels = sample
        inputs = torch.FloatTensor(inputs)
        inputs = inputs.permute(2, 0, 1)
        # 이렇게 하면 tr.ToPILImages등의 transform관련 함수들 사용 가능
        # tr.Resize(128)은 tr.Resize((128, 128))와 동일

        transf = tr.Compose([tr.ToPILImage(), tr.Resize(128), tr.ToTensor(), tr.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        final_output = transf(inputs)

        labels = torch.FloatTensor(labels)
        return final_output, labels

ds2 = MyDataset(train_images, train_labels, transform = MyTransform())
train_loader2 = DataLoader(ds2, batch_size = 10, shuffle=True)

dataiter2 = iter(train_loader2)
images2, labels2 = dataiter2.next()
print('images2.size() : ', images2.size()) # [10 x 3 x 128 x 128]
# print(images2)

