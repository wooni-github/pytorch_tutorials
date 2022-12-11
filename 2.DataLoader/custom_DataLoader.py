# import torchvision
# import torchvision.transforms as tr # Data를 불러오면서 바로 전처리를 하게 할 수 있도록 해줌

import torch
from torch.utils.data import DataLoader, Dataset # DataLoader : BatchSize형태로 만들어줌. Dataset : 튜닝에 사용
import numpy as np

print('### 자신만의 Dataloder를 이용하는 예제\n')
train_images = np.random.randint(256, size=(20, 32, 32, 3))
# Width 32 x Height 32 x Channel 3 의 이미지 (0~255의 값을 갖음) N 20장
train_labels = np.random.randint(2, size=(20, 1)) # 20장의 이미지에 0 or 1의 class label 부여

print('train_images.shape [Batch x Height x Width x Channel] : ', train_images.shape) # [batch x height x width x channel]
print('train_labels.shape [Batch x Label] : ', train_labels.shape) # [batch x label]

class TensorData(Dataset): # from torch.utils.data import DataLoader, Dataset 의 Dataset을 상속받음
    def __init__(self, x_data, y_data):
        self.x_data = torch.FloatTensor(x_data)
        self.x_data = self.x_data.permute(0, 3, 1, 2) # [Batch x Channel x Height x Width]로 바꿔줌
        self.y_data = torch.LongTensor(y_data) # Long타입으로
        self.len = self.y_data.shape[0] # 데이터갯수

    def __getitem__(self, index):
        # getitem을 통해 batch데이터 생성 (이 예제에선 입력 이미지, 결과 클래스)
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        # len을 통해 전체 데이터 수 반환
        return self.len

    # getitem과 len은 정형화 되어 있다고 봐도 됨. [ transform을 활용하기에 편리함 ]

train_data = TensorData(train_images, train_labels)
train_loader = DataLoader(train_data, batch_size=10, shuffle=True)

print('train_data[0][0].size() : ', train_data[0][0].size()) # [3 x 32 x 32]
dataiter = iter(train_loader)
images, labels = dataiter.next()
print('images.size() : ', images.size()) # [10 x 3 x 32 x 32]