import torch
import torch.optim as optim
import torch.nn as nn
from torchvision import transforms, datasets
from torchsummary import summary

from MNIST_Network import *
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type = int, default=64)
    parser.add_argument('--epoch', type=int, default=20)
    args = parser.parse_args()

    USE_CUDA = torch.cuda.is_available()
    DEVICE = torch.device("cuda" if USE_CUDA else "cpu")

    model = FCL_Regression().to(DEVICE)

    summary(model, input_size=(1, 28, 28))

    EPOCHS = args.epoch
    BATCH_SIZE = args.batch_size

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data',
                       train=True,
                       download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=BATCH_SIZE, shuffle=True) # Training 단계에서는 shuffle 수행

    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5) # optimizer 설정

    criterion = nn.functional.l1_loss

    print('Save initial weights as : FIRST.pth')
    torch.save(model.state_dict(), 'FIRST.pth')  # 전체 모델 저장

    max_iteration = len(train_loader)*EPOCHS
    print('Batch ', BATCH_SIZE)
    print('Total epoch', EPOCHS)
    print('Total iterations', max_iteration)
    print()

    iteration = 0
    min_loss = 10e6

    for epoch in range(EPOCHS):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(DEVICE), target.to(DEVICE) # Data -> Device
            output = model(data) # Input data -> Network(Input) -> Output 획득
            # Classification 문제일 때는 MNIST 0~9 10개의 class에 속하는 확률로 [0.1 0.02 .... 0.8,, .01] 의 [1x10]크기의 벡터를 반환함

            target = target.float()
            output = torch.squeeze(output)

            loss = criterion(output, target) # Loss 계산

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if iteration % 2000 == 0:
                print(('Iteration [{}/{} = {}%], loss {}').format(iteration, max_iteration, round(iteration/max_iteration*100.0, 3), round(loss.item(), 3)))
                print(target)
                print(output)
                print(loss)
                print()

            if iteration == max_iteration//4:
                # 검증용 중간 weights 저장 (Epoch 25%지점)
                torch.save(model.state_dict(), 'MIDDLE.pth')

            if iteration/max_iteration > 0.8 and min_loss > loss.item():
                # 학습이 80% 이상 진행된 이후 minimum loss를 갖는 weights를 저장함
                min_loss = loss.item()
                torch.save(model.state_dict(), 'LAST.pth')
            iteration += 1
    print('Save training results as :', 'LAST.pth')