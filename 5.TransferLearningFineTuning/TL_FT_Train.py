import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
import argparse
from TL_FT_Network import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_folder', default = 'hymenoptera_data/train/')
    parser.add_argument('--batch_size', type = int, default=64)
    parser.add_argument('--epoch', type=int, default=50)
    args = parser.parse_args()

    USE_CUDA = torch.cuda.is_available()
    DEVICE = torch.device("cuda" if USE_CUDA else "cpu")

    model = model().to(DEVICE)

    EPOCHS = args.epoch
    BATCH_SIZE = args.batch_size

    train_loader = torch.utils.data.DataLoader(datasets.ImageFolder(args.input_folder, transform_train)
                                               , batch_size=BATCH_SIZE, shuffle=True)

    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

    criterion = nn.functional.cross_entropy

    iteration = 0
    min_loss = 10e6
    max_iteration = len(train_loader)*EPOCHS
    for epoch in range(EPOCHS):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(DEVICE), target.to(DEVICE) # Data -> Device
            output = model(data) # Input data -> Network(Input) -> Output 획득

            loss = criterion(output, target) # Loss 계산

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if iteration % 20 == 0:
                print(('Iteration [{}/{} = {}%], loss {}').format(iteration, max_iteration, round(iteration/max_iteration*100.0, 3), round(loss.item(), 3)))

            if iteration/max_iteration > 0.8 and min_loss > loss.item():
                # 학습이 80% 이상 진행된 이후 minimum loss를 갖는 weights를 저장함
                min_loss = loss.item()
                torch.save(model.state_dict(), 'LAST.pth')
            iteration += 1
    print('Save training results as :', 'LAST.pth')