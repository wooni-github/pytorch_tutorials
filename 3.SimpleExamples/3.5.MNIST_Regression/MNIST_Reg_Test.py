import torch
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import argparse

from MNIST_Network import *

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--load', choices=['FIRST', 'MIDDLE', 'LAST'], default='LAST')
    args = parser.parse_args()

    USE_CUDA = torch.cuda.is_available()
    DEVICE = torch.device("cuda" if USE_CUDA else "cpu")
    pth = args.load + '.pth'

    model = FCL_Regression()

    model = model.to(DEVICE) # Model -> Device
    model.load_state_dict(torch.load(pth))
    model.eval() # eval을 설정해줘야 dropout, batch_normalization 등을 해제함

    BATCH_SIZE = 1 # 추론 단계에서는 1장식 추론을 수행할 것

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data',
                       train=False,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=BATCH_SIZE, shuffle=False) # 추론 단계에서는 shuffle 수행하지 않음


    acc = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(DEVICE), target.to(DEVICE) # Data -> Device
            output = model(data) # Input data -> Network(Input) -> Output 획득
            pred = output.item()
            target = int(target.item())

            acc += (pred - float(target))**2

    print(pth, ' : Accuracy', round((acc**0.5)/len(test_loader), 3))


    # 이미지 5장에 대해서만 plt 디스플레이
    fig = plt.figure()
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = model(data)

            target = int(target.item())

            pred = output.item()

            subplot = fig.add_subplot(1, 5, batch_idx + 1)
            subplot.imshow(data.cpu().reshape((28, 28)),
                           cmap=plt.cm.gray_r)
            plt.title('GT : ' + str(target) + '\n Pred : ' + str(round(pred, 2)))
            plt.axis('off')
            if batch_idx == 4:
                break
    plt.show()
