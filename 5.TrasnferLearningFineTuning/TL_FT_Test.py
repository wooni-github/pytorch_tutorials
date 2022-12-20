import torch
from torchvision import datasets
import matplotlib.pyplot as plt
import argparse
import numpy as np

from TL_FT_Network import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_folder', default = '../hymenoptera_data/test/')
    parser.add_argument('--fine_tuning', choices = [0, 1], default = 1)
    args = parser.parse_args()

    USE_CUDA = torch.cuda.is_available()
    DEVICE = torch.device("cuda" if USE_CUDA else "cpu")

    model = model(fine_tuning=args.fine_tuning)

    image_datasets = datasets.ImageFolder(args.input_folder, transform_valid)

    test_loader = torch.utils.data.DataLoader(image_datasets, batch_size=1, shuffle=True)
    # 추론시 보통 shuffle 수행하지 않지만, 가시화 결과를 클래스 여러 개(2개)에서 보기 위해 shuffle 수행

    classes = image_datasets.classes

    if args.fine_tuning:
        model.load_state_dict(torch.load('LAST.pth'))
    else:
        with open('imagenet_classes.txt') as f:
            classes_imagenet = [line.strip() for line in f.readlines()]
        classes_imagenet = classes_imagenet[4:]  # .txt파일의 최초 4줄은 의미 없음

    model = model.to(DEVICE)
    model.eval()

    correct = len(test_loader)

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):

            data, target = data.to(DEVICE), target.to(DEVICE)  # Data -> Device
            output = model(data)  # Input data -> Network(Input) -> Output 획득

            pred = (torch.max(output, 1)[1]).item()

            if args.fine_tuning :
                if classes[target] != classes[pred]:
                    correct -= 1
            else:
                # classes는 폴더명이(클래스명) 'ants', 'bees' 이지만,
                # ImageNet dataset은 클래스명이 'ant', 'bee' 임

                if classes[target][:-1] != classes_imagenet[pred].split()[1]:
                    correct -= 1

        print('Accuracy', round(correct / len(test_loader) * 100.0, 3), '%')



    def imshow(inp):
        inp = inp.numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        inp = std * inp + mean
        inp = np.clip(inp, 0, 1)
        return inp


    # 이미지 5장에 대해서만 plt 디스플레이
    fig = plt.figure(figsize = (10, 10))
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = model(data)

            pred = (torch.max(output, 1)[1]).item()

            if args.fine_tuning :
                target = classes[target]
                pred = classes[pred]
            else:
                target = classes[target][:-1]
                pred = classes_imagenet[pred].split()[1]

            subplot = fig.add_subplot(5, 4, batch_idx//5*5 + batch_idx%5 + 1)
            subplot.imshow(imshow(data.cpu()[0]), cmap=plt.cm.gray_r)
            plt.title('GT : ' + target + '\n Pred : ' + pred)
            plt.axis('off')

            if batch_idx == 19:
                break

    plt.subplots_adjust(wspace = 0.8, hspace = 0.8)
    plt.show()
