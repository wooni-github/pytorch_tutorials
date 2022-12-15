
from torchvision import models
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--network', choices=['resnet50', 'vgg11', 'googlenet'], default='resnet50')
    args = parser.parse_args()

    imgs = [Image.open(i) for i in ['dog.jpg', 'goose.jpg', 'koala.jpg']]

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    if args.network == 'resnet50':
        model = models.resnet50(pretrained=True)
    elif args.network == 'vgg11':
        model = models.vgg11(pretrained=True)
    elif args.network == 'googlenet':
        model = models.googlenet(pretrained=True)
    model.eval()


    with open('imagenet_classes.txt') as f:
        classes = [line.strip() for line in f.readlines()]
    classes = classes[4:] # .txt파일의 최초 4줄은 의미 없음

    fig = plt.figure()

    for i, img in enumerate(imgs):
        out = model(torch.unsqueeze(transform(img), 0))
        _, index = torch.max(out, 1) #가장 확률이 높은 것 뽑아냄
        subplot = fig.add_subplot(1, 3, i+1)
        subplot.imshow(img,cmap=plt.cm.gray_r)
        plt.title(classes[index])
        plt.axis('off')
    plt.show()

