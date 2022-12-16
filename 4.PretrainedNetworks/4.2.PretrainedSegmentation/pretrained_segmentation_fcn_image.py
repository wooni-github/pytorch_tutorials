import torchvision
import torch
import argparse
import utils
import cv2
from PIL import Image

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default = 'test_image.jpg')
    args = parser.parse_args()

    USE_CUDA = torch.cuda.is_available()
    DEVICE = torch.device("cuda" if USE_CUDA else "cpu")

    model = torchvision.models.segmentation.fcn_resnet50(pretrained=True)
    model.to(DEVICE)
    model.eval()

    image = Image.open(args.input)

    outputs = utils.get_segment_labels(image, model, DEVICE)
    outputs = outputs['out'] # output은 클래스에 속하는 확률이 담긴 'out'과 auxilliary loss가 담긴 'aux'로 구성되어 있음.

    overlay, segmented = utils.visualize(image, outputs)
    cv2.imshow('Overlay image', overlay)
    cv2.imshow('Segmented image', segmented)
    cv2.waitKey()
