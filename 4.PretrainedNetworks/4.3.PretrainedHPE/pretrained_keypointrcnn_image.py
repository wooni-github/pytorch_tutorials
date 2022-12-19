import torch
import torchvision
import cv2
import argparse
import utils
from PIL import Image
from torchvision.transforms import transforms as transforms

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default = 'test_image.jpg')
    args = parser.parse_args()


    USE_CUDA = torch.cuda.is_available()
    DEVICE = torch.device("cuda" if USE_CUDA else "cpu")

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    model = torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained=True, num_keypoints=17)
    model.to(DEVICE)
    model.eval()

    image = Image.open(args.input)

    with torch.no_grad():
        outputs = model(transform(image).unsqueeze(0).to(DEVICE))

    output_image = utils.draw_keypoints(outputs, image, PIL_image = True)


    cv2.imshow('Output image', output_image)
    cv2.waitKey()
