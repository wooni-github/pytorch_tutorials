import torchvision.transforms as transforms
import cv2
import numpy as np
import torch

label_map = [
               (0, 0, 0),  # background
               (128, 0, 0), # aeroplane
               (0, 128, 0), # bicycle
               (128, 128, 0), # bird
               (0, 0, 128), # boat
               (128, 0, 128), # bottle
               (0, 128, 128), # bus
               (128, 128, 128), # car
               (64, 0, 0), # cat
               (192, 0, 0), # chair
               (64, 128, 0), # cow
               (192, 128, 0), # dining table
               (64, 0, 128), # dog
               (192, 0, 128), # horse
               (64, 128, 128), # motorbike
               (192, 128, 128), # person
               (0, 64, 0), # potted plant
               (128, 64, 0), # sheep
               (0, 192, 0), # sofa
               (128, 192, 0), # train
               (0, 64, 128) # tv/monitor
]

def get_segment_labels(image, model, DEVICE):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    image = transform(image).to(DEVICE).unsqueeze(0)
    outputs = model(image)
    return outputs

def visualize(input, outputs):
    labels = torch.argmax(outputs.squeeze(), dim=0).detach().cpu().numpy()
    segmented_image = np.zeros([len(labels), len(labels[0]), 3]).astype(np.uint8)

    for label_num in range(0, len(label_map)):
        index = labels == label_num
        segmented_image[index] = np.array(label_map)[label_num]

    image = np.array(input)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    segmented_image = cv2.cvtColor(segmented_image, cv2.COLOR_RGB2BGR)
    cv2.addWeighted(segmented_image, 0.5, image, 0.5, 0, image)
    return image, segmented_image
