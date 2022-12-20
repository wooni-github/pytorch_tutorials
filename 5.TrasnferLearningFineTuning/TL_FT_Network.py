from torchvision import models
from torchvision import transforms
import torch.nn as nn

transform_train = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_valid = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def model(fine_tuning = True):
    model = models.resnet50(pretrained=True)

    print('Original resnet50 : trained for ImageNet dataset(1000 classes) => last fc : [in 2048 - out 1000]')
    print('Original model fc (Transfer Learning)', model.fc)
    print()

    if not fine_tuning:
        return model

    model.fc = nn.Linear(model.fc.in_features, 2)
    print('Changed resnet50 : train for hymenoptera dataset (2 classes) => last fc : [in 2048 - out 2]')
    print('Changed model fc (Fine-Tuning) ', model.fc)

    return model


