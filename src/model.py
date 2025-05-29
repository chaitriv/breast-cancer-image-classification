import timm
import torch.nn as nn
import torchvision.models as models

from .config import DEVICE

def get_efficientnet():
    model = timm.create_model('efficientnet_b0', pretrained=True)
    model.classifier = nn.Linear(model.classifier.in_features, 1)
    return model.to(DEVICE)

def get_resnet():
    model = models.resnet50(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 1)
    return model.to(DEVICE)

def get_vit():
    model = timm.create_model('vit_base_patch16_224', pretrained=True)
    model.head = nn.Linear(model.head.in_features, 1)
    return model.to(DEVICE)
