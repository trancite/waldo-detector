import torch
from torch import nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from torchvision import transforms
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(BASE_DIR, "model", "waldo_detector_64x64_v1.0.pth")

def load_model(path: str = model_path, device: str = "cpu") -> nn.Module:

    weights = MobileNet_V2_Weights.DEFAULT
    model = mobilenet_v2(weights=weights)
    model.classifier[1] = nn.Linear(1280, 1)  

    if model_path:
        model.load_state_dict(torch.load(path, map_location=device))

    model.to(device)
    model.eval()
    return model


def get_preprocessor():
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])
