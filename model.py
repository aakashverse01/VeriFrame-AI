
import torch
import torch.nn as nn
from torchvision import models


def create_model():
    # ResNet18 architecture
    model = models.resnet18(weights=None)

    # Replace output layer for two classes
    # Class 0 = Real, Class 1 = Fake
    model.fc = nn.Linear(
        model.fc.in_features,
        2
    )

    return model


if __name__ == "__main__":
    model = create_model()

    print("VeriFrame AI - Model Setup")
    print("Architecture: ResNet18")
    print("Output Classes: 2 (Real / Fake)")
    print("Device: CPU")
    print("Model initialized successfully!")
