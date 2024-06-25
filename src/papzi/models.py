import torch.nn as nn
from torchvision.models import (
    resnet18,
    ResNet18_Weights,
    resnet50,
    ResNet50_Weights,
)

ResNet50_Weights.IMAGENET1K_V2


# Define the ResNet model by inheriting from nn.Module
class ResNet18(nn.Module):
    def __init__(self, num_classes):
        super(ResNet18, self).__init__()

        self.model = resnet18(weights=ResNet18_Weights.DEFAULT)

        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)


# Define a function to flatten the output


# Define the ResNet model by inheriting from nn.Module
class ResNet50(nn.Module):
    def __init__(self, num_classes):
        super(ResNet50, self).__init__()
        # self.resnet = resnet18(weights=ResNet18_Weights.DEFAULT)

        self.model = resnet50(weights=ResNet50_Weights.DEFAULT)
        #
        self.model.fc = nn.Sequential(  # type: ignore
            nn.Linear(self.model.fc.in_features, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes),
            nn.Softmax(
                dim=1
            ),  # Apply softmax activation to get class probabilities
        )

        # nn.Sequential(
        #     nn.Linear(self.model.fc.in_features, 256),
        #     nn.ReLU(),
        #     nn.Dropout(0.5),
        #     nn.Linear(256, num_classes),

        # )

    def forward(self, x):
        return self.model(x)
