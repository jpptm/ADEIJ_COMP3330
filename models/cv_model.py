import torch
import torchvision


class CVModel(torch.nn.Module):
    def __init__(self, num_classes):
        super(CVModel, self).__init__()

        resnet18 = torchvision.models.resnet18(pretrained=True)

        # Freeze resnet layers
        for param in resnet18.parameters():
            param.requires_grad = False

		# Replace last fully connected layer with global average pooling layer
        resnet18.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))

        # First stab in the dark:
        # "Classification of Alzheimer's Disease using Convolutional Neural Networks and Transfer Learning" by S. K. Sahu et al.
        # This paper uses a frozen ResNet-18 backbone to classify 6 different stages of Alzheimer's disease.

        # Replace last fully connected layer with a new fully connected layer with 256 units and ReLU activation
        resnet18.fc = torch.nn.Sequential(
            torch.nn.Linear(resnet18.fc.in_features, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, num_classes)
        )

        self.model = resnet18

    def forward(self, x):
        x = x.type(torch.uint8)
        return self.model(x)