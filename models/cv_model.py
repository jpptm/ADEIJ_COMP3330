import torch
import torchvision
from torchvision.models import resnet18, ResNet18_Weights

# testing out some models
# basing them off the:
# "Classification of Alzheimer's Disease using Convolutional Neural Networks and Transfer Learning" by S. K. Sahu et al.
# This paper uses a frozen backbones from some famous models to classify 6 different stages of Alzheimer's disease.
# Will generalise this shortly

# class CVModel(torch.nn.Module):
#     def __init__(self, num_classes):
#         super(CVModel, self).__init__()

#         resnet18 = torchvision.models.resnet18(pretrained=True)

#         # Freeze resnet layers
#         for param in resnet18.parameters():
#             param.requires_grad = False

# 		# Replace last fully connected layer with global average pooling layer
#         # Apparently allows input images of different sizes, do we want this?
#         # I was just copying the study
#         resnet18.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))

#         # Replace last fully connected layer with a new fully connected layer with 256 units and ReLU activation
#         resnet18.fc = torch.nn.Sequential(
#             torch.nn.Linear(resnet18.fc.in_features, 256),
#             torch.nn.ReLU(),
#             torch.nn.Linear(256, num_classes)
#         )

#         self.model = resnet18

#     def forward(self, x):
#         return self.model(x)


# Model used to test the average pooling
class CVModel(torch.nn.Module):
    def __init__(self, num_classes):
        super(CVModel, self).__init__()

        # the warning says to use 'weights' instead of pretrained. This will assure we have most recent weights
        resnet18 = torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT)

        # Freeze resnet layers
        for param in resnet18.parameters():
            param.requires_grad = False

        # resnet18.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
        resnet18.fc = torch.nn.Linear(resnet18.fc.in_features, num_classes)

        self.model = resnet18

    def forward(self, x):
        return self.model(x)

# class CVModel(torch.nn.Module):
#     def __init__(self, num_classes):
#         super(CVModel, self).__init__()

#         resnet50 = torchvision.models.resnet50(weights=ResNet50_Weights.DEFAULT)

#         # Freeze resnet layers
#         for param in resnet50.parameters():
#             param.requires_grad = False

# 		# Replace last fully connected layer with global average pooling layer
#         # Apparently allows input images of different sizes, do we want this?
#         # I was just copying the study
#         resnet50.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))

#         # Replace last fully connected layer with a new fully connected layer with 256 units and ReLU activation
#         resnet50.fc = torch.nn.Sequential(
#             torch.nn.Linear(resnet50.fc.in_features, 256),
#             torch.nn.ReLU(),
#             torch.nn.Linear(256, num_classes)
#         )

#         self.model = resnet50

#     def forward(self, x):
#         return self.model(x)

# class CVModel(torch.nn.Module):
#     def __init__(self, num_classes):
#         super(CVModel, self).__init__()

#         vgg16 = torchvision.models.vgg16(pretrained=True)

#         # Freeze resnet layers
#         for param in vgg16.parameters():
#             param.requires_grad = False

# 		# Replace last fully connected layer with global average pooling layer
#         vgg16.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))

#         # First stab in the dark:
#         # "Classification of Alzheimer's Disease using Convolutional Neural Networks and Transfer Learning" by S. K. Sahu et al.
#         # This paper uses a frozen vgg16 backbone to classify 6 different stages of Alzheimer's disease.

#         # Replace last fully connected layer with a new fully connected layer with 256 units and ReLU activation
#         vgg16.fc = torch.nn.Sequential(
#             torch.nn.Linear(vgg16.fc.in_features, 256),
#             torch.nn.ReLU(),
#             torch.nn.Linear(256, num_classes)
#         )

#         self.model = vgg16

#     def forward(self, x):
#         return self.model(x)