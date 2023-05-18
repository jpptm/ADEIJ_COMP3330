import torch
import torchvision
from torchvision.models import resnet18, ResNet18_Weights, resnet50, ResNet50_Weights, vgg16, VGG16_Weights
from efficientnet_pytorch import EfficientNet
import timm

class CVModel(torch.nn.Module):
    def __init__(self, num_classes, kind, hidden_size, name):
        super(CVModel, self).__init__()
        self.name = name

        if kind == 'resnet50':
            model = torchvision.models.resnet50(weights = ResNet50_Weights.DEFAULT)
        elif kind == 'resnet18':
            model = torchvision.models.resnet18(weights = ResNet18_Weights.DEFAULT)
        elif kind == 'efficientnet':
            model = EfficientNet.from_pretrained('efficientnet-b2')
        elif kind == 'vgg':
            model = torchvision.models.vgg16(weights=VGG16_Weights.DEFAULT)
        elif kind == 'vit':
            model = timm.create_model('vit_base_patch16_224', pretrained=True)
            model.head = torch.nn.Linear(model.head.in_features, hidden_size)
        else:
            raise ValueError(f"Unsupported model kind: {kind}")

        # Freeze model layers
        for param in model.parameters():
            param.requires_grad = False

        if kind == 'efficientnet':
            # Replace the last fully connected layer of EfficientNet with a new fully connected layer
            in_features = model._fc.in_features
            model._fc = torch.nn.Linear(in_features, hidden_size)
        elif kind == 'vgg':
            # Replace the last fully connected layer of vGG with a new fully connected layer
            in_features = model.classifier[6].in_features
            model.classifier[6] = torch.nn.Linear(in_features, hidden_size)
        elif kind == 'vit':
            # Replace the last fully connected layer of ViT with a new fully connected layer
            model.head = torch.nn.Sequential(
                torch.nn.Linear(model.head.in_features, hidden_size),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_size, num_classes)
            )
        else:
            # Replace the last fully connected layer of ResNet with a new fully connected layer
            model.fc = torch.nn.Sequential(
                torch.nn.Linear(model.fc.in_features, hidden_size),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_size, num_classes)
            )

        self.model = model

    def forward(self, x):
        return self.model(x)


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
# class CVModel(torch.nn.Module):
#     def __init__(self, num_classes):
#         super(CVModel, self).__init__()
# class CVModel(torch.nn.Module):
#     def __init__(self, num_classes):
#         super(CVModel, self).__init__()

#         # the warning says to use 'weights' instead of pretrained. This will assure we have most recent weights
#         resnet18 = torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT)

#         # Freeze resnet layers
#         for param in resnet18.parameters():
#             param.requires_grad = False

#         # resnet18.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
#         resnet18.fc = torch.nn.Linear(resnet18.fc.in_features, num_classes)

#         self.model = resnet18

#     def forward(self, x):
#         return self.model(x)

# class CVModel(torch.nn.Module):
#     def __init__(self, num_classes):
#         super(CVModel, self).__init__()

#         # the warning says to use 'weights' instead of pretrained. This will assure we have most recent weights
#         resnet18 = torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT)

#         # Freeze resnet layers
#         for param in resnet18.parameters():
#             param.requires_grad = False

#         # resnet18.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
#         resnet18.fc = torch.nn.Linear(resnet18.fc.in_features, num_classes)

#         self.model = resnet18

#     def forward(self, x):
#         return self.model(x)

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

#         vgg16 = torchvision.models.vgg16(weights=VGG16_Weights.DEFAULT)

#         # Freeze resnet layers
#         for param in vgg16.parameters():
#             param.requires_grad = False

# 		# Replace last fully connected layer with global average pooling layer
#         vgg16.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))

#         # First stab in the dark:
#         # "Classification of Alzheimer's Disease using Convolutional Neural Networks and Transfer Learning" by S. K. Sahu et al.
#         # This paper uses a frozen vgg16 backbone to classify 6 different stages of Alzheimer's disease.
#         in_features = vgg16.classifier[6].in_features

#         # Replace last fully connected layer with a new fully connected layer with 256 units and ReLU activation
#         vgg16.classifier = torch.nn.Sequential(
#             *list(vgg16.classifier.children())[:-1],  # Keep all layers except the last one
#             torch.nn.Linear(in_features, 256),
#             torch.nn.ReLU(),
#             torch.nn.Linear(256, num_classes)
#         )

#         self.model = vgg16

#     def forward(self, x):
#         return self.model(x)