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
            for name, param in model.named_parameters():
                if 'blocks.10' in name:  # The second last transformer block
                    param.requires_grad = True
                else:
                    param.requires_grad = False
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

