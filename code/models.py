import torch
import torch.nn as nn
from torchvision import models

class AlexNetXray(nn.Module):

    def __init__(self, hidden_size_1=4096, hidden_size_2=512, dropout=0.3):
        # Take the features and avgpool layers from AlexNet and write our own classifeir layers
        super(AlexNetXray, self).__init__()
        alexTemp = models.alexnet(pretrained=True)
        self.features = alexTemp.features
        self.avgpool = alexTemp.avgpool
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(9216, hidden_size_1),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_size_1, hidden_size_2),
            nn.ReLU(),
            nn.Linear(hidden_size_2, 2)
        )

    def forward(self, inputs):
        x = self.features(inputs)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        out = self.classifier(x)
        return out
        

class ResNet50Xray(nn.Module):

    def __init__(self, hidden_size=256, dropout=0.3):
        # Take the layers prior to the last classifier layer from ResNet-50 and write our own classifeir layers
        super(ResNet50Xray, self).__init__()
        resnetTemp = models.resnet50(pretrained=True)
        self.conv1 = resnetTemp.conv1
        self.bn1 = resnetTemp.bn1
        self.relu = resnetTemp.relu
        self.maxpool = resnetTemp.maxpool
        self.layer1 = resnetTemp.layer1
        self.layer2 = resnetTemp.layer2
        self.layer3 = resnetTemp.layer3
        self.layer4 = resnetTemp.layer4
        self.avgpool = resnetTemp.avgpool
        self.fc = nn.Sequential(
            nn.Linear(2048, hidden_size),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_size, 2)
        )    
    
    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        out = self.fc(x)
        
        return out


class InceptionNetXray(nn.Module):
    
    def __init__(self):
        super(InceptionNetXray, self).__init__()
        self.model = models.inception_v3(pretrained=True)
        # Modify the auxiliary classification layer
        num_features_fc_aux = self.model.AuxLogits.fc.in_features
        self.model.AuxLogits.fc = nn.Linear(num_features_fc_aux, 2)
        # Modify the primary classification layer
        num_features_fc = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features_fc, 2)
        
    def forward(self, inputs):
        return self.model(inputs)
        