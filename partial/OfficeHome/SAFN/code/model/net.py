import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import models
from torch.autograd import Variable
import torch.nn.functional as F
import math


class ResBase50(nn.Module):
    def __init__(self):
        super(ResBase50, self).__init__()
        model_resnet50 = models.resnet50(pretrained=True)
        self.conv1 = model_resnet50.conv1
        self.bn1 = model_resnet50.bn1
        self.relu = model_resnet50.relu
        self.maxpool = model_resnet50.maxpool
        self.layer1 = model_resnet50.layer1
        self.layer2 = model_resnet50.layer2
        self.layer3 = model_resnet50.layer3
        self.layer4 = model_resnet50.layer4
        self.avgpool = model_resnet50.avgpool

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        return x


class ResClassifier(nn.Module):
    def __init__(self, class_num=65, extract=True, dropout_p=0.5):
        super(ResClassifier, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(2048, 1000),
            nn.BatchNorm1d(1000, affine=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p)
            )
        self.fc2 = nn.Linear(1000, class_num)
        self.extract = extract
        self.dropout_p = dropout_p

    def forward(self, x):
        fc1_emb = self.fc1(x)
        if self.training:
            fc1_emb.mul_(math.sqrt(1 - self.dropout_p))            
        logit = self.fc2(fc1_emb)

        if self.extract:
            return fc1_emb, logit         
        return logit
