import torch
import torch.nn as nn
from .resnet import resnet18, resnet50, wide_resnet50_2, wide_resnet101_2
from .headblock import HeadBlock


class MultiTaskNet(nn.Module):
    def __init__(self, freeze_mid_layers=False, pretrained=True):
        super(MultiTaskNet, self).__init__()
        # self.resnet = resnet18(pretrained=pretrained, freeze_mid_layers=freeze_mid_layers)
        self.resnet = wide_resnet50_2(pretrained=pretrained)
        self.sig = nn.Sigmoid()

        # Ordinal Classification
        self.age_head = HeadBlock(8)
        self.softmax = nn.Softmax(dim=1)
        # Binary Classification Task so output of dimension 2
        self.gender_head = HeadBlock(1)

    def forward(self, x):
        out = self.resnet(x)

        age = self.age_head(out)
        gender = self.gender_head(out)

        # Apply sigmoid transformations to the outputs
        gender = self.sig(gender).reshape(-1)

        return age, gender

    def transform_scores_to_predictions(self, age, gender):

        with torch.no_grad():
            age = self.softmax(age)
            age_pred = torch.argmax(age, dim=1)

            gender_pred = gender > 0.5

        return age_pred, gender_pred

    def predict(self, x):

        age, gender = self.forward(x)
        age_pred, gender_pred = self.transform_scores_to_predictions(
            age, gender
        )

        return age_pred, gender_pred
