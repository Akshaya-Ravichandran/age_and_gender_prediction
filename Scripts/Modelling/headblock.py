import torch.nn as nn
from .resnet import resnet18


class HeadBlock(nn.Module):
    def __init__(self, out_channels, in_channels=1000):
        super(HeadBlock, self).__init__()

        self.fc1 = nn.Linear(in_channels, 500)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(500, out_channels)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)

        return out
