"""LeNet in PyTorch."""
import torch.nn as nn
from torch.nn import functional


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        out = functional.relu(self.conv1(x))
        out = functional.max_pool2d(out, 2)
        out = functional.relu(self.conv2(out))
        out = functional.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = functional.relu(self.fc1(out))
        out = functional.relu(self.fc2(out))
        out = self.fc3(out)
        return out
