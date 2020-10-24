"""LeNet in PyTorch."""
from torch.nn import Conv2d, Linear, Module, functional


class LeNet(Module):

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = Conv2d(3, 6, 5)
        self.conv2 = Conv2d(6, 16, 5)
        self.fc1 = Linear(16*5*5, 120)
        self.fc2 = Linear(120, 84)
        self.fc3 = Linear(84, 10)

    def forward(self, x):  # pylint: disable=arguments-differ  # pytorch uses a bad name "inputs"
        out = functional.relu(self.conv1(x))
        out = functional.max_pool2d(out, 2)
        out = functional.relu(self.conv2(out))
        out = functional.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = functional.relu(self.fc1(out))
        out = functional.relu(self.fc2(out))
        out = self.fc3(out)
        return out
