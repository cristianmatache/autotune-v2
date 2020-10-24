from torch.nn import AvgPool2d, AvgPool3d, Conv2d, Linear, Module, functional


class CudaConvNet(Module):

    def __init__(self, alpha: float = 0.00005, beta: float = 0.010001):
        super().__init__()

        self.conv1 = Conv2d(3, 32, 5, 1, 2)  # n_input, n_output, ks, stride, padding
        self.conv2 = Conv2d(32, 32, 5, 1, 2)
        self.conv3 = Conv2d(32, 64, 5, 1, 2)

        self.lrn1 = LRN(3, alpha, beta)  # local size = 3
        self.lrn2 = LRN(3, alpha, beta)

        self.fc1 = Linear(64 * 4 * 4, 10)

    def forward(self, x):  # pylint: disable=arguments-differ  # pytorch uses a bad name "inputs"
        out = self.conv1(x)
        out = functional.relu(functional.max_pool2d(out, 3, 2, 1), inplace=True)  # ks, stride, padding
        out = self.lrn1(out)

        out = functional.relu(self.conv2(out), inplace=True)
        out = functional.avg_pool2d(out, 3, 2, 1)
        out = self.lrn2(out)

        out = functional.relu(self.conv3(out), inplace=True)
        out = functional.avg_pool2d(out, 3, 2, 1)

        out = out.view(out.size(0), -1)

        out = self.fc1(out)
        return out


class LRN(Module):
    """Helper class for local response normalisation Is this efficient?"""

    def __init__(self, local_size: int = 1, alpha: float = 1.0, beta: float = 0.75, across_channels: bool = True):
        super().__init__()
        self.across_channels = across_channels
        if across_channels:
            self.average = AvgPool3d(kernel_size=(local_size, 1, 1), stride=1, padding=(int((local_size-1.0)/2), 0, 0))
        else:
            self.average = AvgPool2d(kernel_size=local_size, stride=1, padding=int((local_size-1.0)/2))
        self.alpha = alpha
        self.beta = beta

    def forward(self, x):  # pylint: disable=arguments-differ  # pytorch uses a bad name "inputs"
        if self.across_channels:
            div = x.pow(2).unsqueeze(1)
            div = self.average(div).squeeze(1)
            div = div.mul(self.alpha).add(1.0).pow(self.beta)
        else:
            div = x.pow(2)
            div = self.average(div)
            div = div.mul(self.alpha).add(1.0).pow(self.beta)
        x = x.div(div)
        return x
