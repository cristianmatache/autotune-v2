from torch.nn import functional, Linear, BatchNorm2d, Conv2d, Module


class CudaConvNet2(Module):

    def __init__(self, n_channels: int, n_units_1: int, n_units_2: int, n_units_3: int):
        super().__init__()

        self.conv1 = Conv2d(n_channels, n_units_1, 5, 1, 2)  # n_channels, n_output, ks, stride, padding
        self.conv2 = Conv2d(n_units_1, n_units_2, 5, 1, 2)
        self.conv3 = Conv2d(n_units_2, n_units_3, 5, 1, 2)

        self.norm1 = BatchNorm2d(n_units_1)
        self.norm2 = BatchNorm2d(n_units_2)
        self.norm3 = BatchNorm2d(n_units_3)

        self.fc1 = Linear(n_units_3 * 4 * 4, 10)

    def forward(self, x):  # pylint: disable=arguments-differ  # pytorch uses a bad name "inputs"
        out = functional.relu(self.conv1(x), inplace=True)
        out = self.norm1(out)
        out = functional.max_pool2d(out, 3, 2, 1)  # ks, stride, padding

        out = functional.relu(self.conv2(out), inplace=True)
        out = self.norm2(out)
        out = functional.max_pool2d(out, 3, 2, 1)

        out = functional.relu(self.conv3(out), inplace=True)
        out = self.norm3(out)
        out = functional.max_pool2d(out, 3, 2, 1)

        out = out.view(out.size(0), -1)

        out = self.fc1(out)
        return out
