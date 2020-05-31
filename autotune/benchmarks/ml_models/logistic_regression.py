import torch.nn as nn


class LogisticRegression(nn.Module):
    def __init__(self, input_size: int, num_classes: int):
        super().__init__()
        self.linear = nn.Linear(input_size, num_classes)

    def forward(self, x):  # pylint: disable=arguments-differ  # pytorch uses a bad name "inputs"
        out = x.view(-1, 28 * 28)
        out = self.linear(out)
        return out
