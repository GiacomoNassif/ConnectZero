import torch
import torch.nn as nn
from functools import reduce


class InitialConvBlock(nn.Module):
    def __init__(self, stream_size):
        super(InitialConvBlock, self).__init__()

        self.network = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=stream_size, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(stream_size),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.network(x)


class ResBlock(nn.Module):
    def __init__(self, stream_size, stride=1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(stream_size, stream_size, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(stream_size)
        self.conv2 = nn.Conv2d(stream_size, stream_size, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(stream_size)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        self.activation(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        self.activation(out)

        return out


class ResStream(nn.Module):
    def __init__(self, stream_size, stream_length):
        super(ResStream, self).__init__()
        self.network = nn.Sequential(*[ResBlock(stream_size) for _ in range(stream_length)])

    def forward(self, x):
        return self.network(x)


class ValueBlock(nn.Module):
    def __init__(self, in_channels, board_size: tuple[int]):
        super(ValueBlock, self).__init__()

        self.network = nn.Sequential(
            nn.Conv2d(in_channels, 3, kernel_size=1),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3 * reduce(lambda x, y: x * y, board_size), 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.network(x)


class ProbabilityBlock(nn.Module):
    def __init__(self, in_channels, board_size, action_size):
        super(ProbabilityBlock, self).__init__()

        self.network = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * reduce(lambda x, y: x * y, board_size), action_size),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        return self.network(x)


class FinalLayer(nn.Module):
    def __init__(self, in_channels, board_size, action_size):
        super(FinalLayer, self).__init__()

        self.value = ValueBlock(in_channels, board_size)
        self.probability = ProbabilityBlock(in_channels, board_size, action_size)

    def forward(self, x):
        return self.probability(x), self.value(x)


class AlphaNet(nn.Module):
    def __init__(self, stream_size, stream_length, board_size, action_size):
        super(AlphaNet, self).__init__()

        self.initialConvolution = InitialConvBlock(stream_size)
        self.residualStream = ResStream(stream_size, stream_length)
        self.finalLayer = FinalLayer(stream_size, board_size, action_size)

    def forward(self, x):
        x = self.initialConvolution(x)
        x = self.residualStream(x)
        return self.finalLayer(x)


class AlphaLoss(nn.Module):
    def __init__(self):
        super(AlphaLoss, self).__init__()

        self.loss_value = nn.MSELoss(reduction='sum')
        self.loss_policy = nn.CrossEntropyLoss()

    def forward(self, y_value, value, y_policy, policy):
        return self.loss_value(y_value, value) + self.loss_policy(y_policy, policy)
