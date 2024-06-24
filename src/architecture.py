import torch.nn as nn
from torch import flatten


class CNN_SSH(nn.Module):
    def __init__(self):
        super(CNN_SSH, self).__init__()

        self.cnn_layers_1 = nn.Sequential(
            nn.Conv2d(1, 5, kernel_size=4, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(5),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Conv2d(5, 7, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(7),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(p=0.5),
        )

        self.cnn_layers_2 = nn.Sequential(
            nn.Conv2d(7, 9, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(9),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(p=0.5),
        )
        self.avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d(output_size=(1, 1)))

        self.linear_layers = nn.Sequential(nn.Linear(9, 2), nn.Softmax(dim=1))

    def forward(self, x):
        x = self.cnn_layers_1(x)
        x = self.cnn_layers_2(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x


# =============================================================================
#
# =============================================================================


class CNN_SSH_ThermEncod(nn.Module):
    def __init__(self, levels, in_layer_1=1, out_layer_1=7, out_layer_2=9):
        super(CNN_SSH_ThermEncod, self).__init__()
        self.in_layer_1 = in_layer_1
        self.out_layer_1 = out_layer_1
        self.out_layer_2 = out_layer_2

        self.cnn_layers_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=in_layer_1 * levels,
                out_channels=5 * levels,
                kernel_size=4,
                stride=2,
                padding=2,
                bias=False,
            ),
            nn.BatchNorm2d(5 * levels),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Conv2d(
                5 * levels, 7 * levels, kernel_size=3, stride=1, padding=1, bias=False
            ),
            nn.BatchNorm2d(7 * levels),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(p=0.5),
        )

        self.cnn_layers_2 = nn.Sequential(
            nn.Conv2d(
                7 * levels, 9 * levels, kernel_size=3, stride=1, padding=1, bias=False
            ),
            # W_out = ___
            nn.BatchNorm2d(9 * levels),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(p=0.5),
        )
        self.avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d(output_size=(1, 1)))

        self.linear_layers = nn.Sequential(nn.Linear(9 * levels, 2), nn.Softmax(dim=1))

    def forward(self, x):
        x = self.cnn_layers_1(x)
        x = self.cnn_layers_2(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x
