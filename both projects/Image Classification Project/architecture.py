import torch
import torch.nn as nn


def conv_layer_bn(in_channels, out_channels, kernel_size: int = 3, padding="same"):
    layer = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )
    return layer


class MyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            conv_layer_bn(in_channels=1, out_channels=32),
            conv_layer_bn(in_channels=32, out_channels=32),
            nn.MaxPool2d(kernel_size=2, stride=2),
            conv_layer_bn(in_channels=32, out_channels=64),
            conv_layer_bn(in_channels=64, out_channels=64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            conv_layer_bn(in_channels=64, out_channels=128),
            conv_layer_bn(in_channels=128, out_channels=128),
            conv_layer_bn(in_channels=128, out_channels=128),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.25),
            conv_layer_bn(in_channels=128, out_channels=256),
            conv_layer_bn(in_channels=256, out_channels=256),
            conv_layer_bn(in_channels=256, out_channels=256),
            conv_layer_bn(in_channels=256, out_channels=256),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.25),
            conv_layer_bn(in_channels=256, out_channels=512),
            conv_layer_bn(in_channels=512, out_channels=512),
            nn.MaxPool2d(kernel_size=2, stride=1),
        )

        in_features = 512 * 5 * 5
        self.fc = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(in_features=1024, out_features=1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(in_features=1024, out_features=20)
        )

    def forward(self, input_images: torch.Tensor) -> torch.Tensor:
        x = self.cnn(input_images)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        return x


model = MyCNN()
