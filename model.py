import torch
import torch.nn as nn


class NetModel(nn.Module):

    def __init__(self):
        super(NetModel, self).__init__()

        self.features = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=12, kernel_size=3),           # out [1, 12, 5, 30, 30]
            nn.ReLU(),
            nn.Conv3d(in_channels=12, out_channels=24, kernel_size=(1, 3, 3)),  # out [1, 24, 5, 28, 28]
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),              # out [1, 24, 5, 14, 14]
            nn.ReLU(),
            nn.Conv3d(in_channels=24, out_channels=48, kernel_size=(3, 3, 3)),  # out [1, 48, 3, 12, 12]
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),              # out [1, 48, 3, 6, 6]
            nn.ReLU(),
            nn.Conv3d(in_channels=48, out_channels=64, kernel_size=(3, 3, 3)),  # out [1, 64, 1, 4, 4]
            nn.ReLU(),
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(1, 4, 4)),  # out [1, 64, 1, 1, 1]
            nn.ReLU(),
        )

        self.classifier = nn.Sequential(
            nn.Linear(in_features=64, out_features=128),  # out [1, 128]
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=2),   # out [1, 2]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
