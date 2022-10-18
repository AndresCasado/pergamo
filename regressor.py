import torch.nn as nn


class Regressor(nn.Module):
    def __init__(self, in_channels=69, out_channels=4424):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.flatten = nn.Flatten()

        self.hidden_size1 = int(self.out_channels)
        self.hidden_size2 = int(self.out_channels)

        self.model = nn.Sequential(
            nn.Linear(self.in_channels, self.hidden_size1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(self.hidden_size1, self.hidden_size2),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(self.hidden_size2, self.out_channels * 3),
            nn.Unflatten(1, (self.out_channels, 3)),
        )

    def forward(self, x):
        x = self.flatten(x)
        z = self.model(x)
        return z
