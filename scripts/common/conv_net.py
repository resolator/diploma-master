from torch import nn


class ConvNet6(nn.Module):
    def __init__(self, out_channels=256, dropout=0.15):
        super().__init__()

        print('========== ConvNet6 args ==========')
        print('out_channels: {}; dropout: {};'.format(
            out_channels, dropout
        ))

        self.fe = nn.Sequential(
            nn.Conv2d(1, 32, (5, 5), (1, 1), (2, 2)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),

            nn.Conv2d(32, 64, (5, 5), (1, 1), (2, 2)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),

            nn.Conv2d(64, 128, (3, 3), (1, 1), (1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),

            nn.Conv2d(128, 128, (3, 3), (1, 1), (1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),

            nn.Conv2d(128, 256, (3, 3), (1, 1), (1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),

            nn.Conv2d(256, out_channels, (1, 1)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),

            nn.Dropout(dropout)
        )

    def forward(self, x):
        """Forward pass.

        Parameters
        ----------
        x : torch.tensor
            Input batch of images with shape (BS, 1, 64, W).

        Returns
        -------
        torch.tensor
            Features map of shape (BS, out_channels, 1, W // 4).
        """

        return self.fe(x)

    def freeze(self):
        """Freeze the network."""
        for param in self.parameters():
            param.requires_grad = False

        print('\n[INFO] The backbone has been frozen.\n')

    def defrost(self):
        """Defrost the network."""
        for param in self.parameters():
            param.requires_grad = True

        print('\n[INFO] The backbone has been defrost.\n')
