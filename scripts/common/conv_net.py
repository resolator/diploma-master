import timm
from torch import nn
from .layers import FlexibleLayerNorm, DepthwiseSepConv2D, Gate


def get_models_list():
    return ['conv_net5', 'conv_net6', 'resnet18']


def get_backbone(name='conv_net6',
                 out_channels=256,
                 dropout=0.15,
                 expand_h=False,
                 gates=0,
                 k=1,
                 gate_dropout=0.4):
    if name == 'conv_net5':
        return ConvNet5(out_channels, dropout), out_channels
    elif name == 'conv_net6':
        return ConvNet6(out_channels,
                        dropout,
                        expand_h,
                        k,
                        gates,
                        gate_dropout), out_channels
    elif name == 'resnet18':
        print('WARNING: backbone out channels is forced to 256 for resnet.')
        fe = timm.create_model(name,
                               pretrained=True,
                               in_chans=1,
                               num_classes=0,
                               global_pool='')
        fe.conv1 = nn.Identity()
        fe.bn1 = nn.Identity()
        fe.act1 = nn.Identity()
        fe.maxpool = nn.Identity()
        fe.layer1[0].conv1 = nn.Conv2d(1, 64, (3, 3), (1, 1), (1, 1),
                                       bias=False)
        fe.layer4 = nn.Identity()

        return fe, 256
    else:
        raise AssertionError('backbone must be in:', get_models_list())

class BaseNet(nn.Module):
    def __init__(self):
        super().__init__()

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


class ConvNet6(BaseNet):
    def __init__(self,
                 out_channels=256,
                 dropout=0.15,
                 expand_h=False,
                 k=1,
                 gates=0,
                 gate_dropout=0.4):
        super().__init__()

        print('========== ConvNet6 args ==========')
        print('out_channels: {}; dropout: {}; expand_h: {}; k: {}; gates: {}; '
              'gate_dropout: {};'.format(
            out_channels, dropout, expand_h, k, gates, gate_dropout
        ))

        self.fe = nn.Sequential(
            nn.Conv2d(1, 32, (5, 5), (1, 1), (2, 2)),
            FlexibleLayerNorm([-2, -1]),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),

            nn.Conv2d(32, 64, (5, 5), (1, 1), (2, 2)),
            FlexibleLayerNorm([-2, -1]),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),

            nn.Conv2d(64, 128, (3, 3), (1, 1), (1, 1)),
            FlexibleLayerNorm([-2, -1]),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),

            nn.Conv2d(128, 128, (3, 3), (1, 1), (1, 1)),
            FlexibleLayerNorm([-2, -1]),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),

            nn.Conv2d(128, 256, (3, 3), (1, 1), (1, 1)),
            FlexibleLayerNorm([-2, -1]),
            nn.ReLU(),
            nn.Identity() if expand_h else nn.MaxPool2d((2, 1)),

            nn.Conv2d(256, out_channels, (1, 1)),
            FlexibleLayerNorm([-2, -1]),
            nn.ReLU(),
            nn.Identity() if expand_h else nn.MaxPool2d((2, 1)),

            nn.Dropout(dropout)
        )

        self.gates = None
        if gates > 0:
            self.gates = nn.Sequential(
                *[ConvNet6.get_gate_block(out_channels, k, gate_dropout)
                  for _ in range(gates)]
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
            Features map of shape (BS, out_channels, 1, W // 4) for
            expand_h=False and (BS, out_channels, H // 16, W // 4) otherwise.

        """
        x = self.fe(x)
        if self.gates is not None:
            x = self.gates(x)

        return x

    @staticmethod
    def get_gate_block(channels, k, dropout=0.4):
        return nn.Sequential(
            DepthwiseSepConv2D(channels, channels * 2, (1, 9), k=k),
            Gate(dim=[-2, -1]),
            nn.Dropout(dropout)
        )


class ConvNet5(BaseNet):
    def __init__(self, out_channels=256, dropout=0.15):
        super().__init__()

        print('========== ConvNet5 args ==========')
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

            nn.Conv2d(128, 256, (3, 3), (1, 1), (1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),

            nn.Conv2d(256, out_channels, (3, 3), (1, 1), (1, 1)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),

            nn.Dropout(dropout)
        )

    def forward(self, x):
        """Forward pass.

        Parameters
        ----------
        x : torch.tensor
            Input batch of images with shape (BS, 1, H, W).

        Returns
        -------
        torch.tensor
            Features map of shape (BS, out_channels, H // 16, W // 4).

        """
        return self.fe(x)
