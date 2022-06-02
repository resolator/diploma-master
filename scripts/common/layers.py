import torch
import numpy as np
from torch import nn


class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_len=600):
        """Positional encoding layer.

        Parameters
        ----------
        d_model : int
            Number of channels for the output from layer before this.
        max_len : int, optional
            Maximal length of the layer before, by default 600.

        """
        super().__init__()

        print('========== PositionalEncoder args ==========')
        print('d_model: {}; max_len: {};'.format(
            d_model, max_len
        ))

        position = torch.arange(max_len)
        a = torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model)
        div_term = torch.exp(a).unsqueeze(1)

        pe = torch.zeros(1, d_model, max_len)
        pe[0, 0::2, :] = torch.cos(position * div_term)
        pe[0, 1::2, :] = torch.sin(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x):
        if len(x.shape) == 3:
            return x + self.pe[:, :, :x.size(-1)]  # BS, C, W
        elif len(x.shape) == 4:
            return x + self.pe[:, :, :x.size(-1)].unsqueeze(2)  # BS, C, H, W
        else:
            raise AssertionError(
                'input shape must be 3 (for sequences) or 4 (for images).'
            )


class PositionalEncoder2D(nn.Module):
    def __init__(self, d_model, max_height=8, max_len=600):
        """Positional encoding 2D layer.

        Parameters
        ----------
        d_model : int
            Number of channels for the output from layer before this.
        max_height : int, optional
            Maximal height of the layer before, by default 4.
        max_len : int, optional
            Maximal length of the layer before, by default 600.

        """
        super().__init__()

        print('========== PositionalEncoder2D args ==========')
        print('d_model: {}; max_height: {}; max_len: {};'.format(
            d_model, max_height, max_len
        ))

        pos_w = torch.arange(0, max_len).unsqueeze(1)
        pos_h = torch.arange(0, max_height).unsqueeze(1)
        pe = torch.zeros(1, d_model, max_height, max_len)

        d_model = int(d_model // 2)
        a = torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model)
        div_term = torch.exp(a)

        x1 = torch.sin(pos_w * div_term).T.unsqueeze(1).repeat(1, max_height, 1)
        x2 = torch.cos(pos_w * div_term).T.unsqueeze(1).repeat(1, max_height, 1)
        x3 = torch.sin(pos_h * div_term).T.unsqueeze(2).repeat(1, 1, max_len)
        x4 = torch.cos(pos_h * div_term).T.unsqueeze(2).repeat(1, 1, max_len)

        pe[0, 0:d_model:2, :, :] = x1
        pe[0, 1:d_model:2, :, :] = x2
        pe[0, d_model::2, :, :] = x3
        pe[0, d_model + 1::2, :, :] = x4

        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :, :x.size(-2), :x.size(-1)]  # BS, C, H, W


class FlexibleLayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        """Flexible LayerNorm.

        This norm don't need the exact shape size but also it has no trainable
        parameters.

        Parameters
        ----------
        dim : int, list
            Number of dimensions to perform the norm.
        eps : float
            Epsilon, by default 1e-5.

        """
        super().__init__()

        self.eps = eps
        self.dim = dim

    def forward(self, x):
        mean = x.mean(self.dim, keepdim=True)
        std = x.std(self.dim, keepdim=True)

        return (x - mean) / (std + self.eps)


class Gate(nn.Module):
    def __init__(self, dim, eps=1e-5):
        """Gate layer.

        Number of input channels must be even.

        Parameters
        ----------
        dim : int, list
            Number of dimensions to perform the FlexibleLayerNorm.
        eps : float
            Epsilon for FlexibleLayerNorm, by default 1e-5.

        """
        super().__init__()

        self.x1_norm = FlexibleLayerNorm(dim=dim, eps=eps)
        self.x2_norm = FlexibleLayerNorm(dim=dim, eps=eps)

    def forward(self, x):
        channels = x.size(1)
        assert channels % 2 == 0, 'channels number must be even'

        x1, x2 = x[:, :channels // 2], x[:, channels // 2:]
        x1 = self.x1_norm(torch.tanh(x1))
        x2 = self.x2_norm(torch.sigmoid(x2))

        return x1 * x2
