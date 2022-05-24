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
            Maximal length of the layer before, by default 512.

        """
        super().__init__()

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


class FlexibleLayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        """Flexible LayerNorm.

        This norm don't need the exact shape size but also it has no trainable
        parameters.

        Parameters
        ----------
        dim : int, list
            Number of dimensions to perform the norm, by default -1
        eps : float
            Epsilon, by default 1e-5

        """
        super().__init__()

        self.eps = eps
        self.dim = dim

    def forward(self, x):
        mean = x.mean(self.dim, keepdim=True)
        std = x.std(self.dim, keepdim=True)

        return (x - mean) / (std + self.eps)
