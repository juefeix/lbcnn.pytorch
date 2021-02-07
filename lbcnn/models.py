import numpy as np

import torch
import torch.nn as nn



class RandomBinaryConv(nn.Module):
    """Random Binary Convolution.
    
    See Local Binary Convolutional Neural Networks.
    """
    def __init__(self, in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 sparsity=0.9,
                 bias=False,
                 seed=1234):
        """

        Parameters
        ----------
        sparsity : float

        """
        num_elements = out_channels * in_channels * kernel_size * kernel_size
        assert not bias, "bias=True not supported"
        weight = torch.zeros((num_elements, ), requires_grad=False).float()
        index = np.random.choice(num_elements, int(sparsity * num_elements))
        weight[index] = torch.bernoulli(torch.ones_like(weight)[index] * 0.5) * 2 - 1
        weight = weight.reshape((out_channels, in_channels, kernel_size, kernel_size))
        self.register_buffer('weight', weight)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        """Custom load.
        """
        super()._load_from_state_dict(state_dict=state_dict,
                                      prefix=prefix, 
                                      local_metadata=local_metadata,
                                      strict=strict,
                                      missing_keys=missing_keys,
                                      unexpected_keys=unexpected_keys,
                                      error_msgs=error_msgs)


class RandomLocalBinaryConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=0,
                 sparsity=0.9,
                 bias=False,
                 seed=1234):
        self.random_binary_conv = RandomBinaryConv(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            sparsity=sparsity,
            seed=seed)
        self.bn = nn.BatchNorm2d(in_channels)
        self.fc = nn.Conv2d(out_channels, in_channels, 1, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        y = self.bn(x)
        y = self.random_binary_conv(y)
        y = self.relu(y)
        y = self.fc(y)
        o = x + y
        return o
