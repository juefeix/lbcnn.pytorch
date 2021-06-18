import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class RandomBinaryConv(nn.Module):
    """Random Binary Convolution.
    
    See Local Binary Convolutional Neural Networks.
    """
    def __init__(self, in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 sparsity=0.9,
                 bias=False,
                 padding=0,
                 groups=1,
                 dilation=1,
                 seed=1234,
                 deconv=False,
                 output_padding=0):
        """
        TODO(zcq) Write a cuda/c++ version.

        Parameters
        ----------
        sparsity : float

        """
        super().__init__()
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.stride = stride
        self.kernel_size = kernel_size
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.output_padding = output_padding
        num_elements = out_channels * in_channels * kernel_size * kernel_size
        assert not bias, "bias=True not supported"
        weight = torch.zeros((num_elements, ), requires_grad=False).float()
        index = np.random.choice(num_elements, int(sparsity * num_elements))
        weight[index] = torch.bernoulli(torch.ones_like(weight)[index] * 0.5) * 2 - 1
        weight = weight.view((out_channels, in_channels, kernel_size, kernel_size))
        self.register_buffer('weight', weight)

        self.deconv = deconv

    def forward(self, x):
        if self.deconv:
            return F.conv_transpose2d(x, self.weight, stride=self.stride,
                                      padding=self.padding, dilation=self.dilation,
                                      groups=self.groups, output_padding=self.output_padding)
        else:
            return F.conv2d(x, self.weight, stride=self.stride,
                            padding=self.padding, dilation=self.dilation,
                            groups=self.groups)


class RandomBinaryConvV1(nn.Module):
    """Random Binary Convolution.
    
    See Local Binary Convolutional Neural Networks.
    """
    def __init__(self, in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 sparsity=0.9,
                 bias=False,
                 padding=0,
                 dilation=1,
                 groups=1,
                 seed=1234):
        """

        TODO(zcq) Write a cuda/c++ version.

        Parameters
        ----------
        sparsity : float

        """
        super().__init__()
        self.out_channels = out_channels
        self.in_channels = in_channels
        num_elements = out_channels * in_channels * kernel_size * kernel_size
        assert not bias, "bias=True not supported"
        weight = torch.zeros((num_elements, ), requires_grad=False).float()
        index = np.random.choice(num_elements, int(sparsity * num_elements))
        weight[index] = torch.bernoulli(torch.ones_like(weight)[index] * 0.5) * 2 - 1
        weight = weight.view((out_channels, in_channels * kernel_size * kernel_size)).t()
        weight = weight.transpose(0, 1)
        pos_weight = (weight == 1).type(torch.bool)
        neg_weigth = (weight == -1).type(torch.bool)
        self.register_buffer('pos_weight', pos_weight)
        self.register_buffer('neg_weigth', neg_weigth)
        self.unfold = nn.Unfold(kernel_size=kernel_size, padding=padding,
                                stride=stride, dilation=dilation)

    def forward(self, x):
        b, _, h, w = x.shape
        input = self.unfold(x).transpose(1, 2)[..., None] # N HW CKK 1
        pos_input = torch.where(self.pos_weight[None, None, -1, self.out_channels],
                                input, torch.zeros_like(input)) # N HW CKK O
        neg_input = torch.where(self.neg_weight[None, None, -1, self.out_channels],
                                input, torch.zeros_like(input)) # N HW CKK O
        pos_input = torch.sum(pos_input, dim=-2, keepdim=False)
        neg_input = torch.sum(neg_input, dim=-2, keepdim=False)
        output = (pos_input - neg_input).view((b, self.out_channels, h, w))
        return output


class LBConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 sparsity=0.9,
                 bias=False,
                 seed=1234,
                 act=F.relu):
        """Use this to replace a conv + activation.
        """
        super().__init__()
        self.random_binary_conv = RandomBinaryConv(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            sparsity=sparsity,
            seed=seed)
        # self.bn = nn.BatchNorm2d(in_channels)
        self.fc = nn.Conv2d(out_channels, out_channels, 1, 1)
        self.act = act

    def forward(self, x):
        # y = self.bn(x)
        y = self.random_binary_conv(x)
        if self.act is not None:
            y = self.act(y)
        y = self.fc(y)
        return y


class LBConvBN(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 sparsity=0.9,
                 bias=False,
                 seed=1234,
                 padding=1,
                 dilation=1,
                 groups=1,
                 act=F.relu,
                 norm_type='bn',
                 output_padding=0,
                 deconv=False):
        """Use this to replace a conv + activation.
        """
        super().__init__()
        # assert padding == kernel_size // 2, "kernel_size: %d, padding: %d" % (kernel_size, padding)
        # assert dilation == 1, dilation
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.transposed = False
        self.output_padding = 0
        self.dilation = dilation
        self.groups = groups
        self.random_binary_conv = RandomBinaryConv(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            sparsity=sparsity,
            seed=seed,
            dilation=dilation,
            groups=groups,
            padding=padding,
            deconv=deconv,
            output_padding=output_padding)
        if norm_type is None:
            self.bn = None
        elif norm_type == 'bn':
            self.bn = nn.BatchNorm2d(out_channels)
        elif norm_type == 'in':
            self.bn = nn.InstanceNorm2d(out_channels)
        else:
            raise ValueError("%s not supported" % norm_type)

        self.fc = nn.Conv2d(out_channels, out_channels, 1, 1)
        self.act = act

        if bias:
            self.bias = nn.Parameter(
                torch.Tensor((out_channels, )), requires_grad=True)
        else:
            self.bias = None

    def forward(self, x):
        y = self.random_binary_conv(x)
        if self.bn is not None:
            y = self.bn(y)
        if self.act is not None:
            y = self.act(y)
        y = self.fc(y)

        if self.bias is not None:
            y += self.bias
        return y
