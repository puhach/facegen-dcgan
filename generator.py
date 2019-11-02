import torch.nn as nn
import torch.nn.functional as F

def deconv(in_channels, out_channels, kernel_size, stride, padding, activation=None, batch_norm=True):
    """
    A helper function which combines a transpose convolutional layer, batch normalization, and activation.
    :param in_channels: The number of channels (filters) in the input.
    :param out_channels: The number of channels produced by the deconvolution.
    :param kernel_size: The size of the convolving kernel.
    :param stride: The stride of the convolution.
    :param padding: Zero-padding added to both sides of each dimension in the input.
    :param activation: Adds an element-wise activation to the resulting sequence of layers (unless None is passed).
    :param batch_norm: Controls whether to apply batch normalization to the output of the deconvolution.
    :return: A sequential container of layers.
    """

    layers = []
    
    layers.append(nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, 
                                     kernel_size=kernel_size, stride=stride, padding=padding, 
                                     bias = not batch_norm))
    
    if batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))
        
    if activation:
        layers.append(activation)
    
    return nn.Sequential(*layers)