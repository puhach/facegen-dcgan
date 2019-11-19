import torch
import torch.nn as nn
#import torch.nn.functional as F

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



class Generator(nn.Module):
    
    def __init__(self, target_size, out_channels, z_size, conv_dim, depth=4):
        """
        Initializes the Generator. The generator should upsample an input and generate a new image of 
        the same size as the training data.
        :param target_size: The size of the image to generate (single value).
        :param out_channels: The number of color channels in target images.
        :param z_size: The length of the input latent vector, z.
        :param conv_dim: The depth of the inputs to the *last* transpose convolutional layer.
        :param depth: The number of convolutional layers.
        """
        super(Generator, self).__init__()

        # save the initialization parameters
        self.target_size = target_size
        self.out_channels = out_channels
        self.z_size = z_size
        self.conv_dim = conv_dim
        self.depth = depth

        # calculate the initial image size on the basis of the target image size 
        # and the depth of transpose convolutions
        self.initial_size = target_size // (2**depth)
        self.in_channels = conv_dim * (2**(depth-1))
        assert self.initial_size >= 2, "Decrease the depth of the network."
                
        self.fc = nn.Linear(in_features=z_size, out_features=self.in_channels*self.initial_size**2)
        
        cur_channels = self.in_channels
        image_size = self.initial_size
        deconv_blocks = []
        for _ in range(depth-1):
            deconv_blocks.append(deconv(in_channels=cur_channels, out_channels=cur_channels//2, 
                                       kernel_size=4, stride=2, padding=1, 
                                       activation = nn.ReLU(), batch_norm = True))
            cur_channels //= 2
            image_size *= 2
            
        assert cur_channels == conv_dim
        
        assert image_size*2 == target_size
        
        deconv_blocks.append(nn.ConvTranspose2d(in_channels=conv_dim, out_channels=self.out_channels, 
                                                kernel_size=4, stride=2, padding=1))
        
                
        self.deconv_layers = nn.Sequential(*deconv_blocks)
        

    def forward(self, x):
        """
        Forward propagation of the generator.
        :param x: The input to the neural network.     
        :return: A generated image tensor of the size (target_size, target_size, 3).
        """

        #print("x:", x.shape)
        x = self.fc(x) # (z_size) -> (512*2*2)
        #print("fc:", x.shape)
        x = x.view(x.shape[0], self.in_channels, self.initial_size, self.initial_size) # (512*2*2) -> (512,2,2)
        #print("x.resh:", x.shape)
        # 1. (512,2,2) -> (256,4,4)
        # 2. (256,4,4) -> (128,8,8)
        # 3. (128,8,8) -> (64,16,16)
        # 4. (64,16,16) -> (3,32,32)
        x = torch.tanh(self.deconv_layers(x))
        #x = F.tanh(self.deconv_layers(x))
        return x

    def get_init_params(self):
        """
        Returns initialization parameters of the generator as a tuple.
        """
        return (self.target_size, self.out_channels, self.z_size, self.conv_dim, self.depth)
