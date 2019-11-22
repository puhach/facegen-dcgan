import torch
import torch.nn as nn


def conv(in_channels, out_channels, kernel_size, stride, padding, activation, batch_norm=True):
    """
    A helper function combining a convolutional layer, batch normalization, and activation.
    :param in_channels: The number of channels in the input.
    :param out_channels: The number of channels produced by the convolution.
    :param kernel_size: The size of the convolving kernel.
    :param stride: The stride of the convolution.
    :param padding: Zero padding added to both sides of the input.
    :param activation: If not None, adds an element-wise activation to the resulting sequence of layers.
    :param batch_norm: Specifies whether to apply batch normalization to the output of the convolution.
    :return: A sequential container of layers.
    """

    layers = []
    layers.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
                            kernel_size=kernel_size, stride=stride, padding=padding,
                            bias = not batch_norm # disable bias only in case of batch normalization
                           ))
    
    if batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))

    if activation:
        layers.append(activation)
        
    return nn.Sequential(*layers)


class Discriminator(nn.Module):

    def __init__(self, image_size, in_channels, conv_dim, depth):
        """
        Initializes the Discriminator. This is a convolutional classifier only without any maxpooling layers. 
        To deal with this complex data a deep network with normalization is used. 

        :param image_size: The size of input images (single number).
        :param in_channels: The number of channels in the input image.
        :param conv_dim: The depth of the first convolutional layer.
        :param depth: The number of convolutional layers.
        """

        super(Discriminator, self).__init__()

        # save the initialization parameters
        self.image_size = image_size
        self.in_channels = in_channels
        self.conv_dim = conv_dim
        self.depth = depth

        conv_blocks = []
        for i in range(depth):
            conv_blocks.append(conv(in_channels, conv_dim, kernel_size=4, stride=2, padding=1,
                                    ##activation = 'leaky_relu' if i < depth-1 else 'sigmoid',
                                    #activation = nn.LeakyReLU(negative_slope=0.2) if i < depth-1 else nn.Sigmoid(),
                                    activation = nn.LeakyReLU(negative_slope=0.2),
                                    batch_norm = i>0))
            in_channels = conv_dim  # new number of input feature maps
            conv_dim *= 2 # new number of output feature maps
            image_size = (image_size - 4 + 2*1)//2 + 1 # new image size
        
        ## output layer (could use a fully connected layer instead - not sure what is better)
        #conv_blocks.append(conv(in_channels=in_channels, out_channels=1, kernel_size=image_size, stride=1, padding=0, 
        #                       activation = nn.Sigmoid(), batch_norm = False))
        self.conv_layers = nn.Sequential(*conv_blocks)
        
        self.out_layer = nn.Linear(in_features=in_channels*image_size*image_size, out_features=1)


    def forward(self, x):
        """
        Forward propagation of the discriminator.
        :param x: The input to the neural network.     
        :return: Discriminator logits.
        """

        # 1. (3,32,32) -> (64,16,16)
        # 2. (64,16,16) -> (128,8,8)
        # 3. (128,8,8) -> (256,4,4)
        # 4. (256,4,4) -> (512,2,2)
        # 5. (512,2,2) -> (1,1,1)
        x = self.conv_layers(x)
        #print("x':", x.shape)
        #x = x.view(x.shape[0], 1) # remove extra dimensions for the convolutional output layer
        x = x.view(x.shape[0], -1) # flatten (in case of fully connected output layer)
        #x = F.sigmoid(self.out_layer(x))
        x = torch.sigmoid(self.out_layer(x))
        #print("squeezed:", x.shape)
        return x

    def get_init_params(self):
        """
        Returns initialization parameters of the discriminator as a tuple.
        """
        return (self.image_size, self.in_channels, self.conv_dim, self.depth)