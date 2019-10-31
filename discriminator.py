import torch.nn as nn
import torch.nn.functional as F

class Discriminator(nn.Module):

    def __init__(self, conv_dim, in_channels=3, image_size=32, depth=5):
        
        super(Discriminator, self).__init__()

        conv_blocks = []
        for i in range(depth-1):
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