import torch.nn as nn
import torch.nn.functional as F

class Discriminator(nn.Module):

    def __init__(self, conv_dim, in_channels=3, image_size=32, depth=5):
        super(Discriminator, self).__init__()