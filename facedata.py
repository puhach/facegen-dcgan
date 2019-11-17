import torch
from torchvision import datasets
from torchvision import transforms
#import matplotlib.pyplot as plt
import numpy as np


def get_data_loader(batch_size, image_size, data_dir):
    """
    Batch the neural network data using DataLoader.
    :param batch_size: The number of images in a batch.
    :param img_size: The square size to resize the input images to.
    :param data_dir: Directory where image data is located.
    :return: DataLoader with batched data.
    """
    
    preprocessing = transforms.Compose([transforms.Resize((image_size, image_size)),                                         
                                        transforms.ToTensor()])

    dataset = datasets.ImageFolder(root=data_dir, transform=preprocessing)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    return dataloader


def scale(x, input_range=(0, 1), target_range=(-1, 1)):
    """
    Scale takes in an image and returns that image with pixel values scaled to the target range. 
    :param x: The input image(s) as a tensor.
    :param input_range: The tuple (min_input, max_input) specifying the lower and the upper bound of the input values.
    :param target_range: The tuple (min_target, max_target) specifying the lower and the upper bound of the scaled values. 
    :return: A tensor of the same shape as the input containing scaled values.
    """
    
    # scale input to (0, 1) range
    d = input_range[1] - input_range[0]
    assert d > 0, "Invalid input range {0}".format(input_range)
    x = (x - input_range[0]) / d
    #print('min:', x.min())
    #print('max:', x.max())

    # scale from (0, 1) range to the target range
    x = target_range[0] + x*(target_range[1] - target_range[0])
    return x


def postprocess(samples):
    """
    Converts the generated samples to the format appropriate for image display and saving.
    :param samples: Generated image tensors of shape (batch, channels, height, width)
                    containing values ranging from -1 to +1.
    :return: A converted image tensor of shape (batch, height, width, channels) containing 
            uint8 values in range (-1, +1).
    """

    # (batch, channels, height, width) -> (batch, height, width, channels)
    samples = samples.detach().cpu().numpy().transpose(0, 2, 3, 1)

    # scale [-1, 1] back to [0, 255]
    samples = scale(samples, input_range=(-1, +1), target_range=(0, 255))
    return samples.astype(np.uint8)