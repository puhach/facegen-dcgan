import torch
from torchvision import datasets
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np


def preview(dataloader, plot_size):
    """
    Reads one batch of images from the dataloader and displays a few of them in a preview window.
    :param dataloader: A dataloader object to read the images from.
    :param plot_size: The number of images to preview (no more than the batch size).
    """

    # obtain one batch of training images
    dataiter = iter(dataloader)
    images, _ = dataiter.next() # _ for no labels

    # plot the images in the batch, along with the corresponding labels
    fig = plt.figure(num = 'Training Data Preview', figsize=(12, 4))
    
    plot_size = min(len(images), plot_size)
    #plot_size = len(images)
    #n = int(math.sqrt(plot_size))

    for idx in np.arange(plot_size):
        ax = fig.add_subplot(2, plot_size/2, idx+1, xticks=[], yticks=[])
        #ax = fig.add_subplot(n, int(math.ceil(plot_size/n)), idx+1, xticks=[], yticks=[])
        ax.imshow(np.transpose(images[idx], (1, 2, 0)))
    
    plt.show()


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


def scale(x, feature_range=(-1, 1)):
    """
    Scale takes in an image and returns that image, scaled with a feature_range of pixel values from -1 to 1. 
    :param x: The input image. This function assumes that it has already been scaled to (0, 1).
    :return: A tensor of the same shape as the input containing scaled values.
    """
    # scale to feature_range and return scaled x    
    return feature_range[0] + x*(feature_range[1] - feature_range[0])