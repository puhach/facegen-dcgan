import argparse
import torch
from torchvision import datasets
from torchvision import transforms

import matplotlib.pyplot as plt
import numpy as np

import math

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
    fig = plt.figure(figsize=(12, 4))
    
    plot_size = min(len(images), plot_size)
    #plot_size = len(images)
    #n = int(math.sqrt(plot_size))

    for idx in np.arange(plot_size):
        ax = fig.add_subplot(2, plot_size/2, idx+1, xticks=[], yticks=[])
        #ax = fig.add_subplot(n, int(math.ceil(plot_size/n)), idx+1, xticks=[], yticks=[])
        ax.imshow(np.transpose(images[idx], (1, 2, 0)))

    plt.show()


def get_dataloader(batch_size, image_size, data_dir):
    """
    Batch the neural network data using DataLoader
    :param batch_size: The number of images in a batch
    :param img_size: The square size to resize the input images to
    :param data_dir: Directory where image data is located
    :return: DataLoader with batched data
    """
    
    preprocessing = transforms.Compose([transforms.Resize((image_size, image_size)),                                         
                                        transforms.ToTensor()])

    dataset = datasets.ImageFolder(root=data_dir, transform=preprocessing)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    preview(dataloader, 20)

    return dataloader



def train(args):
    print("Loading data")
    #print(f"training for {args.epochs} epochs with a learning rate = {args.lr}")
    dataloader = get_dataloader(batch_size=64, image_size=32, data_dir='celeba')


def generate(args):
    #print(f"generate to {args.path}")
    print(args.path)

# create the top-level parser
parser = argparse.ArgumentParser()
subparsers = parser.add_subparsers()

# create the parser for the "foo" command
parser_train = subparsers.add_parser('train')
parser_train.add_argument('-epochs', type=int, default=2)
parser_train.add_argument('-lr', type=float, default=0.001)
parser_train.set_defaults(func=train)

# create the parser for the "bar" command
parser_gen = subparsers.add_parser('generate')
parser_gen.add_argument('-path', type=str, required=True, 
    help='The path to the file where the generated image has to be stored.')
parser_gen.set_defaults(func=generate)

args = parser.parse_args("train -lr 0.001 -epochs=4".split())
#args = parser.parse_args("generate -path z:/test.jpg".split())
#args = parser.parse_args()
args.func(args)
