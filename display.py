import math
import numpy as np
import matplotlib.pyplot as plt


def preview_input(dataloader, plot_size):
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


def plot_training_losses(losses):
    """
    Plots the training losses for the generator and discriminator recorded after each epoch.
    :param losses: A list of tuples of the discriminator and generator losses.
    """
    
    fig, ax = plt.subplots(num='Training Losses')    
    losses = np.array(losses)
    plt.plot(losses.T[0], label='Discriminator', alpha=0.5)
    plt.plot(losses.T[1], label='Generator', alpha=0.5)
    plt.title("Training Losses")
    plt.legend()
    plt.show()
    

def view_samples(samples):
    """
    A helper function for viewing a list of generated images.
    :param samples: The list of generated samples (tensors).
    """

    nrows = int(math.sqrt(len(samples)))
    ncols = int(math.ceil(len(samples) / nrows))

    #fig, axes = plt.subplots(num='Generated Samples Preview', 
    #    figsize=(ncols,nrows), nrows=nrows, ncols=ncols, sharey=True, sharex=True)
    fig = plt.figure(num = 'Generated Samples Preview', figsize=(ncols, nrows))

    for i, sample in enumerate(samples):
        ax = fig.add_subplot(nrows, ncols, i+1, xticks=[], yticks=[])
        #ax = fig.add_subplot(n, int(math.ceil(plot_size/n)), idx+1, xticks=[], yticks=[])
        #img = sample.detach().cpu().numpy()
        #img = np.transpose(img, (1, 2, 0))
        #img = ((img + 1)*255 / (2)).astype(np.uint8)
        #ax.imshow(np.transpose(images[idx], (1, 2, 0)))
        #ax.imshow(img.reshape((32,32,3)))
        ax.imshow(sample)


    plt.show()

