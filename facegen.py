import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pickle as pkl
import facedata
import checkpoint
from discriminator import Discriminator
from generator import Generator
import matplotlib.pyplot as plt
import imageio
import os
#import pathlib
import math


def weights_init_normal(m):
    """
    Applies initial weights to certain layers in the model.
    The weights are taken from a normal distribution with mean = 0, std dev = 0.02.
    :param m: A module or layer in the network.
    """

    # classname will be something like:
    # `Conv`, `BatchNorm2d`, `Linear`, etc.
    classname = m.__class__.__name__
    
    # Apply initial weights to convolutional and linear layers
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        #print(classname)
        if hasattr(m, 'weight') and m.weight is not None:
            nn.init.normal_(m.weight, mean=0, std=0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.normal_(m.bias, mean=0, std=0.02)
 

def build_network(d_conv_dim, g_conv_dim, z_size):
    # define discriminator and generator
    D = Discriminator(d_conv_dim)
    G = Generator(z_size=z_size, conv_dim=g_conv_dim)

    # initialize model weights
    D.apply(weights_init_normal)
    G.apply(weights_init_normal)

    #print(D)
    #print()
    #print(G)
    
    return D, G


def real_loss(D_out):
    """
    Calculates how close discriminator outputs are to being real.
    :param D_out: Discriminator logits.
    :return: Real loss.
    """    
    loss = torch.mean((D_out - 1)**2)
    return loss

def fake_loss(D_out):
    """
    Calculates how close discriminator outputs are to being fake.
    :param D_out: Discriminator logits.
    :return: Fake loss.
    """
    loss = torch.mean((D_out - 0)**2)
    return loss

def run_training(D, G, d_optimizer, g_optimizer, dataloader, z_size, n_epochs, train_on_gpu, print_every=50):
    """
    Trains the adversarial networks for the specified number of epochs.
    :param D: The discriminator network.
    :param G: The generator network.
    :param d_optimizer: The discriminator optimizer.
    :param g_optimizer: The generator optimizer.
    :param dataloader: Provides an iterable over the training set.
    :param z_size: The latent vector size.
    :param n_epochs: The number of epochs to train for.
    :param train_on_gpu: Determines whether to train the networks on GPU (faster).
    :param print_every: Controls how often to print and record the models' losses.
    :return: D and G losses.
    """
    
    # move models to GPU
    if train_on_gpu:
        print('Training on GPU...')
        D.cuda()
        G.cuda()
    else:
        print('CUDA is not available. Training on CPU...')

    # keep track of loss and generated, "fake" samples
    samples = []
    losses = []

    # Get some fixed data for sampling. These are images that are held
    # constant throughout training, and allow us to inspect the model's performance
    sample_size=16
    fixed_z = np.random.uniform(-1, 1, size=(sample_size, z_size))
    fixed_z = torch.from_numpy(fixed_z).float()
    # move z to GPU if available
    if train_on_gpu:
        fixed_z = fixed_z.cuda()

    # epoch training loop
    for epoch in range(n_epochs):

        # batch training loop
        for batch_i, (real_images, _) in enumerate(dataloader):

            batch_size = real_images.size(0)
            real_images = facedata.scale(real_images)

            if train_on_gpu:
                real_images = real_images.cuda()
            
            # 1. Train the discriminator on real and fake images
            #d_out = D(real_images)
            d_real_loss = real_loss(D(real_images))
            
            z = torch.from_numpy(np.random.uniform(-1, 1, size=(batch_size, z_size))).float()
            if train_on_gpu:
                z = z.cuda()
                
            d_fake_loss = fake_loss(D(G(z)))
            
            d_loss = d_real_loss + d_fake_loss
            
            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()

            # 2. Train the generator with an adversarial loss
            z = torch.from_numpy(np.random.uniform(-1, 1, size=(batch_size, z_size))).float()
            if train_on_gpu:
                z = z.cuda()
                
            g_loss = real_loss(D(G(z)))
            
            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()
            
            
            # Print some loss stats
            if batch_i % print_every == 0:
                #checkpoint.save('checkpoint.pt', D, G)
                #checkpoint.load('checkpoint.pt')
                # append discriminator loss and generator loss
                losses.append((d_loss.item(), g_loss.item()))
                # print discriminator and generator loss
                print('Epoch [{:5d}/{:5d}] | d_loss: {:6.4f} | g_loss: {:6.4f}'.format(
                        epoch+1, n_epochs, d_loss.item(), g_loss.item()))


        # after each epoch generate and save sample (fake images)
        G.eval() # for generating samples
        samples_z = G(fixed_z)
        samples.append(samples_z)
        G.train() # back to training mode

    
    # save the trained model
    checkpoint.save('model.pth', D, G)


    # Save training generator samples
    with open('train_samples.pkl', 'wb') as f:
        pkl.dump(samples, f)
    

    # finally return losses
    return losses


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

    fig, axes = plt.subplots(num='Generated Samples Preview', 
        figsize=(6,6), nrows=4, ncols=4, sharey=True, sharex=True)

    for ax, img in zip(axes.flatten(), samples):
        img = img.detach().cpu().numpy()
        img = np.transpose(img, (1, 2, 0))
        img = ((img + 1)*255 / (2)).astype(np.uint8)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        im = ax.imshow(img.reshape((32,32,3)))

    plt.show()


def train(args):

    print("Loading data...")

    dataloader = facedata.get_data_loader(batch_size=64, image_size=32, data_dir='celeba')

    facedata.preview(dataloader, 20)

    #imgs, _ = iter(dataloader).next()
    #scaled_imgs = scale(imgs)
    #print(f'Min: {scaled_imgs.min()} Max: {scaled_imgs.max()}')

    #print(f"training for {args.epochs} epochs with a learning rate = {args.lr}")
    
    z_size = 128

    D, G = build_network(d_conv_dim=64, g_conv_dim=64, z_size=z_size)

    # Create optimizers for the discriminator D and generator G
    d_optimizer = optim.Adam(D.parameters(), lr=0.0002, betas=[0.5, 0.999])
    g_optimizer = optim.Adam(G.parameters(), lr=0.0002, betas=[0.5, 0.999])
    
    n_epochs = args.epochs

    losses = run_training(D, G, d_optimizer, g_optimizer, dataloader, 
        z_size=z_size, n_epochs=n_epochs, train_on_gpu=torch.cuda.is_available())

    plot_training_losses(losses)

    with open('train_samples.pkl', 'rb') as f:
        samples = pkl.load(f)

    # view samples from the last epoch of training
    _ = view_samples(samples[-1])




def generate(args):
    print("Generating...")
    _, G = checkpoint.load('model.pth')
    #print(f"generate to {args.path}")

    use_gpu = False
    sample_size=16
    z = np.random.uniform(-1, 1, size=(sample_size, G.z_size))
    z = torch.from_numpy(z).float()
    # move z to GPU if available
    if use_gpu:
        G.cuda()
        z = z.cuda()

    G.eval() # for generating samples
    
    samples = G(z)

    view_samples(samples)

    # TODO: use this conversion before viewing samples and adjust the view_samples function
    # (batch, channels, size, size) -> (batch, size, size, channels)
    samples = samples.detach().cpu().numpy().transpose(0, 2, 3, 1)
    
    # TODO: perhaps, create a separate function to perform universal scaling
    # scale [-1, 1] back to [0, 255]
    samples = ((samples + 1) * 255 / 2).astype(np.uint8)

    # save images
    output_dir = 'generated'
    os.makedirs(output_dir, exist_ok=True)
    for i in range(len(samples)):
        file_name = str(i+1)
        file_name = output_dir + '/' + file_name.zfill(int(math.log10(sample_size)) + 1) + '.jpg'
        #print(samples[i].shape)
        imageio.imwrite(file_name, samples[i])

    print(args.path)


# create the top-level parser
parser = argparse.ArgumentParser()
subparsers = parser.add_subparsers()

# create the parser for the "foo" command
parser_train = subparsers.add_parser('train')
parser_train.add_argument('-epochs', type=int, default=2, help='The number of epochs to train for.')
parser_train.add_argument('-lr', type=float, default=0.001, help='The learning rate.')
parser_train.set_defaults(func=train)

# create the parser for the "bar" command
parser_gen = subparsers.add_parser('generate')
parser_gen.add_argument('-path', type=str, required=True, 
    help='The path to the file where the generated image has to be stored.')
parser_gen.set_defaults(func=generate)

#args = parser.parse_args("train -lr 0.001 -epochs=4".split())
args = parser.parse_args("generate -path z:/test.jpg".split())
#args = parser.parse_args()
args.func(args)
