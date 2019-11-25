import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pickle as pkl
import facedata
import display
import checkpoint
from discriminator import Discriminator
from generator import Generator
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
 

def build_network(image_size, z_size, d_conv_dim, d_conv_depth, g_conv_dim, g_conv_depth):
    """
    Creates the discriminator and the generator.
    :param image_size: The size of input and target images.
    :param z_size: The length of the input latent vector, z.
    :param d_conv_dim: The depth of the first convolutional layer of the discriminator.
    :param d_conv_depth: The number of convolutional layers of the discriminator.
    :param g_conv_dim: The depth of the inputs to the *last* transpose convolutional layer of the generator.    
    :param g_conv_depth: The number of convolutional layers of the generator.
    :return: A tuple of discriminator and generator instances.
    """

    # define discriminator and generator
    D = Discriminator(image_size=image_size, in_channels=3, conv_dim=d_conv_dim, depth=d_conv_depth)
    G = Generator(target_size=image_size, out_channels=3, z_size=z_size, conv_dim=g_conv_dim, depth=g_conv_depth)

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

def run_training(D, G, d_optimizer, g_optimizer, dataloader, z_size, n_epochs, train_on_gpu, 
    model_path, sample_generated, print_every=50):
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
    :param model_path: The path to a file where the model artifact will be saved.
    :param sample_generated: Specifies whether to save generated image samples for preview.
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


    # Check if the user asked to sample generated images
    if sample_generated: 

        # Get some fixed data for sampling. These are images that are held
        # constant throughout training, and allow us to inspect the model's performance.
        
        samples = []    # keep track of the generated samples
        
        sample_size=16
        
        fixed_z = np.random.uniform(-1, 1, size=(sample_size, z_size))
        fixed_z = torch.from_numpy(fixed_z).float()
        
        # Move z to GPU if available
        if train_on_gpu:
            fixed_z = fixed_z.cuda()


    losses = [] # keep track of the training loss    

    # epoch training loop
    for epoch in range(n_epochs):

        # batch training loop
        for batch_i, (real_images, _) in enumerate(dataloader):

            batch_size = real_images.size(0)
            real_images = facedata.scale(real_images, input_range=(0, 1))

            if train_on_gpu:
                real_images = real_images.cuda()
            
            # 1. Train the discriminator on real and fake images
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
                # append discriminator loss and generator loss
                losses.append((d_loss.item(), g_loss.item()))
                
                # print discriminator and generator loss
                print('Epoch [{:5d}/{:5d}] | Batch [{:5d}/{:5d}] | d_loss: {:6.4f} | g_loss: {:6.4f}'.format(
                        epoch+1, n_epochs, batch_i, len(dataloader), d_loss.item(), g_loss.item()))


        # After each epoch generate and save sample (fake images)
        if sample_generated:            
            G.eval() # for generating samples
            samples_z = G(fixed_z)
            samples.append(samples_z)
            G.train() # back to training mode

    
    checkpoint.save(model_path, D, G) # save the trained model

    # If needed, save the generated samples
    if sample_generated:
        with open('train_samples.pkl', 'wb') as f:
            pkl.dump(samples, f)
        

    # finally return losses
    return losses


def train(args):

    print("Loading data...")

    dataloader = facedata.get_data_loader(batch_size=args.batch_size, image_size=args.imsize, data_dir='celeba')

    if args.previn:
        display.preview_input(dataloader, 20)
   
    z_size = args.z_size

    D, G = build_network(image_size=args.imsize, d_conv_dim=64, d_conv_depth=4, g_conv_dim=64, g_conv_depth=4, z_size=z_size)

    # Create optimizers for the discriminator D and generator G
    d_optimizer = optim.Adam(D.parameters(), lr=args.lr, betas=[args.beta1, args.beta2])
    g_optimizer = optim.Adam(G.parameters(), lr=args.lr, betas=[args.beta1, args.beta2])
    
    n_epochs = args.epochs

    losses = run_training(D, G, d_optimizer, g_optimizer, dataloader, 
        z_size=z_size, n_epochs=n_epochs, train_on_gpu=torch.cuda.is_available(),
        model_path=args.model, sample_generated=args.prevgen)
    
    if args.losses:
        display.plot_training_losses(losses)

    if args.prevgen:
        with open('train_samples.pkl', 'rb') as f:
            samples = pkl.load(f)

        # view samples from the last epoch of training
        display.view_samples(facedata.postprocess(samples[-1]))

    print('Done!')


def generate(args):

    print('Loading the model artifact "{0}"'.format(args.model))

    _, G = checkpoint.load(args.model, args.gpu)
    
    n = args.n
    z = np.random.uniform(-1, 1, size=(n, G.z_size))
    z = torch.from_numpy(z).float()
    
    # move z to GPU if available
    if args.gpu:
        G.cuda()
        z = z.cuda()

    G.eval() # for generating samples

    samples = G(z)

    samples = facedata.postprocess(samples)

    display.view_samples(samples)
  
    
    # save images
    print('Generating to "{0}"'.format(args.output))

    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)

    for i in range(len(samples)):
        file_name = str(i+1)
        file_name = output_dir + '/' + file_name.zfill(int(math.log10(n)) + 1) + args.ext
        imageio.imwrite(file_name, samples[i])

    print('Done!')




def validate_positive_int(value):
    """
    Checks whether the value is a valid positive integer.
    """

    try:
        n = int(value)
    except ValueError as val_err:
        raise argparse.ArgumentTypeError(val_err)

    if n < 1:
        raise argparse.ArgumentTypeError('"{0}" is not a positive integer value.'.format(value))

    return n

def validate_positive_float(value):
    """
    Checks whether the value is a valid positive float.
    """
    try:
        n = float(value)
    except ValueError as val_err:
        raise argparse.ArgumentTypeError(val_err)

    if n <= 0:
        raise argparse.ArgumentTypeError('"{0}" is not a positive float value.'.format(value))

    return n



def validate_image_size(value):
    """
    Checks whether the value is a valid image size.
    """
    n = validate_positive_int(value)

    if (n & (n - 1)) != 0:
        raise argparse.ArgumentTypeError('"{0}" is not a power of 2.'.format(value))

    return n


# create the top-level parser
parser = argparse.ArgumentParser()
subparsers = parser.add_subparsers()

# create the parser for the "foo" command
parser_train = subparsers.add_parser('train')
parser_train.add_argument('-imsize', type=validate_image_size, required=True, 
    help='The size of input and output images. Must be a single value, a power of 2.')
parser_train.add_argument('-epochs', type=validate_positive_int, default=2, help='The number of epochs to train for.')
parser_train.add_argument('-lr', type=validate_positive_float, default=0.0002, 
    help='The learning rate. Default is 0.0002.')
parser_train.add_argument('-beta1', type=validate_positive_float, default=0.5,
    help='The exponential decay rate for the first moment estimates. Default is 0.5.')
parser_train.add_argument('-beta2', type=validate_positive_float, default=0.999,
    help='The exponential decay rate for the second moment estimates. Default is 0.999.')
parser_train.add_argument('-batch', dest='batch_size', type=validate_positive_int, default=64, help='The batch size. Default is 64.')
parser_train.add_argument('-zsize', dest='z_size', type=validate_positive_int, default=128, 
    help='The latent vector size. Default is 128.')
parser_train.add_argument('-model', type=str, default='model.pth',
    help='The path to a file where the model artifact will be saved. If omitted, defaults to model.pth.')
# 'store_true' and 'store_false' - These are special cases of 'store_const' used for storing the values True and False 
# respectively. In addition, they create default values of False and True respectively.
parser_train.add_argument('-previn', action='store_true', 
    help='If present, the input data preview will be displayed before training.')
parser_train.add_argument('-no-samples', dest='prevgen', action='store_false',
    help='Disables the preview of images generated while training.')
parser_train.add_argument('-losses', action='store_true',
    help='Activates plotting of the learning curves.')
parser_train.set_defaults(func=train)

# create the parser for the "bar" command
parser_gen = subparsers.add_parser('generate')
parser_gen.add_argument('-n', type=validate_positive_int, required=True,
    help='Specifies the number of images to generate.')
parser_gen.add_argument('-model', type=str, default='model.pth',
    help='The path to a file containing the model artifact. If omitted, defaults to model.pth.')
parser_gen.add_argument('-output', type=str, required=True, 
    help='The path to the file where the generated image has to be stored.')
parser_gen.add_argument('-ext', type=str, default='.jpg', 
    help='Allows to specify the generated image format. Defaults to .jpg.')
#parser_gen.add_argument('-gpu', default=torch.cuda.is_available(), action='store_true',
#    help='Determines whether GPU acceleration should be used for image generation.')
parser_gpu = parser_gen.add_mutually_exclusive_group(required=False)
parser_gpu.add_argument('-gpu', dest='gpu', action='store_true',
    help='Use GPU acceleration for generating images. Set by default if GPU is available.')
parser_gpu.add_argument('-cpu', dest='gpu', action='store_false',
    help='Do not use GPU acceleration for generating images.')
parser_gpu.set_defaults(gpu=torch.cuda.is_available())
parser_gen.set_defaults(func=generate)

#args = parser.parse_args("train -lr 0.0001 -epochs=1 -imsize=64 -model z:/model.pth".split())
#args = parser.parse_args("train -epochs=1 -imsize=32 -model z:/model.pth".split())
#args = parser.parse_args("generate -n 10 -model model.pth -output z:/generated -ext=.png".split())
args = parser.parse_args()
args.func(args)
