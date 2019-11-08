import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from discriminator import Discriminator
from generator import Generator


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
    
    return dataloader

def scale(x, feature_range=(-1, 1)):
    """
    Scale takes in an image and returns that image, scaled with a feature_range of pixel values from -1 to 1. 
    :param x: The input image. This function assumes that it has already been scaled to (0, 1).
    """
    # scale to feature_range and return scaled x    
    return feature_range[0] + x*(feature_range[1] - feature_range[0])


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
        print(classname)
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

    print(D)
    print()
    print(G)
    
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

def run_training(D, G, n_epochs, print_every=50):
    """
    Trains the adversarial networks for the specified number of epochs.
    :param D: The discriminator network.
    :param G: The generator network.
    :param n_epochs: The number of epochs to train for.
    :param print_every: Controls how often to print and record the models' losses.
    :return: D and G losses.
    """
    
    # move models to GPU
    if train_on_gpu:
        D.cuda()
        G.cuda()

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
        for batch_i, (real_images, _) in enumerate(celeba_train_loader):

            batch_size = real_images.size(0)
            real_images = scale(real_images)

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

    # Save training generator samples
    with open('train_samples.pkl', 'wb') as f:
        pkl.dump(samples, f)
    
    # finally return losses
    return losses


def train(args):

    print("Loading data")

    dataloader = get_dataloader(batch_size=64, image_size=32, data_dir='celeba')

    preview(dataloader, 20)

    imgs, _ = iter(dataloader).next()
    scaled_imgs = scale(imgs)
    print('Min: ', scaled_imgs.min())
    print('Max: ', scaled_imgs.max())

    #print(f"training for {args.epochs} epochs with a learning rate = {args.lr}")
    D, G = build_network(d_conv_dim=64, g_conv_dim=64, z_size=128)

    # Create optimizers for the discriminator D and generator G
    d_optimizer = optim.Adam(D.parameters(), lr=0.0002, betas=[0.5, 0.999])
    g_optimizer = optim.Adam(G.parameters(), lr=0.0002, betas=[0.5, 0.999])
    
    # Check for a GPU
    train_on_gpu = torch.cuda.is_available()
    if not train_on_gpu:
        print('No GPU found. Please use a GPU to train your neural network.')
    else:
        print('Training on GPU!')




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
