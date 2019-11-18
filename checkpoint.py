import os
from datetime import datetime
import torch
from generator import Generator
from discriminator import Discriminator

def save(filename, D, G):
    """
    Saves the discriminator and the generator to a file, so they can later be used for inference.
    :param filename: The file name to save the networks to.
    :param D: The discriminator.
    :param G: The generator.
    """
    #filename = prefix + "_" + datetime.now().strftime('%Y%m%d-%H%M%S')
    #save_filename = os.path.splitext(os.path.basename(filename))[0] + '.pt'
    #save_filename = prefix
    torch.save(
        {
            'D_init_params': D.get_init_params(),
            'D_state': D.state_dict(), 
            'G_init_params': G.get_init_params(),
            'G_state': G.state_dict()
        }, 
        filename)
    #return save_filename


def load(filename, use_gpu):
    """
    Loads the trained models of the discriminator and the generator from a file.
    :param filename: The file name to load the networks from.    
    :param use_gpu: Determines whether to load the model to GPU or CPU.
    """

    #save_filename = os.path.splitext(os.path.basename(filename))[0] + '.pt'
    
    checkpoint = torch.load(filename, map_location=lambda storage, loc: storage.cuda() if use_gpu else storage)

    D = Discriminator(*checkpoint['D_init_params'])
    D.load_state_dict(checkpoint['D_state'])

    G = Generator(*checkpoint['G_init_params'])
    G.load_state_dict(checkpoint['G_state'])

    print('Checkpoint loaded.')
    #print(f'Checkpoint {filename} loaded')

    #print(checkpoint)
    #return torch.load(save_filename)
    return D, G

