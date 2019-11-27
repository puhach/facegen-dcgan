# Face Generator
This project generates new faces using a Deep Convolutional Generative Adversarial Network (DCGAN). 

![Generated Samples](./generated_samples_preview.png)

The model succeeds in reaching equilibrium between the generator and the discriminator. After several epochs of training it generates faces which are clearly recognizable.

# Setup

You will need Python 3.5+. 

GPU device is recommended to accelerate training. 

## Install PyTorch 1.1+ and TorchVision

If you are planning to use CUDA (recommended for faster training), run this command:

```
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
```

If you don't have CUDA, use the following:

```
conda install pytorch torchvision cpuonly -c pytorch
```

## Install Matplotlib

```
conda install matplotlib
```
Matplotlib is used for input data preview, to display generated images and learning curves.


## Install Imageio

```
conda install imageio
```

Imageio is a Python library that provides an easy interface to read and write a wide range of image data.



