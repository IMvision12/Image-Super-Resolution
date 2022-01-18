# Image Super Resolution Using EDSR and SRGAN

1. [Enhanced Deep Residual Networks for Single Image Super-Resolution](https://arxiv.org/abs/1707.02921) (EDSR)
2. [Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network](https://arxiv.org/abs/1609.04802) (SRGAN).

# EDSR 
### Architecture of EDSR :
<p float="left">
  <img src="https://github.com/IMvision12/Image-Super-Resolution/blob/main/Images/edsr.png" width="460" />
  <img src="https://github.com/IMvision12/Image-Super-Resolution/blob/main/Images/residual.PNG" width="360" /> 

  ## About Model:
    1. Baseline model used 16 residual blocks and original model with 32 blocks
    2. No of filters used in all conv2d layers of baseline model were 64 and in original model it was 256
    3. Total no of parameters in baseline model were 1.5M, whereas in original model it was 43M
    4. Loss Function used was L1
  
# SRGAN
### Architecture of SRGAN :
<p float="left">
  <img src="https://github.com/IMvision12/Image-Super-Resolution/blob/main/Images/srgan.PNG" width="800" />
  
  
  ## About Generator and Discriminator:
    1. Total 16 residual blocks were used in Generator Network
    2. Within the residual block, two convolutional layers are used, with small 3×3 kernels and 64 feature maps followed by batch-normalization layers and ParametricReLU.
    3. In Discriminator Network, there are eight convolutional layers with of 3×3 filter kernels, increasing by a factor of 2 from 64 to 512 kernels. 
    4. Strided convolutions are used to reduce the image resolution each time the number of features is doubled.
  ## About Loss Function
    1. The SRGAN uses perpectual loss function
    2. perpectual loss = content loss + adversarial loss
  
# Dataset
* [DIV2K](https://www.tensorflow.org/datasets/catalog/div2k) is a popular single-image super-resolution dataset which contains 1,000 images with different scenes and is splitted to 800 for training, 100 for validation and 100 for testing. This dataset contains low resolution images with different types of degradations. I have used x4 bicubic downsampled images as low resolution image

# Results of EDSR 
![alt_text](https://github.com/IMvision12/Image-Super-Resolution/blob/main/Images/edsr_1.PNG)
![alt_text](https://github.com/IMvision12/Image-Super-Resolution/blob/main/Images/edsr_2.PNG)
