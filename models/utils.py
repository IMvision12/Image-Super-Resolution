import tensorflow as tf
import numpy as np


def NormalizeEDSR(x):
    mean = np.array([0.4488, 0.4371, 0.4040]) * 255
    return (x - mean) / 127.5


def DenormalizeEDSR(x):
    mean = np.array([0.4488, 0.4371, 0.4040]) * 255
    return (x * 127.5) + mean


def NormalizeSRGAN(x):
    return x / 255.0


def Normalize2SRGAN(x):
    return x / 127.5 - 1


def Denormalize2SRGAN(x):
    return (x + 1) * 127.5


def pixel_shuffle(scale):
    return lambda x: tf.nn.depth_to_space(x, scale)


def PSNR(sr, hr):
    psnr_value = tf.image.psnr(hr, sr, max_val=255)[0]
    return psnr_value
