import tensorflow as tf
import numpy as np

DIV2K_RGB_MEAN = np.array([0.4488, 0.4371, 0.4040]) * 255

def normalize(x,rgb_mean=DIV2K_RGB_MEAN):
    return (x - rgb_mean) / 127.5

def denormalize(x,rgb_mean=DIV2K_RGB_MEAN):
    return (x * 127.5 )+ rgb_mean

def pixel_shuffle(scale):
    return lambda x: tf.nn.depth_to_space(x, scale)

def PSNR(sr, hr):
    psnr_value = tf.image.psnr(hr, sr, max_val=255)[0]
    return psnr_value