import tensorflow as tf

def normalize(x):
    return x / 255.0

def normalize2(x):
    return x / 127.5 - 1

def denormalize2(x):
    return (x + 1) * 127.5

def pixel_shuffle(scale):
    return lambda x: tf.nn.depth_to_space(x, scale)

def PSNR(sr, hr):
    psnr_value = tf.image.psnr(hr, sr, max_val=255)[0]
    return psnr_value