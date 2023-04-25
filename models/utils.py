import tensorflow as tf
import numpy as np

@tf.keras.utils.register_keras_serializable("models")
class NormalizeEDSR(tf.keras.layers.Layer):
    def __init__(self):
        super(NormalizeEDSR, self).__init__()
        self.mean = np.array([0.4488, 0.4371, 0.4040]) * 255
        
    def call(self, inputs):
        return tf.divide(tf.subtract(inputs, self.mean), 127.5)

@tf.keras.utils.register_keras_serializable("models")
class DenormalizeEDSR(tf.keras.layers.Layer):
    def __init__(self):
        super(DenormalizeEDSR, self).__init__()
        self.mean = np.array([0.4488, 0.4371, 0.4040]) * 255
        
    def call(self, inputs):
        return tf.add(tf.multiply(inputs, 127.5), self.mean)

@tf.keras.utils.register_keras_serializable("models")
class NormalizeSRGAN(tf.keras.layers.Layer):
    def __init__(self):
        super(NormalizeSRGAN, self).__init__()
        
    def call(self, inputs):
        return tf.divide(inputs, 255.0)

@tf.keras.utils.register_keras_serializable("models")
class Normalize2SRGAN(tf.keras.layers.Layer):
    def __init__(self):
        super(Normalize2SRGAN, self).__init__()
        
    def call(self, inputs):
        return tf.divide(tf.subtract(inputs, 127.5), 127.5)
    
@tf.keras.utils.register_keras_serializable("models")
class Denormalize2SRGAN(tf.keras.layers.Layer):
    def __init__(self):
        super(Denormalize2SRGAN, self).__init__()
        
    def call(self, inputs):
        return tf.multiply(tf.add(inputs, 1), 127.5)

def pixel_shuffle(scale):
    return lambda x: tf.nn.depth_to_space(x, scale)

def PSNR(sr, hr):
    psnr_value = tf.image.psnr(hr, sr, max_val=255)[0]
    return psnr_value
