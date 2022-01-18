import tensorflow as tf
from utils import normalize,denormalize,pixel_shuffle


# ResBlock
def ResBlock(inputs):
    x = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu')(inputs)
    x = tf.keras.layers.Conv2D(64, 3, padding='same')(x)
    x = tf.keras.layers.Add()([inputs, x])
    return x

def Upsampling(inputs, factor=2, **kwargs):
    x = tf.keras.layers.Conv2D(64 * (factor ** 2), 3, padding='same', **kwargs)(inputs)
    x = tf.keras.layers.Lambda(shuffle_pixels(scale=factor))(x)
    x = tf.keras.layers.Conv2D(64 * (factor ** 2), 3, padding='same', **kwargs)(x)
    x = tf.keras.layers.Lambda(shuffle_pixels(scale=factor))(x)
    return x


# EDSR Model
def EDSR():
    input_layer = tf.keras.layers.Input(shape=(None, None, 3))
    x = tf.keras.layers.Lambda(normalize)(input_layer)
    x = x_new = tf.keras.layers.Conv2D(64, 3, padding='same')(x)

    for _ in range(16):
        x_new = ResBlock(x_new)

    x_new = tf.keras.layers.Conv2D(64, 3, padding='same')(x_new)
    x = tf.keras.layers.Add()([x, x_new])

    x = Upsampling(x)
    x = tf.keras.layers.Conv2D(3, 3, padding='same')(x)
    output_layer = tf.keras.layers.Lambda(denormalize)(x)
    return tf.keras.models.Model(input_layer, output_layer)

#edsr = EDSR()
#edsr.summary()