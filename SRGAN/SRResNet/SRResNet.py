import tensorflow as tf
from SRGAN.utils import pixel_shuffle,normalize,denormalize2

def Upsampling(inputs):
    x = tf.keras.layers.Conv2D(256, kernel_size=3, padding='same')(inputs)
    x = tf.keras.layers.Lambda(pixel_shuffle(scale=2))(x)
    x = tf.keras.layers.PReLU(shared_axes=[1, 2])(x)
    return x

def ResBlock(inputs):
    x = tf.keras.layers.Conv2D(64, kernel_size=3, padding='same')(inputs)
    x = tf.keras.layers.BatchNormalization(momentum=0.8)(x)
    x = tf.keras.layers.PReLU(shared_axes=[1, 2])(x)
    x = tf.keras.layers.Conv2D(64, kernel_size=3, padding='same')(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.8)(x)
    x = tf.keras.layers.Add()([inputs, x])
    return x

def SRResNet_Generator(num_filters=64):
    input_layer = tf.keras.layers.Input(shape=(None, None, 3))
    x = tf.keras.layers.Lambda(normalize)(input_layer)
    x = tf.keras.layers.Conv2D(64, kernel_size=9, padding='same')(x)
    x = x_new = tf.keras.layers.PReLU(shared_axes=[1, 2])(x)

    for _ in range(16):
        x = ResBlock(x)

    x = tf.keras.layers.Conv2D(64, kernel_size=3, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Add()([x_new, x])

    x = Upsampling(x)
    x = Upsampling(x)

    x = tf.keras.layers.Conv2D(3, kernel_size=9, padding='same', activation='tanh')(x)
    output_layer = tf.keras.layers.Lambda(denormalize2)(x)

    return tf.keras.models.Model(input_layer, output_layer)