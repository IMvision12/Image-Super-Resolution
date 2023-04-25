from .utils import Normalize2SRGAN
import tensorflow as tf


def Discriminator_block(inputs, num_filters, strides=1):
    x = tf.keras.layers.Conv2D(
        num_filters, kernel_size=3, strides=strides, padding="same"
    )(inputs)
    x = tf.keras.layers.BatchNormalization(momentum=0.8)(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    x = tf.keras.layers.Conv2D(num_filters, kernel_size=3, strides=2, padding="same")(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.8)(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    return x


def Discriminator(num_filters=64):
    input_layer = tf.keras.layers.Input(shape=(96, 96, 3))
    x = tf.keras.layers.Lambda(Normalize2SRGAN)(input_layer)
    x = tf.keras.layers.Conv2D(num_filters, kernel_size=3, strides=1, padding='same')(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)

    x = tf.keras.layers.Conv2D(num_filters, kernel_size=3, strides=2, padding='same')(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.8)(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)

    x = Discriminator_block(x, 128)
    x = Discriminator_block(x, 256)
    x = Discriminator_block(x, 512)

    x = tf.keras.layers.Flatten()(x)

    x = tf.keras.layers.Dense(1024)(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    output_layer = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    return tf.keras.models.Model(input_layer, output_layer)