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

@tf.keras.utils.register_keras_serializable("models")
class Discriminator(tf.keras.Model):
    def __init__(self, num_filters=64, **kwargs):
        input_layer = tf.keras.layers.Input(shape=(96, 96, 3))
        x = Normalize2SRGAN()(input_layer)
        x = tf.keras.layers.Conv2D(
            num_filters, kernel_size=3, strides=1, padding="same"
        )(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)

        x = tf.keras.layers.Conv2D(
            num_filters, kernel_size=3, strides=2, padding="same"
        )(x)
        x = tf.keras.layers.BatchNormalization(momentum=0.8)(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)

        x = Discriminator_block(x, 128)
        x = Discriminator_block(x, 256)
        x = Discriminator_block(x, 512)

        x = tf.keras.layers.Flatten()(x)

        x = tf.keras.layers.Dense(1024)(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
        output_layer = tf.keras.layers.Dense(1, activation="sigmoid")(x)
        super().__init__(inputs=input_layer, outputs=output_layer, **kwargs)

        self.num_filters = num_filters

    def get_config(self):
        return {
            "num_filters": self.num_filters,
        }