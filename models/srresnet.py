import tensorflow as tf
from .utils import pixel_shuffle, NormalizeSRGAN, Denormalize2SRGAN


def Upsampling(inputs, num_filters):
    x = tf.keras.layers.Conv2D(num_filters, kernel_size=3, padding="same")(inputs)
    x = tf.keras.layers.Lambda(pixel_shuffle(scale=2))(x)
    x = tf.keras.layers.PReLU(shared_axes=[1, 2])(x)
    return x


def ResBlock(inputs, num_filters):
    x = tf.keras.layers.Conv2D(num_filters, kernel_size=3, padding="same")(inputs)
    x = tf.keras.layers.BatchNormalization(momentum=0.8)(x)
    x = tf.keras.layers.PReLU(shared_axes=[1, 2])(x)
    x = tf.keras.layers.Conv2D(num_filters, kernel_size=3, padding="same")(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.8)(x)
    x = tf.keras.layers.Add()([inputs, x])
    return x

@tf.keras.utils.register_keras_serializable("models")
class SRResNet(tf.keras.Model):
    def __init__(self, num_blocks=16, num_filters=64, **kwargs):
        input_layer = tf.keras.layers.Input(shape=(None, None, 3))
        x = NormalizeSRGAN()(input_layer)
        x = tf.keras.layers.Conv2D(num_filters, kernel_size=9, padding="same")(x)
        x = x_new = tf.keras.layers.PReLU(shared_axes=[1, 2])(x)

        for _ in range(num_blocks):
            x = ResBlock(x, num_filters)

        x = tf.keras.layers.Conv2D(num_filters, kernel_size=3, padding="same")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Add()([x_new, x])

        x = Upsampling(x)
        x = Upsampling(x)

        x = tf.keras.layers.Conv2D(3, kernel_size=9, padding="same", activation="tanh")(
            x
        )
        output_layer = Denormalize2SRGAN()(x)
        super().__init__(inputs=input_layer, outputs=output_layer, **kwargs)

        self.num_blocks = num_blocks
        self.num_filters = num_filters

    def get_config(self):
        return {
            "num_blocks": self.num_blocks,
            "num_filters": self.num_filters,
        }