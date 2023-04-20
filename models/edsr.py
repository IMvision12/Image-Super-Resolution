import tensorflow as tf
from .utils import NormalizeEDSR, DenormalizeEDSR, pixel_shuffle


# ResBlock
def ResBlock(inputs):
    x = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu")(inputs)
    x = tf.keras.layers.Conv2D(64, 3, padding="same")(x)
    x = tf.keras.layers.Add()([inputs, x])
    return x


def Upsampling(inputs, factor=2, **kwargs):
    x = tf.keras.layers.Conv2D(64 * (factor**2), 3, padding="same", **kwargs)(inputs)
    x = tf.keras.layers.Lambda(pixel_shuffle(scale=factor))(x)
    x = tf.keras.layers.Conv2D(64 * (factor**2), 3, padding="same", **kwargs)(x)
    x = tf.keras.layers.Lambda(pixel_shuffle(scale=factor))(x)
    return x


class EDSR(tf.keras.Model):
    def __init__(self, num_blocks=16, num_filters=64, **kwargs):
        input_layer = tf.keras.layers.Input(shape=(None, None, 3))
        x = NormalizeEDSR()(input_layer)
        x = x_new = tf.keras.layers.Conv2D(num_filters, 3, padding="same")(x)

        for _ in range(num_blocks):
            x_new = ResBlock(x_new)

        x_new = tf.keras.layers.Conv2D(num_filters, 3, padding="same")(x_new)
        x = tf.keras.layers.Add()([x, x_new])

        x = Upsampling(x)
        x = tf.keras.layers.Conv2D(3, 3, padding="same")(x)
        output_layer = DenormalizeEDSR()(x)
        super().__init__(inputs=input_layer, outputs=output_layer, **kwargs)

        self.num_blocks = num_blocks
        self.num_filters = num_filters

    def get_config(self):
        return {
            "num_blocks": self.num_blocks,
            "num_filters": self.num_filters,
        }

#EDSR baseline model with 16 residual blocks and 64 filters
#EDSR model with 32 residual blocks and 256 filters