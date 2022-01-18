from utils import normalize,normalize2,denormalize2,pixel_shuffle
import tensorflow as tf

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

def Generator(num_filters=64):
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


def Discriminator_block(inputs, num_filters, strides=1):
    x = tf.keras.layers.Conv2D(num_filters, kernel_size=3, strides=strides, padding='same')(inputs)
    x = tf.keras.layers.BatchNormalization(momentum=0.8)(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    x = tf.keras.layers.Conv2D(num_filters, kernel_size=3, strides=2, padding='same')(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.8)(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    return x


def Discriminator(num_filters=64):
    input_layer = tf.keras.layers.Input(shape=(96, 96, 3))
    x = tf.keras.layers.Lambda(normalize2)(input_layer)
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