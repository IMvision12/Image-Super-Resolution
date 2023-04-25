import tensorflow as tf

mse = tf.keras.losses.MeanSquaredError()
binary_crossentropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)


def discriminator_loss(fake_output, real_output):
    cross_entropy = tf.keras.losses.BinaryCrossentropy()
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


def generator_loss(fake_output):
    cross_entropy = tf.keras.losses.BinaryCrossentropy()
    return cross_entropy(tf.ones_like(fake_output), fake_output)


def content_loss(content_model, sr, hr):
    mse = tf.keras.losses.MeanSquaredError()
    sr = tf.keras.applications.vgg19.preprocess_input(sr)
    hr = tf.keras.applications.vgg19.preprocess_input(hr)
    hr_feature = content_model(hr) / 12.75
    sr_feature = content_model(sr) / 12.75

    return mse(hr_feature, sr_feature)


def Content_Net(size=None, channels=3, i=5, j=4):
    vgg19 = tf.keras.applications.VGG19(
        weights="imagenet", include_top=False, input_shape=(size, size, channels)
    )
    block_name = "block{}_conv{}".format(i, j)
    model = tf.keras.Model(
        inputs=vgg19.input, outputs=vgg19.get_layer(block_name).output
    )
    return model
