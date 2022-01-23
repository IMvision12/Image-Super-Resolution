from SRResNet import SRResNet_Generator
from SRGAN.data import download_data
import tensorflow as tf
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay

loss_fn = tf.keras.losses.MeanSquaredError()
optim = tf.keras.optimizers.Adam(learning_rate=1e-4)
SRResNet = SRResNet_Generator()
ds = download_data()

@tf.function
def train_step(ds_low, ds_high):
    with tf.GradientTape() as SRResNet_tape:
        ds_low = tf.cast(ds_low, tf.float32)
        ds_high = tf.cast(ds_high, tf.float32)

        sr = SRResNet(ds_low)
        loss_value = loss_fn(ds_high, sr)

    gradients = SRResNet_tape.gradient(loss_value, SRResNet.trainable_variables)
    optim.apply_gradients(zip(gradients, SRResNet.trainable_variables))

    return loss_value


for epoch in range(200):

    for lr, hr in ds.take(1000):
        loss_value = train_step(lr, hr)

    if epoch % 10 == 0 or epoch == 199:
        print(f'Epochs : {epoch}   ||   Loss : {loss_value:.5f}')

