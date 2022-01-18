from data import download_data
from model import EDSR
from utils import PSNR

import tensorflow as tf
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay
AUTOTUNE = tf.data.AUTOTUNE

ds = download_data()
edsr_model = EDSR()
loss_fn = tf.keras.losses.MeanAbsoluteError()
optim_edsr = tf.keras.optimizers.Adam(learning_rate=PiecewiseConstantDecay(boundaries=[200000], values=[1e-4, 5e-5]))

@tf.function
def train_step(ds_low, ds_high):
    with tf.GradientTape() as EDSR_tape:
        ds_low = tf.cast(ds_low, tf.float32)
        ds_high = tf.cast(ds_high, tf.float32)

        sr = edsr_model(ds_low)
        loss_value = loss_fn(ds_high, sr)

        # Calculating PSNR value
        psnr_value = PSNR(ds_high, sr)

    gradients = EDSR_tape.gradient(loss_value, edsr_model.trainable_variables)
    optim_edsr.apply_gradients(zip(gradients, edsr_model.trainable_variables))

    return loss_value, psnr_value


for epoch in range(300):

    for lr, hr in ds.take(1000):
        loss_value, psnr_value = train_step(lr, hr)

    if epoch % 50 == 0 or epoch == 299:
        print(f'Epochs : {epoch}   ||   Loss : {loss_value:.5f}   ||   PSNR : {psnr_value:.5f}')

edsr_model.save('model/',save_format='tf')
edsr_model.save("model.h5")