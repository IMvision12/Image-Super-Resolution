from loss import discriminator_loss,mse_based_loss,generator_loss,content_loss,Content_Net
from model import Generator,Discriminator
from utils import PSNR
from data import download_data
import tensorflow as tf
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay

#Loading SRResNet Pre-trained Model
#Using SRResNet as generator
new_gen = tf.keras.models.load_model("SRResNet_model.h5")

#Discriminator Model
new_disc = Discriminator()

#VGG19 feature extractor
content_model = Content_Net()

#Optimizers
generator_optimizer = tf.keras.optimizers.Adam(learning_rate=PiecewiseConstantDecay(boundaries=[10000], values=[1e-4, 1e-5]))
discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=PiecewiseConstantDecay(boundaries=[10000], values=[1e-4, 1e-5]))

#Dataset
ds = download_data()

@tf.function
def train_step(low_res, high_res):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        low_res = tf.cast(low_res, tf.float32)
        high_res = tf.cast(high_res, tf.float32)

        SR = new_gen(low_res, training=True)

        HR_output = new_disc(high_res)
        SR_ouput = new_disc(SR)

        # Calculating PSNR Value of generated Image
        psnr_value = PSNR(high_res, SR)
        # Discriminator Loss
        loss_disc = discriminator_loss(SR_ouput, HR_output)

        # Generator Loss
        gen_loss = generator_loss(SR_ouput)
        cont_loss = content_loss(content_model, SR, high_res)
        perceptual_loss = cont_loss + 1e-3 * gen_loss

    gen_grads = gen_tape.gradient(perceptual_loss, new_gen.trainable_variables)
    generator_optimizer.apply_gradients(zip(gen_grads, new_gen.trainable_variables))

    disc_grads = disc_tape.gradient(loss_disc, new_disc.trainable_variables)
    discriminator_optimizer.apply_gradients(zip(disc_grads, new_disc.trainable_variables))

    return perceptual_loss, loss_disc, psnr_value


for epoch in range(250):

    for lr, hr in ds.take(1000):
        perceptual_loss, loss_disc, psnr_value = train_step(lr, hr)

    if epoch % 50 == 0 or epoch == 249:
        print(f'Epochs: {epoch}  Generator_Loss:{perceptual_loss:.3f}  PSNR:{psnr_value:.3f}')