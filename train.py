import tensorflow as tf
from models.utils import PSNR
from loss import discriminator_loss, generator_loss, content_loss, Content_Net

class train_srgan:
    def __init__(self, generator, Discriminator):
        self.content_model = Content_Net()
        self.content_loss = content_loss
        self.generator_loss = generator_loss
        self.discriminator_loss = discriminator_loss
        self.PSNR = PSNR
        self.generator = generator
        self.discriminator = Discriminator
        self.generator_optimizer = tf.keras.optimizers.Adam(learning_rate=tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries=[10000], values=[1e-4, 1e-5]))
        self.discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries=[10000], values=[1e-4, 1e-5]))

    @tf.function
    def train_step(self, low_res, high_res):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            low_res = tf.cast(low_res, tf.float32)
            high_res = tf.cast(high_res, tf.float32)

            SR = self.generator(low_res, training=True)

            HR_output = self.discriminator(high_res)
            SR_ouput = self.discriminator(SR)

            # Calculating PSNR Value of generated Image
            psnr_value = self.PSNR(high_res, SR)
            # Discriminator Loss
            loss_disc = self.discriminator_loss(SR_ouput, HR_output)

            # Generator Loss
            gen_loss = self.generator_loss(SR_ouput)
            cont_loss = self.content_loss(self.content_model, SR, high_res)
            perceptual_loss = cont_loss + 1e-3 * gen_loss

        gen_grads = gen_tape.gradient(perceptual_loss, self.generator.trainable_variables)
        self.generator_optimizer.apply_gradients(zip(gen_grads, self.generator.trainable_variables))

        disc_grads = disc_tape.gradient(loss_disc, self.discriminator.trainable_variables)
        self.discriminator_optimizer.apply_gradients(zip(disc_grads, self.discriminator.trainable_variables))

        return perceptual_loss, loss_disc, psnr_value
    
class train_edsr_srresnet:
    def __init__(self, model):
        self.model = model
        self.loss_fn = tf.keras.losses.MeanAbsoluteError()
        
        self.optim = tf.keras.optimizers.Adam(learning_rate=tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries=[10000], values=[1e-4, 1e-5]))

    @tf.function
    def train_step(self, low_res, high_res):
        with tf.GradientTape() as tape:
            low_res = tf.cast(low_res, tf.float32)
            high_res = tf.cast(high_res, tf.float32)

            sr = self.model(low_res)
            loss_value = self.loss_fn(high_res, sr)

            # Calculating PSNR value
            psnr_value = PSNR(high_res, sr)
        
        gradients = tape.gradient(loss_value, self.model.trainable_variables)
        self.optim.apply_gradients(zip(gradients, self.model.trainable_variables))

        return loss_value, psnr_value
