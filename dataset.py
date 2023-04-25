import tensorflow as tf
import tensorflow_datasets as tfds

AUTOTUNE = tf.data.AUTOTUNE


def flip_left_right(lr_img, hr_img):
    rn = tf.random.uniform(shape=(), maxval=1)
    return tf.cond(
        rn < 0.5,
        lambda: (lr_img, hr_img),
        lambda: (tf.image.flip_left_right(lr_img), tf.image.flip_left_right(hr_img)),
    )


def random_rotate(lr_img, hr_img):
    rn = tf.random.uniform(shape=(), maxval=4, dtype=tf.int32)
    return tf.image.rot90(lr_img, rn), tf.image.rot90(hr_img, rn)


def random_crop(lr_img, hr_img, hr_crop_size=96, scale=4):
    lr_crop_size = hr_crop_size // scale
    lr_img_shape = tf.shape(lr_img)[:2]

    lr_width = tf.random.uniform(
        shape=(), maxval=lr_img_shape[1] - lr_crop_size + 1, dtype=tf.int32
    )
    lr_height = tf.random.uniform(
        shape=(), maxval=lr_img_shape[0] - lr_crop_size + 1, dtype=tf.int32
    )

    hr_width = lr_width * scale
    hr_height = lr_height * scale

    lr_img_cropped = lr_img[
        lr_height : lr_height + lr_crop_size, lr_width : lr_width + lr_crop_size
    ]
    hr_img_cropped = hr_img[
        hr_height : hr_height + hr_crop_size, hr_width : hr_width + hr_crop_size
    ]

    return lr_img_cropped, hr_img_cropped


def preprocessing(_cache):
    ds = _cache

    ds = ds.map(
        lambda lr, hr: random_crop(lr, hr, scale=4), num_parallel_calls=AUTOTUNE
    )
    ds = ds.map(random_rotate, num_parallel_calls=AUTOTUNE)
    ds = ds.map(flip_left_right, num_parallel_calls=AUTOTUNE)

    ds = ds.shuffle(buffer_size=500)
    ds = ds.batch(16)
    ds = ds.repeat(None)
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds


def download_data():
    div2k_data = tfds.image.Div2k(config="bicubic_x4")
    div2k_data.download_and_prepare()

    train = div2k_data.as_dataset(split="train", as_supervised=True)
    train_cache = train.cache()

    val = div2k_data.as_dataset(split="validation", as_supervised=True)
    val_cache = val.cache()

    train_ds = preprocessing(train_cache)
    val_ds = preprocessing(val_cache)
    return train_ds, val_ds
