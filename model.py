import tensorflow as tf
from tensorflow.keras import layers

LATENT_DIM = 100
NUM_CLASSES = 3

def build_generator():

    noise = layers.Input(shape=(LATENT_DIM,))
    label = layers.Input(shape=(1,))

    label_embedding = layers.Embedding(NUM_CLASSES, LATENT_DIM)(label)
    label_embedding = layers.Flatten()(label_embedding)

    model_input = layers.multiply([noise, label_embedding])

    x = layers.Dense(16*16*256, use_bias=False)(model_input)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)

    x = layers.Reshape((16,16,256))(x)

    x = layers.Conv2DTranspose(128,4,strides=2,padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)

    x = layers.Conv2DTranspose(64,4,strides=2,padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)

    x = layers.Conv2DTranspose(32,4,strides=2,padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)

    output = layers.Conv2DTranspose(3,3,strides=1,padding="same",activation="tanh")(x)

    model = tf.keras.Model([noise,label],output)

    return model


def build_discriminator():

    image = layers.Input(shape=(128,128,3))
    label = layers.Input(shape=(1,))

    label_embedding = layers.Embedding(NUM_CLASSES,128*128*3)(label)
    label_embedding = layers.Reshape((128,128,3))(label_embedding)

    x = layers.Concatenate()([image,label_embedding])

    x = layers.Conv2D(64,4,strides=2,padding="same")(x)
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Conv2D(128,4,strides=2,padding="same")(x)
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Conv2D(256,4,strides=2,padding="same")(x)
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Flatten()(x)

    output = layers.Dense(1)(x)

    model = tf.keras.Model([image,label],output)

    return model