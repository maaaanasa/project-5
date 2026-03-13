import tensorflow as tf
import numpy as np
from model import build_generator, build_discriminator

# ===============================
# Settings
# ===============================
IMAGE_SIZE = 128
BATCH_SIZE = 16
LATENT_DIM = 100
EPOCHS = 150

# ===============================
# Load Dataset
# ===============================
dataset = tf.keras.preprocessing.image_dataset_from_directory(
    "dataset",
    image_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE
)

# Normalize images (-1 to 1)
dataset = dataset.map(lambda x, y: ((x / 127.5) - 1, y))

# IMPORTANT FIX
dataset = dataset.unbatch()
dataset = dataset.shuffle(1000)
dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)

# ===============================
# Create Models
# ===============================
generator = build_generator()
discriminator = build_discriminator()

# ===============================
# Optimizers
# ===============================
gen_optimizer = tf.keras.optimizers.Adam(0.0002)
disc_optimizer = tf.keras.optimizers.Adam(0.0002)

# ===============================
# Loss Functions
# ===============================
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    return real_loss + fake_loss

# ===============================
# Training Step
# ===============================
@tf.function
def train_step(images, labels):

    noise = tf.random.normal([BATCH_SIZE, LATENT_DIM])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:

        generated_images = generator([noise, labels], training=True)

        real_output = discriminator([images, labels], training=True)
        fake_output = discriminator([generated_images, labels], training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_gen = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_disc = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    gen_optimizer.apply_gradients(zip(gradients_gen, generator.trainable_variables))
    disc_optimizer.apply_gradients(zip(gradients_disc, discriminator.trainable_variables))

# ===============================
# Training Loop
# ===============================
def train(dataset, epochs):

    for epoch in range(epochs):
        print("Epoch:", epoch + 1)

        for image_batch, label_batch in dataset:
            train_step(image_batch, label_batch)

    generator.save("generator_model.h5")
    print("Model Saved!")

# ===============================
# Start Training
# ===============================
train(dataset, EPOCHS)