import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

LATENT_DIM = 100

generator = tf.keras.models.load_model("generator_model.h5")

classes = ["dot","good","joint"]

label = 0

noise = np.random.normal(0,1,(9,LATENT_DIM))
labels = np.full((9,1),label)

generated_images = generator.predict([noise,labels])

generated_images = (generated_images + 1) / 2

fig, ax = plt.subplots(3,3,figsize=(6,6))

count = 0
for i in range(3):
    for j in range(3):
        ax[i,j].imshow(generated_images[count])
        ax[i,j].axis("off")
        count += 1

plt.show()