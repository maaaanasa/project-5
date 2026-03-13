import tensorflow as tf
import matplotlib.pyplot as plt

IMG_SIZE = 128
BATCH_SIZE = 32

DATASET_PATH = "dataset"

# Load dataset
dataset = tf.keras.preprocessing.image_dataset_from_directory(
    DATASET_PATH,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE
)

# Save class names BEFORE mapping
class_names = dataset.class_names

# Normalize images
dataset = dataset.map(lambda x, y: (x / 255.0, y))

print("Classes:", class_names)

# Display sample images
plt.figure(figsize=(8,8))

for images, labels in dataset.take(1):
    for i in range(9):
        ax = plt.subplot(3,3,i+1)
        plt.imshow(images[i])
        plt.title(class_names[labels[i]])
        plt.axis("off")

plt.show()