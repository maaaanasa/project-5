import streamlit as st
import tensorflow as tf
import numpy as np

LATENT_DIM = 100

st.title("Surface Defect Generator using GAN")

st.write("Generate synthetic surface defect images")

# Load model
try:
    generator = tf.keras.models.load_model("generator_model.h5")
    st.success("Generator model loaded")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

classes = ["dot", "good", "joint"]

selected_class = st.selectbox("Select Defect Type", classes)

num_images = st.slider("Number of Images", 1, 9, 4)

if st.button("Generate Images"):

    st.write("Generating images...")

    label_index = classes.index(selected_class)

    noise = np.random.normal(0,1,(num_images,LATENT_DIM))
    labels = np.full((num_images,1),label_index)

    st.write("Noise shape:", noise.shape)
    st.write("Labels shape:", labels.shape)

    try:
        generated_images = generator.predict([noise,labels])
    except Exception as e:
        st.error(f"Prediction error: {e}")
        st.stop()

    generated_images = (generated_images + 1) / 2

    st.subheader("Generated Images")

    for i in range(num_images):
        st.image(generated_images[i], caption=f"Image {i+1}")