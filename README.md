# Surface Defect Generation using Conditional GAN

This project implements a Conditional Generative Adversarial Network (cGAN) to generate synthetic surface defect images.

The goal is to learn patterns from a dataset of industrial surface images and generate new synthetic defect samples. These generated images can be used for data augmentation and training defect detection systems.

The project provides a complete pipeline including:

Data preprocessing

GAN model architecture

Model training

Synthetic image generation

Streamlit web interface

Module 1 — Data Pipeline

Purpose:

Collects and preprocesses real surface images (normal + defected) for training.

Steps / Actions:

Reads images from dataset folder.

Resizes/crops images to uniform size (128x128).

Normalizes pixel values for training (0-1).

Splits data into train/test sets.

Libraries / Tools:

Python, NumPy, OpenCV (cv2), os

Output:

Preprocessed dataset ready for cGAN training.

Module 2 — cGAN Architecture

Purpose:

Implements Conditional Generative Adversarial Network to generate defect images.

Components:

Generator:

Takes random noise + condition (defect type).

Generates synthetic defect images.

Discriminator:

Classifies images as real vs generated (fake).

Uses label information for conditional learning.

Libraries / Tools:

TensorFlow / Keras, NumPy

Output:

Model ready to generate defect images.

Module 3 — Training Engine

Purpose:

Trains cGAN using real + generated images.

Steps / Actions:

Defines loss functions for Generator & Discriminator.

Uses Adam optimizer for training.

Trains for 50+ epochs with batch size 64.

Saves trained generator/discriminator models.

Libraries / Tools:

TensorFlow, Keras, Matplotlib (for loss plots)

Output:

Trained generator & discriminator models saved (generator_model.h5).

Module 4 — Evaluation

Split into submodules:

4A — Visual Image Check

Displays generated images visually for sanity check.

4B — Statistical Image Metrics

Computes metrics like PSNR, SSIM to quantify image quality.

4C — QA / Engineer Review (Optional)

Human inspection to verify defect realism.

4D — Downstream Classifier Impact

Uses generated images to augment training data for defect classifier.

Checks if classifier performance improves with generated data.

Libraries / Tools:

Matplotlib, scikit-image, NumPy

Module 5 — Deployment Layer (UI + API)

Purpose:

Provides a user interface to generate images using trained generator.

Optional API for automation.

Steps / Actions:

Streamlit interface for user input (defect type).

Generates image on click using trained model.

Saves or displays generated images.

Libraries / Tools:

Streamlit, TensorFlow/Keras

Output:

Interactive UI for synthetic defect image generation.

Module 6 — Monitoring & Updates

Purpose:

Monitors model performance after deployment.

Updates models with new data if needed.

Steps / Actions:

Logs generated images + usage statistics.

Optional retraining pipeline if model drift detected.

Libraries / Tools:

Python logging, NumPy, Matplotlib

## Streamlit UI
![Streamlit UI](https://raw.githubusercontent.com/maaaanasa/project-5/main/images/Streamlit%20UI.png)

## Generated Images 1
![Generated Images](https://raw.githubusercontent.com/maaaanasa/project-5/main/images/generated%20images.png)

# Project Title
## Module 1 - Data Preprocessing
### Step 1: Load Dataset
### Step 2: Clean Data
## Module 2 - Model Training
### Generator Network
### Discriminator Network

## Generated Images 2
![Generated Images 2](https://raw.githubusercontent.com/maaaanasa/project-5/main/images/generated%20images-2.png)

