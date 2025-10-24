# Project 1
Aircraft Damage Classification and Captioning
# Project Overview

his is a complete end-to-end deep learning project that combines computer vision (VGG16 feature extraction) with transformer-based image captioning (BLIP).
The goal of this project is to develop an automated deep learning model that can accurately classify aircraft damage from images and generate descriptive captions or summaries of that damage.

# Environment Setup
pip install tensorflow keras torch transformers pillow matplotlib numpy

# Dataset



# Model 1 :
Aircraft Damage Classifier

Architecture
The model uses VGG16 (pretrained on ImageNet) for feature extraction, followed by fully connected layers for classification

#Training Results :
The model typically achieves high validation accuracy after several epochs (≈90–95%).
Training and validation curves are plotted using Matplotlib for visual analysis

# Model 2 :Transformer for Image Captioning
Model Used
Pretrained BLIP (Bootstrapped Language-Image Pretraining) model

# How to Run
1) Open the notebook in Google Colab or Jupyter
2) Run all cells sequentially (dataset download happens automatically).
3) Wait for model training to complete.
4) Evaluate accuracy and visualize training results
5) Run the BLIP section to generate captions and summaries for sample images.


# Project 2
EcoClean: Automated Waste Classification using VGG16
# Project Overview
EcoClean currently lacks an efficient and scalable method to automate the waste sorting process. Manual sorting of waste is labor-intensive and error-prone, often leading to contamination of recyclable materials.

This project leverages deep learning and computer vision to automate waste classification — distinguishing between organic (O) and recyclable (R) waste using transfer learning with a pre-trained VGG16 model.

# Aim of the Project
The aim of this project is to develop an automated image classification model capable of accurately differentiating recyclable and organic waste

# Dataset
The dataset used is publicly available and can be accessed here:
📂 O vs R Dataset (Recyclable vs Organic Waste)

It contains two classes:
O → Organic Waste
R → Recyclable Waste

# Technologies Used

TensorFlow / Keras
VGG16 (Transfer Learning)
NumPy, Matplotlib
scikit-learn (for metrics)
Google Colab

# How to Run 
1)Open the notebook in Google Colab.
2)Final Project Notebook
3)Run all cells sequentially — the dataset will be downloaded and extracted automatically.

The model will train, evaluate, and generate plots and reports.


# Advanced Transformers
# Overview
This project implements a Transformer-based model from scratch using TensorFlow and Keras to predict synthetic stock prices.
It demonstrates the use of multi-head self-attention and Transformer encoder blocks for capturing temporal dependencies in time series data.

# Dataset

Type: Synthetic time series data
Generated: Using a simple trend + noise formula
File: stock_prices.csv
Features: One feature — “Close” price
Length: 2000 time steps

# Model Architecture
Main Components:

MultiHeadSelfAttention Layer: Implements scaled dot-product attention across multiple heads.
TransformerBlock: Combines attention, feed-forward layers, normalization, and dropout.
TransformerEncoder: Stacks multiple Transformer blocks for deeper learning.

Overall Structure:
Input sequence projected into embedding space
Multiple Transformer Encoder layers process sequence
Flattened and passed through a dense output layer for prediction

# How to Run
pip install tensorflow numpy pandas scikit-learn matplotlib
Then run the notebook in Google Colab or locally
python Advanced_Transformers.ipynb




# Autoencoders
 #Overview
 This project demonstrates how to build and train an Autoencoder using the MNIST dataset of handwritten digits.
An Autoencoder is a type of neural network used for unsupervised learning, where the model learns to compress (encode) input data into a smaller representation and then reconstruct (decode) it back to the original form.

# Dataset

Dataset: MNIST Handwritten Digits
Description: 70,000 grayscale images (28×28 pixels) of digits 0–9
Usage: The dataset is automatically loaded using tensorflow.keras.datasets.mnist


# How to Run
 1)You can run this notebook directly in Google Colab or locally
pip install tensorflow numpy matplotlib
2) Then execute the notebook
python Autoencoders.ipynb


# Custom Training Loop in Keras
This project demonstrates how to build and train neural networks using custom training loops in TensorFlow and Keras.
It walks through two main parts:
Implementing a manual training loop on the MNIST dataset.
Building a simple binary classification model using standard model.fit() training.

The goal is to understand how Keras models can be trained both manually (step-by-step control) and automatically (using built-in training)
# Enironment
pip install tensorflow numpy

# Dataset
Dataset
MNIST Dataset
https://www.tensorflow.org/datasets/catalog/mnist
Description: Handwritten digits (0–9) dataset with 60,000 training and 10,000 test images (28×28 grayscale).

# Requirements
TensorFlow
NumPy

# How to Run
1)Open in Google Colab or Jupyter Notebook
2)Run all cells sequentially.
3)Observe printed loss and accuracy logs per step and epoch
4)Compare manual vs automatic training behaviors.



# Author
Raghda Elsakka
Machine Learning Engineer
LinkedIn:https://www.linkedin.com/in/raghda-elsakka-463541202/
 




