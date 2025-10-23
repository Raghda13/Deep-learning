#final_project2
EcoClean: Automated Waste Classification

#Project Overview
EcoClean currently lacks an efficient and scalable method to automate the waste sorting process. Manual sorting of waste is labor-intensive and prone to errors, leading to contamination of recyclable materials. This project leverages machine learning and computer vision to automate the classification of waste products, improving efficiency and reducing contamination rates.

The project uses transfer learning with a pre-trained VGG16 model to classify images of waste as either recyclable or organic.

#Aim of the Project

The aim of this project is to develop an automated waste classification model that can accurately differentiate between recyclable and organic waste based on images. By the end of this project, you will have:

_Trained and fine-tuned a model using transfer learning
_Evaluated the model’s performance
_Produced a trained model ready for real-world waste management applications

_ #How to Run the Project
1)Clone the repository or download the files.
2)Install dependencies" e.g., using pip"
 pip install tensorflow matplotlib scikit-learn tqdm requests
3)Run the script to:
Download and extract the dataset
Train the model (feature extraction and fine-tuning)
Save the trained models (O_R_tlearn_vgg16.keras and O_R_tlearn_fine_tune_vgg16.keras)
Evaluate the models on the test set
Plot loss, accuracy, and sample predictions

#Technologies Used
Python 3.x
TensorFlow / Keras
NumPy
Matplotlib
scikit-learn


2)Autoencoder on MNIST Dataset

This repository contains a simple implementation of an autoencoder using TensorFlow/Keras to compress and reconstruct images from the MNIST dataset.
Autoencoders are neural networks that learn to compress input data into a smaller representation (encoding) and then reconstruct the original data from this representation (decoding). This project demonstrates a basic fully connected autoencoder.

#Overview

The autoencoder in this project consists of:
Encoder: Compresses the 784-dimensional MNIST images into a 32-dimensional bottleneck representation.
Decoder: Reconstructs the original image from the bottleneck.
Training: Uses the binary crossentropy loss function and Adam optimizer.

#Requirements

Python 3.x
TensorFlow 2.x
NumPy

#Setup

Clone the repository
cd autoencoder-mnist
Run the notebook in Google Colab or Jupyter Notebook:
jupyter notebook Autoencoders.ipynb

3)Stock Price Prediction with Transformer Encoder
This repository contains an advanced transformer-based model for time series forecasting applied to synthetic stock prices. The model leverages multi-head self-attention and transformer encoder layers to predict stock prices based on past values.
#Overview

The project demonstrates:
Data Preparation:
Synthetic stock price generation with trend + noise
Normalization with MinMaxScaler
Time-series windowing for supervised learning
Transformer Components:
Multi-Head Self-Attention (MultiHeadSelfAttention)
Transformer Block (TransformerBlock)
Transformer Encoder with multiple layers (TransformerEncoder)
Model Training:
Input projection to embedding dimension
Flattening encoder outputs
Dense layer for regression output
The model is trained using mean squared error (MSE) to predict future stock prices.

#Requirements

Python 
TensorFlow 
NumPy
Pandas
scikit-learn
Matplotlib

#Setup
Clone the repository:
git clone https://github.com/your-username/transformer-stock-prediction.git
cd transformer-stock-prediction
Run the notebook in Google Colab or Jupyter Notebook:
jupyter notebook Advanced\ Transformers.ipynb


4)Custom Training Loop in Keras
This project demonstrates how to build and train deep learning models using custom training loops in TensorFlow/Keras, giving full control over the forward and backward passes, gradient updates, and logging.

The notebook also includes a simple binary classification example using the standard .fit() API for comparison.
#Overview

This project covers two main parts:
 Part 1: Custom Training Loop on MNIST
Loads and preprocesses the MNIST handwritten digits dataset.
Defines a simple fully connected neural network.
Implements a manual training loop using:
tf.GradientTape() for gradient computation
Manual accuracy tracking
Custom callback for logging after each epoch

 Part 2: Binary Classification Example
Demonstrates a traditional Keras training setup using .compile() and .fit().
Uses a simple model trained on synthetic binary data to show a typical workflow.

#Setup

1)Clone this repository:
3)cd custom-training-loop-keras
4)Run the notebook in Google Colab or Jupyter Notebook:
5)jupyter notebook "custom training loop in keras.ipynb"
