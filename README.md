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



# Author
Raghda Elsakka
Machine Learning Engineer
LinkedIn:https://www.linkedin.com/in/raghda-elsakka-463541202/
 




