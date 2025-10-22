Advanced Transformers for Stock Price Prediction
This project demonstrates the application of advanced Transformer architectures to time series prediction. Using synthetic stock price data, I built a Transformer Encoder from scratch and trained it to predict future stock prices. This project helped me understand the inner workings of Transformers, Multi-Head Self-Attention, and model integration in TensorFlow.


How to Run
git clone  https://github.com/Raghda13/Deep-learning
cd advanced_transformers.py

2)pip install numpy pandas tensorflow scikit-learn matplotlib
3)Run the script 
python transformer_stock_prediction.py


2)his project explores how deep learning models are trained under the hood using TensorFlow.
Instead of relying only on model.fit(), I implemented a custom training loop that manually controls:

The forward pass
Loss computation
Gradient calculation using tf.GradientTape()
Weight updates via the optimizer

3)final_project2
EcoClean: Automated Waste Classification
Project Overview

EcoClean currently lacks an efficient and scalable method to automate the waste sorting process. Manual sorting of waste is labor-intensive and prone to errors, leading to contamination of recyclable materials. This project leverages machine learning and computer vision to automate the classification of waste products, improving efficiency and reducing contamination rates.

The project uses transfer learning with a pre-trained VGG16 model to classify images of waste as either recyclable or organic.

#Aim of the Project

The aim of this project is to develop an automated waste classification model that can accurately differentiate between recyclable and organic waste based on images. By the end of this project, you will have:

Trained and fine-tuned a model using transfer learning

Evaluated the model’s performance

Produced a trained model ready for real-world waste management applications

Final Output: A trained model that classifies waste images into recyclable and organic categories.

#Dataset

The dataset contains images of recyclable (R) and organic (O) waste, split into training and testing directories. It is automatically downloaded and extracted by the provided script:

url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/kd6057VPpABQ2FqCbgu9YQ/o-vs-r-split-reduced-1200.zip"
file_name = "o-vs-r-split-reduced-1200.zip"

How to Run the Project

Clone the repository or download the files.

Install dependencies (e.g., using pip):

pip install tensorflow matplotlib scikit-learn tqdm requests


#Run the script to:

Download and extract the dataset
Train the model (feature extraction and fine-tuning)
Save the trained models (O_R_tlearn_vgg16.keras and O_R_tlearn_fine_tune_vgg16.keras)
Evaluate the models on the test set
Plot loss, accuracy, and sample predictions
Inspect the outputs:
Training and validation loss/accuracy plots
Classification reports
Example predictions with actual and predicted labels



 
