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
