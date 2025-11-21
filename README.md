# Fingerprint Blood Group Prediction (ResNet-18)

This project provides a non-invasive system to predict a person’s ABO/Rh blood group from their fingerprint image using deep learning. The solution combines advanced CNN modeling, non-invasive data acquisition, and a practical Flask-based GUI for end-users, offering speed and usability for scenarios where rapid pre-screening is required.

-> Project Overview & Methodology

## Abstract

The system uses a Convolutional Neural Network (CNN), based on a modified ResNet-18 architecture, trained on fingerprint datasets to classify each image into one of the common blood group classes (A+, A-, AB+, AB-, B+, B-, O+, O-). A user-friendly GUI enables real-time blood group prediction by uploading a fingerprint image. The system consistently achieves around 90% classification accuracy in testing, indicating its potential for biometric pre-screening in healthcare and related domains.

## Objectives

Develop a pipeline that accepts fingerprint images and predicts the corresponding blood group.

Achieve at least 90% prediction accuracy on a representative fingerprint dataset.

Package the model within a Python GUI for interactive, real-time, end-user deployment (implemented using Flask for the web UI).

Provide detailed evaluation metrics (classification report and confusion matrix) to ensure model accountability.

## Model Selection and Dataflow

Choice of Model: A ResNet-18 CNN, pretrained on ImageNet, was chosen due to its proven ability to extract hierarchical features from images, even with modest dataset sizes.

Modification: The final fully connected layer was replaced to match the number of target blood group classes (8), allowing direct classification.

Feature Extraction: The system uses automated, CNN-based convolutional layers to extract unique fingerprint features (ridges/minutiae) that are implicitly correlated to blood group during supervised training.

## Dataflow Diagram:

Input (Fingerprint Image via GUI) $\rightarrow$ Preprocessing (Resize $\rightarrow$ Grayscale $\rightarrow$ Normalize) $\rightarrow$ Modified ResNet-18 CNN $\rightarrow$ Prediction Output (Predicted Group & Confidence)

-> Running the Application Locally (Flask UI)

To run the web interface locally, you must have Python and Git installed.

1. Clone the Repository

Since you already have the files locally, you can skip this step, but for anyone else:
git clone [https://github.com/aditiiiii1/fingerprint-blood-group-prediction.git](https://github.com/aditiiiii1/fingerprint-blood-group-prediction.git)
cd fingerprint-blood-group-prediction


2. Setup the Environment

Create and activate a virtual environment to manage dependencies.

i. Create the environment
python3 -m venv env_flask

ii. Activate the environment
### On macOS/Linux:
source env_flask/bin/activate
### On Windows (Command Prompt/PowerShell):
.\env_flask\Scripts\activate


3. Install Dependencies

Install all required libraries (PyTorch, Flask, etc.) using pip.
pip install Flask Pillow torch torchvision requests scikit-learn matplotlib numpy


4. Run the Flask Server

The application runs using the app.py file located inside the blood_group folder.
### Set the Flask application file
export FLASK_APP=blood_group/app.py
### Start the server
flask run


The application will launch on your local machine, typically accessible at: http://127.0.0.1:5000/

-> Performance and Evaluation

Accuracy: The model achieves approximately 90% accuracy on the test dataset.

Metrics: Detailed evaluation metrics include Precision, Recall, and F1-Score (all classes typically $> 88\%$).

Evaluation Tools: scikit-learn and matplotlib are used to generate the classification report and confusion matrix for detailed error analysis.

