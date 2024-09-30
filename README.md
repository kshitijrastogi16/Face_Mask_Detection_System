# Face Mask Detection

## Overview
This project is focused on developing a Face Mask Detection System using deep learning and computer vision techniques. The system is designed to classify whether a person in an image is wearing a mask or not, helping to automate mask compliance in public spaces during the COVID-19 pandemic.
Face Mask Detection has become an important task in maintaining health safety standards during the COVID-19 pandemic. 
This project uses deep learning techniques to classify images into two categories:
### Mask: Person is wearing a mask.
### No Mask: Person is not wearing a mask.
This project builds a robust deep learning model capable of achieving high accuracy by leveraging image classification methods. It can be used in real-time applications to automatically detect mask compliance in areas like airports, train stations, offices, and more.

## Features
### Image Preprocessing: The system uses common computer vision techniques to preprocess input images (resizing, normalization, and augmentation).
### Binary Classification: It classifies images into two categories—mask or no mask.
### Real-Time Detection: The system is capable of detecting mask-wearing compliance in real-time using a webcam feed or any input camera.
### Transfer Learning: The model uses a pre-trained Convolutional Neural Network (CNN) to enhance the accuracy of detection with a smaller training set.

## Technologies Used
### Python: Main programming language
### TensorFlow/Keras: For building and training the deep learning models
### OpenCV: For real-time image processing and video capture
### Matplotlib & Seaborn: For data visualization
### Numpy & Pandas: For data manipulation and analysis
### scikit-learn: For evaluating model performance

## Dataset
The dataset used for this project contains images of people both wearing and not wearing masks. It is divided into two classes:

### With Mask: Images of people wearing face masks.
### Without Mask: Images of people not wearing face masks.

## Dataset summary:
### Total Images: 3,000+ images
### Classes: Mask, No Mask
The dataset is sourced from publicly available repositories, such as Kaggle, or can be custom-collected. Images are preprocessed before being used for training.

## Model Architecture
This project uses a pre-trained MobileNetV2 model as the base for transfer learning. MobileNetV2 is lightweight and efficient, making it suitable for real-time applications. The architecture is fine-tuned for the specific task of mask detection by replacing the top layers to fit the binary classification task (mask/no mask).

### Model Pipeline:
#### Image Preprocessing: Resizing the image to 224x224 and normalizing pixel values.
#### Base Model: MobileNetV2 pre-trained on ImageNet is used for feature extraction.
#### Custom Classifier: The output from MobileNetV2 is fed into a fully connected layer for classification.

### Training:
#### Optimizer: Adam
#### Loss Function: Binary Crossentropy
#### Epochs: 20 (can be adjusted)
#### Batch Size: 32

### Evaluation Metrics:
#### Accuracy
#### Precision
#### Recall
#### F1-Score

### Dependencies include:
#### TensorFlow
#### OpenCV
#### Matplotlib
#### scikit-learn
#### Numpy
#### Pandas

## Usage
### Jupyter Notebook:
#### Open the Face Mask Detection.ipynb notebook.
#### Follow the steps for data preprocessing, model training, and evaluation.
#### You can also use the notebook for testing individual images for mask detection.

### Real-Time Detection:
#### You can use a webcam feed to detect face masks in real-time.
#### To run the real-time detection, execute the following command (ensure you have a webcam attached):

## Results
The model has been evaluated on the test dataset, and it achieves the following performance metrics:
#### Accuracy: 98.5%
#### Precision: 97.9%
#### Recall: 98.7%
#### F1-Score: 98.3%
The confusion matrix and ROC curves indicate strong performance across both classes (Mask and No Mask).

## Model Evaluation
The trained model is evaluated using various metrics, including:
#### Confusion Matrix: Visualizing the number of true positives, false positives, true negatives, and false negatives.
#### ROC Curve: To analyze the trade-off between sensitivity and specificity.
#### Classification Report: For detailed insights into the model's precision, recall, and F1-score.
