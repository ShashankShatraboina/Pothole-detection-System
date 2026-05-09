# Pothole-Detection-System

Pothole detection and classification using ResNet101 deep learning model

---


# About the Project

### 🏠 Home Page
<p align="center">
  <img src="./Screenshot 2026-05-09 180253.png" alt="Home Page" width="800"/>
</p>

### 📸 Image 2
<p align="center">
  <img src="./Screenshot 2026-05-09 180409.png" alt="Image 2" width="800"/>
</p>

### 📸 Image 3
<p align="center">
  <img src="./Screenshot 2026-05-09 180451.png" alt="Image 3" width="800"/>
</p>

### 📸 Image 4
<p align="center">
  <img src="./Screenshot 2026-05-09 180706.png" alt="Image 4" width="800"/>
</p>

### 📸 Image 5
<p align="center">
  <img src="./Screenshot 2026-05-09 180726.png" alt="Image 5" width="800"/>
</p>
# Demo
https://github.com/ShashankShatraboina/Pothole-detection-System/raw/master/Screen%20Recording%202025-11-09%20235818.mp4
</p>

## Problem Statement

Road potholes are one of the major causes of traffic accidents, vehicle damage, and poor road safety. Manual road inspection methods are time-consuming, expensive, and inefficient for monitoring large road networks. Therefore, there is a need for an automated pothole detection system that can accurately identify potholes from road images in real time.

The goal of this project is to develop a pothole detection system using deep learning techniques that can classify road images as either containing potholes or not. The system is designed to assist smart transportation systems, road maintenance authorities, and autonomous vehicles in identifying damaged road surfaces quickly and accurately.

To achieve this, we used a dataset containing road images with potholes and normal road surfaces. The dataset was preprocessed and augmented to improve model performance and generalization. Instead of using YOLO, this project utilizes the powerful ResNet101 convolutional neural network architecture for feature extraction and image classification.

Additionally, we developed a user-friendly web application using React and Flask that allows users to upload road images and receive pothole detection predictions instantly.

By automating pothole detection, this project aims to improve road maintenance efficiency, reduce accidents, and enhance transportation safety.

---

# Machine Learning Model

In this project, we developed a deep learning-based pothole detection model using ResNet101. The dataset consisted of road images containing potholes and non-pothole surfaces. Since the dataset size was limited, we applied data augmentation techniques to increase diversity and improve the robustness of the model.

Data augmentation techniques included:

* Rotation
* Flipping
* Scaling
* Zooming
* Brightness adjustments
* Image shifting

These techniques helped the model generalize better to real-world road conditions such as varying lighting, shadows, and camera angles.

After preprocessing and augmentation, the ResNet101 model was trained on the dataset. ResNet101 is a deep residual neural network architecture that contains 101 layers and uses residual connections to solve the vanishing gradient problem. This allows the model to learn complex image features effectively while maintaining high accuracy.

During training, we used techniques such as:

* Transfer Learning
* Dropout
* Batch Normalization
* Early Stopping
* Regularization

These methods improved model performance and reduced overfitting.

After evaluation on the test dataset, the model achieved high accuracy in detecting potholes from road images. The trained model can accurately classify whether a road surface contains a pothole or not.

Finally, the trained model was saved using pickle for deployment purposes.

---

# About ResNet101 Model

ResNet101 is a deep convolutional neural network architecture introduced by Microsoft Research. It is widely used for image classification, object recognition, and computer vision applications due to its powerful feature extraction capabilities.

In the context of pothole detection, ResNet101 helps identify road damages by learning complex visual patterns from road images.

## How ResNet101 Works

### Input Layer

The model takes a road image as input. The image is converted into numerical pixel arrays before processing.

### Convolutional Layers

The image passes through multiple convolutional layers that extract features such as edges, cracks, textures, and pothole patterns.

### Residual Blocks

Unlike traditional CNNs, ResNet101 uses residual connections (skip connections) that allow information to bypass certain layers. This helps prevent vanishing gradients and enables the network to train effectively even with 101 layers.

### Pooling Layers

Pooling layers reduce spatial dimensions while preserving important image features, helping reduce computational complexity.

### Fully Connected Layers

The extracted features are flattened and passed through dense layers that perform final classification.

### Output Layer

The output layer predicts whether the uploaded image contains a pothole or not.

---

# Created Backend REST API

To make the pothole detection model accessible to users, we developed a REST API using Flask and FastAPI.

The API provides endpoints that accept road images and return pothole detection predictions.

## API Workflow

1. Load the trained ResNet101 model using pickle
2. Receive image input from the client
3. Preprocess the image using PIL/OpenCV
4. Pass the image to the trained model
5. Generate prediction results
6. Return prediction as JSON response

The backend server is capable of handling multiple concurrent requests using Gunicorn.

The API can also be deployed easily on cloud platforms such as:

* AWS
* Google Cloud
* Render
* Heroku

---

# Front-end React App

To improve usability, we created a React-based web application that allows users to upload road images and detect potholes instantly.

## Features of the Frontend

* Simple and clean UI
* Upload road images easily
* Real-time pothole prediction
* Loading animations during processing
* Responsive design for mobile and desktop
* Displays prediction confidence score

The frontend communicates with the Flask/FastAPI backend through REST API calls.

We used:

* React JS
* 
* CSS
* Bootstrap

to build an interactive and responsive user interface.

---


# Tech Stack

## Backend

* Flask
* FastAPI
* Python

## Frontend

* React JS
* CSS 

## Deep Learning

* TensorFlow
* Keras
* ResNet101
* OpenCV
* NumPy
* Pandas

---

