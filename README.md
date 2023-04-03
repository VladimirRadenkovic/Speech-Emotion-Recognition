# Speech-Emotion-Recognition using Convolutional Neural Networks
This repository contains the implementation of a Convolutional Neural Network (CNN) for Speech Emotion Recognition (SER) using various datasets. SER aims to identify human emotions from speech signals, which can have applications in fields such as Human-Computer Interaction, Voice Assistant Systems, and Mental Health Assessment.
## Table of contents
* [Project Description](#project-description)
* [Feature Extraction and Data Augmentation](#feature-extraction)
* [Data Augmentation](#data-augmentation)
* [Model Architecture](#model-architecture)
* [Hyperparameter Tuning](#hyperparameter-tuning)
* [Results](#results)

## Project Description
The aim of this project is to build Convolutional Neural Network for speech emotion classification.
Speech Emotion Recognition, abbreviated as SER, is the act of attempting to recognize human emotion and affective states from speech. This is capitalizing on the fact that voice often reflects underlying emotion through tone and pitch. This is also the phenomenon that animals like dogs and horses employ to be able to understand human emotion. 
Datasets used in this project are:
* Crowd-sourced Emotional Mutimodal Actors Dataset (Crema-D)
* Ryerson Audio-Visual Database of Emotional Speech and Song (Ravdess)
* Surrey Audio-Visual Expressed Emotion (Savee)
* Toronto emotional speech set (Tess)

##Feature Extraction and Data Augmentation


## Model Architecture
The model architecture used in this project is a 1D Convolutional Neural Network (CNN) built using Keras. The architecture is composed of several layers, including convolutional layers, batch normalization, dropout, max-pooling, and fully connected layers. The model is designed to take input data of shape (880, 1), and output probabilities for each of the 7 emotion classes using the softmax activation function.

The following is a brief description of the layers in the model:
1. **Conv1D Layer**: This is the first convolutional layer with 256 filters, a kernel size of 5, a stride of 1, and padding set to "same". It uses the ReLU activation function.
2. **Batch Normalization**: This layer normalizes the activations of the previous layer for faster and more stable training.
3. **Dropout**": This layer randomly sets a fraction of input units to 0, in this case, 20% for convolutional layers and 50% for the output layer, at each update during training to prevent overfitting.
4. **MaxPool1D**: This layer reduces the spatial dimensions of the input by taking the maximum value within a pool size of 5 and a stride of 2.

The above four layers are repeated several times with varying filter sizes, kernel sizes, and strides to extract different levels of features from the input data. The subsequent layers have 512, 256, 256, and 128 filters, respectively.

After the final max-pooling layer, the model is flattened and connected to a fully connected layer (Dense) with 256 neurons and a ReLU activation function. This is followed by another batch normalization and dropout layer, with a dropout rate of 50%.

The output layer is a fully connected (Dense) layer with 7 neurons, corresponding to the 7 emotion classes. The softmax activation function is used to produce probability values for each class.

The model is compiled with the Adam optimizer, categorical cross-entropy loss, and the following performance metrics: accuracy, precision, recall, and F1-score. Custom functions recall_m, precision_m, and f1_m are defined to calculate these metrics during training.

Two callback functions, EarlyStopping and ReduceLROnPlateau, are used during the training process to prevent overfitting and optimize the learning rate, respectively.
