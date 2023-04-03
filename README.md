# Speech-Emotion-Recognition using Convolutional Neural Networks
This repository contains the implementation of a Convolutional Neural Network (CNN) for Speech Emotion Recognition (SER) using various datasets. SER aims to identify human emotions from speech signals, which can have applications in fields such as Human-Computer Interaction, Voice Assistant Systems, and Mental Health Assessment.
## Table of contents
* [Project Description](#project-description)
* [Datasets](#datasets)
* [Feature Extraction and Data Augmentation](#feature-extraction)
* [Data Augmentation](#data-augmentation)
* [Model Architecture](#model-architecture)
* [Hyperparameter Tuning](#hyperparameter-tuning)
* [Results](#results)
* [Dependencies](#dependencies)

## Project Description
The aim of this project is to build Convolutional Neural Network for speech emotion classification.
Speech Emotion Recognition (SER) is the process of identifying human emotions and affective states from speech signals. It capitalizes on the fact that vocal cues, such as tone and pitch, often reflect a speaker's underlying emotions. This phenomenon is also observed in animals, like dogs and horses, which use vocal cues to understand human emotions.

In this project I leverage CNNs to create robust and accurate model for SER. CNNs have shown remarkable success in various pattern recognition tasks, including image and audio analysis, due to their ability to learn spatial hierarchies and capture complex features. By using a CNN, we aim to effectively capture the intricate patterns in speech signals that convey emotional information. 

By accurately recognizing emotions from speech, we can enable more natural and empathetic interactions between humans and technology, ultimately enhancing user experience and fostering better communication.

## Datasets
The following datasets have been used in this project:
* **Crowd-sourced Emotional Mutimodal Actors Dataset (Crema-D)**':
Crema-D is a dataset that contains 7,442 audio and video clips from 91 actors, who express 12 different emotions. The actors are of various ethnic backgrounds and age groups, and the dataset has a balanced gender distribution.
* **Ryerson Audio-Visual Database of Emotional Speech and Song (Ravdess)**:
Ravdess consists of 7356 audio files of various emotional expressions recorded by 24 professional actors. The dataset contains speech and song samples, with each sample being labeled with an emotion and intensity level.
* **Surrey Audio-Visual Expressed Emotion (Savee)**:
Savee is a dataset comprising 480 British English audio-visual samples from four male actors. The actors express seven different emotions, and the recordings are labeled with the corresponding emotion.
* **Toronto emotional speech set (Tess)**:
Tess consists of 200 audio files per emotion, with a total of 2800 audio files. The dataset features emotional speech recordings by two actresses who express seven different emotions.

By combining these datasets, the aim is to create a robust and diverse collection of emotional speech samples that can be effectively used for training and evaluating our emotion recognition model. This ensures a higher degree of accuracy and generalization across various accents, genders, and age groups.

## Feature Extraction
The following features are extracted from the audio data:
* **Zero Crossing Rate**: The rate of sign-changes of the signal during the duration of a particular frame.
* **Root Mean Square Energy (RMSE)**: quantifies the energy of an audio signal by calculating the square root of the average of the squared amplitude values over a given frame. 
* **Mel Frequency Cepstral Coefficients (MFCC)** form a cepstral representation where the frequency bands are not linear but distributed according to the mel-scale. 

These features are extracted using the **librosa** library, which provides a rich set of functions for audio signal processing. 
The audio data is downsampled to a lower sample rate of 8025. Downsampling reduces the computational complexity and can improve the performance of the model, especially when working with large datasets.

## Data Augmentation
Data augmentation techniques help increase the size of the dataset and improve the model's performance by adding variations to the input data. The following data augmentation techniques are applied:
* Adding noise: Random Gaussian noise is added to the input audio data.
* Shifting: The audio data is shifted along the time axis.
* Pitching: The pitch of the audio data is modified using the librosa.effects.pitch_shift function.
* Stretching: The audio data is stretched or compressed along the time axis using the librosa.effects.time_stretch function.

After applying data augmentation, the features are extracted from the augmented data, and the final feature matrix is created by stacking the features obtained from the original and augmented data. This matrix is then used to train and evaluate the model.

The dataset is divided into training, testing, and validation sets, and the extracted features are saved as CSV files for further processing.

**Note**: In many widely-used Kaggle notebooks for this problem, the authors first apply data augmentation and extract features before splitting the data into train, validation, and test sets. This approach may lead to data leakage, as the augmented data from the training set is likely to be present in the test set, resulting in test results appearing much better than they actually are.



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

## Dependencies
The following Python packages are used in this project:
1. Keras
2. Tensorflow
3. Numpy
4. Pandas
5. scikit-learn
6. seaborn
7. matplotlib
8. librosa
9. itertools
10. os
11. warnings
12. random
13. pickle
