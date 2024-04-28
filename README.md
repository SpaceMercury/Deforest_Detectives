# Deforest Detectives

Welcome to the repository for the "Deforest Detectives" project, developed during the TUM Makeathon. Our mission is to leverage advanced machine learning techniques to identify and monitor deforestation activities.

## Project Overview

Deforestation is a critical environmental issue that leads to biodiversity loss, climate change, and disrupted ecosystems. Early and accurate detection of deforestation can help in taking timely action. Our project aims to address this challenge by applying a ResNet model on satellite data to detect signs of deforestation.

## Methodology

We used a Residual Neural Network (ResNet), a type of convolutional neural network that is highly effective in image recognition tasks, to analyze satellite imagery and identify deforested areas. Our choice of ResNet is motivated by its ability to handle very deep networks through skip connections and residual learning.

## Data

The dataset consists of high-resolution satellite images that have been labeled for areas of deforestation. The images being all standaradized and being taken from consistent sources, the model is able to learn the patterns of deforestation and make accurate predictions. We reduced the size to 256 to reduce the computational cost and speed up the process.

## Model Training

We trained our ResNet model on a robust dataset of labeled satellite images. The model was fine-tuned to identify deforested regions with high accuracy, ensuring it learned the complex patterns and characteristics unique to deforested landscapes.

## Results

The trained ResNet model demonstrated promising results in detecting deforestation. The model's predictions were validated against ground-truth data, showing a high level of precision and recall.

## Technologies Used

- PyTorch for implementing and training the ResNet model
- Python for data processing and model development
- Matplotlib and OpenCV for image visualization and augmentation

## Setup and Usage

### Install the dependencies

Make sure you have a running Python environment. If not please refer to the [official Python website](https://www.python.org/downloads/) to install Python on your system.

To install the required dependencies, you can run the following command:
`pip install -r requirements.txt`
This will install all the necessary packages required to run the project. It can take a little bit, don't worry, it's normal.

## Future
