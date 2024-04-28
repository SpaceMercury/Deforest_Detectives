# Deforest Detectives - OnlyForests

Welcome to the repository for the "Deforest Detectives" project, developed during the TUM Makeathon. Our mission is to leverage advanced machine learning techniques to identify and monitor deforestation activities.

## Project Overview

Deforestation is a critical environmental issue that leads to biodiversity loss, climate change, and disrupted ecosystems. Early and accurate detection of deforestation can help in taking timely action. Our project aims to address this challenge by applying a ResNet model on satellite data to detect signs of deforestation. Thanks to that detection we will calculate a "vulnerability" or "risk" score associated with a piece of land.

## Methodology

Initially we used a Residual Neural Network (ResNet), a type of convolutional neural network that is highly effective in image recognition tasks, to analyze satellite imagery and identify deforested areas. But after tinkering and testing we achieved better results with UNET, which is the technology that will be used in the app. We chose the U-Net architecture for our project because of its proven effectiveness in the field of image segmentation.

## Data

The dataset consists of high-resolution satellite images that have been labeled for areas of deforestation. The images being all standaradized and being taken from consistent sources, the model is able to learn the patterns of deforestation and make accurate predictions. We reduced the size to 256 to reduce the computational cost and speed up the process.
IMPORTANT: The dataset is not included in this repository, you need to download it and place it in the data folder.
The dataset is too large to be uploaded to this repository, but you can find it [here](https://www.kaggle.com/datasets/balraj98/deepglobe-land-cover-classification-dataset).

## Results

The trained ResNet model demonstrated promising results in detecting deforestation. Unet achieved better results and is the final part of the demo.

## Technologies Used

- PyTorch for implementing the ResNet model
- PyTorch for implementing Unet
- Python for data processing and model development
- Matplotlib and OpenCV for image visualization and augmentation

## Setup and Usage

### Install the dependencies

Make sure you have a running Python environment. If not please refer to the [official Python website](https://www.python.org/downloads/) to install Python on your system.

To install the required dependencies, you can run the following command:
`pip install -r requirements.txt`
This will install all the necessary packages required to run the project. It can take a little bit, don't worry, it's normal.

### Running

Once you have everything installed run the `mainfile.py` file in your terminal or code editor to execute the model.

## Future
