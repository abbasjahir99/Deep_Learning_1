Linear/Logistic Regression and Neural Networks

This repository explores the implementation of linear regression, logistic regression, and neural networks, including single-layer and multilayer perceptrons, using PyTorch. Additionally, convolutional neural networks (CNNs) are implemented for classification and feature detection.

Features

1. Linear/Logistic Regression as a Neural Network

Regression: Synthetic data generation and training a single-layer neural network.

Classification: Using a SoftMax prediction head and cross-entropy loss for MNIST classification.

2. Multilayer Perceptrons (MLP)

Implementation of a 3-layer MLP with ReLU activations and dropout layers for MNIST classification.

3. Convolutional Neural Networks (CNNs)

Custom 2D Convolution Operator: Implementation from scratch, including padding and stride.

Edge Detection: Learning horizontal edge detector kernels.

ResNet18: Implementation of a modern CNN with residual blocks for image classification.

Prerequisites

Python 3.7+

PyTorch

NumPy

Matplotlib

How to Run

Clone the repository:

git clone <repository_url>
cd <repository_name>

Install dependencies:

pip install -r requirements.txt

Run the Jupyter Notebook:

jupyter notebook hw1.ipynb

Contents

Linear Neural Networks: Implementation for regression and classification.

Trainer Classes: Modular training and evaluation for regression and classification models.

MLP for MNIST: A 3-layer perceptron for image classification.

Custom Conv2D: Implementation of a simple convolutional operator and edge detection kernel.

ResNet18: Building and training a residual network for advanced classification tasks.

Results

Demonstrates training loss, validation loss, and accuracy for each model.

Visualizes learned kernels for convolutional operations.

Acknowledgments

ResNet18 paper: He et al., 2015.

Feel free to contribute or raise issues in the repository!
