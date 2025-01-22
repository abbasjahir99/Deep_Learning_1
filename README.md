# Deep Learning 1: Linear/Logistic Regression and Neural Networks

![License](https://img.shields.io/github/license/abbasjahir99/Deep_Learning_1)
![Python Version](https://img.shields.io/badge/python-3.7%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-1.7.1-brightgreen)

This repository explores the implementation of linear regression, logistic regression, and various neural networks using PyTorch. It includes single-layer and multilayer perceptrons, as well as convolutional neural networks (CNNs) for classification and feature detection.

## Table of Contents

- [Features](#features)
  - [1. Linear/Logistic Regression as a Neural Network](#1-linearlogistic-regression-as-a-neural-network)
  - [2. Multilayer Perceptrons (MLP)](#2-multilayer-perceptrons-mlp)
  - [3. Convolutional Neural Networks (CNNs)](#3-convolutional-neural-networks-cnns)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [How to Run](#how-to-run)
- [Contents](#contents)
- [Results](#results)
- [Acknowledgments](#acknowledgments)
- [Contributing](#contributing)
- [License](#license)

## Features

### 1. Linear/Logistic Regression as a Neural Network
- **Regression**: Synthetic data generation and training a single-layer neural network.
- **Classification**: Using a SoftMax prediction head and cross-entropy loss for MNIST classification.

### 2. Multilayer Perceptrons (MLP)
- Implementation of a 3-layer MLP with ReLU activations and dropout layers for MNIST classification.

### 3. Convolutional Neural Networks (CNNs)
- **Custom 2D Convolution Operator**: Implementation from scratch, including padding and stride.
- **Edge Detection**: Learning horizontal edge detector kernels.
- **ResNet18**: Implementation of a modern CNN with residual blocks for image classification.

## Prerequisites
- Python 3.7+
- PyTorch
- NumPy
- Matplotlib
- Jupyter Notebook

## Installation

1. **Clone the repository:**
    ```bash
    git clone https://github.com/abbasjahir99/Deep_Learning_1.git
    cd Deep_Learning_1
    ```

2. **Create a virtual environment (optional but recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## How to Run

1. **Launch Jupyter Notebook:**
    ```bash
    jupyter notebook DL_1.ipynb
    ```

2. **Explore the Notebook:**
    - Follow the sections to understand and execute different model implementations and experiments.

## Contents

- **Linear Neural Networks**: Implementation for regression and classification.
- **Trainer Classes**: Modular training and evaluation for regression and classification models.
- **MLP for MNIST**: A 3-layer perceptron for image classification.
- **Custom Conv2D**: Implementation of a simple convolutional operator and edge detection kernel.
- **ResNet18**: Building and training a residual network for advanced classification tasks.

## Results

- **Training Metrics**: Visualizations of training loss, validation loss, and accuracy for each model.
- **Kernel Visualizations**: Learned kernels for convolutional operations demonstrating feature detection capabilities.

## Acknowledgments

- **ResNet18 Paper**: He, K., Zhang, X., Ren, S., & Sun, J. (2015). [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385).

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

1. **Fork the repository**
2. **Create a new branch**
    ```bash
    git checkout -b feature/YourFeature
    ```
3. **Commit your changes**
    ```bash
    git commit -m "Add your message"
    ```
4. **Push to the branch**
    ```bash
    git push origin feature/YourFeature
    ```
5. **Open a Pull Request**

## License

This project is licensed under the [MIT License](LICENSE).
