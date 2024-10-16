# MNIST Digit Recognizer

This project implements a Convolutional Neural Network (CNN) for the MNIST Digit Recognizer competition on Kaggle. The model achieves an accuracy of 98.857% on the test set.

## Competition Overview

The MNIST Digit Recognizer is a classic machine learning problem where the goal is to correctly identify digits from a dataset of tens of thousands of handwritten images. This project is based on the Kaggle competition found here:

[Digit Recognizer | Kaggle](https://www.kaggle.com/competitions/digit-recognizer/overview)

## Project Structure

The project is organized into three main directories:

- `data/`: Contains the training, testing, and submission CSV files.
- `model/`: Stores the trained model.
- `notebook/`: Includes Jupyter notebooks for model creation and submission generation.

## Model Architecture

The CNN model used in this project consists of:
- Two convolutional layers
- Max pooling layers
- Dropout for regularization
- Fully connected layers

## Results

The model achieved an accuracy of 98.857% on the Kaggle test set, demonstrating its effectiveness in recognizing handwritten digits.

## Usage

To run this project:

1. Ensure you have the required dependencies installed (PyTorch, pandas, numpy, etc.).
2. Execute the notebooks in the `notebook/` directory to train the model and generate submissions.
3. The trained model will be saved in the `model/` directory.
4. Submission files will be generated in the `data/` directory.

## Future Improvements

Potential areas for improvement include:
- Experimenting with different model architectures
- Implementing data augmentation techniques
- Fine-tuning hyperparameters

Feel free to contribute to this project by submitting pull requests or opening issues for discussion.