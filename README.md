# Image Classification Project - Internship 2024

## Table of Contents
- [Project Overview](#project-overview)
- [Tech Stack](#tech-stack)
- [Configuration](#configuration)
- [Data Loading](#data-loading)
- [Model Training](#model-training)
- [Prediction](#prediction)
- [Usage](#usage)
- [Future Improvements](#future-improvements)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

This repository contains the code and documentation for an image classification project completed during my internship at **[Company Name]** in 2024. The project focuses on classifying images into two categories: cats and dogs. The goal is to build a deep learning model that can accurately classify images into these categories.

## Tech Stack

The project utilizes the following technologies and tools:
- **Python**: The primary programming language used.
- **Jupyter Notebook**: For exploratory data analysis and model development.
- **TensorFlow & Keras**: For building, training, and evaluating deep learning models.
- **OpenCV**: For video processing.
- **Matplotlib**: For visualization.
- **Git**: For version control.

## Configuration

The configuration settings for the project are specified in the `config.py` file. Key settings include:
- **Paths**:
  - `SRC_PATH_TRAIN`: Directory for training images.
  - `SRC_PATH_VALID`: Directory for validation images.
  - `SRC_PATH_TEST`: Directory for test images.
  - `SRC_MODEL_PATH`: Path to the saved model.
  - `IMAGE_PATH`: Path to an image for prediction.
  - `VIDEO_PATH`: Path to a video for prediction.
- **Model Parameters**:
  - `BATCH_SIZE`: Number of images per batch.
  - `IMG_HEIGHT`: Height of images.
  - `IMG_WIDTH`: Width of images.
  - `EPOCHS`: Number of training epochs.
  - `BASE_MODEL_NAME`: Name of the base model to use (e.g., Xception).
  - `INCLUDE_TOP`: Whether to include the top layer of the base model.
  - `PATIENCE`: Patience for early stopping.
- **Compilation Parameters**:
  - `OPTIMIZER`: Optimizer for model training.
  - `LOSS`: Loss function for model training.
  - `METRICS`: Metrics for model evaluation.

## Data Loading

The `DataLoader` class in `Data_loader.py` handles loading and preparing the datasets. It loads images from the specified directories and prepares them for training and validation. It also provides the dataset and class names for the model.

### Key Methods:
- `load_data()`: Loads training, validation, and test datasets.
- `get_data()`: Returns the datasets and class information.

## Model Training

The `ModelTrainer` class in `Model_trainer.py` is responsible for building, compiling, and training the model. It uses a pre-trained base model and adds custom layers for classification.

### Key Methods:
- `build_model()`: Builds the model using the specified base model.
- `compile_model()`: Compiles the model with the chosen optimizer, loss function, and metrics.
- `train_model()`: Trains the model on the provided datasets.
- `validate_model()`: Evaluates the model on the validation dataset.
- `test_model()`: Evaluates the model on the test dataset.

## Prediction

The `ModelPredictor` class in `Model_predictor.py` is used for making predictions on images and videos using the trained model.

### Key Methods:
- `predict_image(image_path)`: Predicts the class of an image.
- `predict_video(video_path, frame_skip=10)`: Predicts the class of a video by analyzing frames.

## Usage

To use the project:

1. **Setup**:
   - Ensure all dependencies are installed. You can install them using the `requirements.txt` file.
   - Configure paths and parameters in `config.py` as needed.

2. **Training**:
   - Uncomment the training section in `main.py` to train the model.
   ```python
   # Training
   trainer = Trainer(train_ds, val_ds, test_ds)
   trainer.train(num_classes, class_names, True, True)
3. **Prediction**:
   -Run the main.py script to make predictions on images and videos..
   ```python
   # Prediction
    predictor = Predictor()
    predictor.predict_image(IMAGE_PATH)
    predictor.predict_video(VIDEO_PATH)
