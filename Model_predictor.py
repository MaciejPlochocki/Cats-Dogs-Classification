import cv2
import numpy as np
from keras.api.models import load_model
from keras.api.preprocessing.image import img_to_array, load_img
import tensorflow as tf


class ModelPredictor:
    """
    The ModelPredictor class is used for loading a deep learning model and making predictions on images and videos.

    Attributes:
    -----------
    model : tensorflow.keras.Model
        The loaded Keras model used for making predictions.
    input_shape : tuple
        A tuple specifying the input shape of images that the model can process, e.g., (224, 224, 3).
    class_names : list
        A list of class names that the model can predict.

    Methods:
    --------
    preprocess_image(image):
        Preprocesses the image to the correct input format for the model.
    predict_image(image_path):
        Predicts the class of the given image.
    predict_video(video_path):
        Predicts the class of the given video.
    """

    def __init__(self, model_path: str, input_shape: tuple, class_names: list):
        """
        Initializes the ModelPredictor object.

        Parameters:
        ----------
        model_path : The path to the saved Keras model file (.h5).
        input_shape : The input shape of images for the model in the format (height, width, number of channels).
        class_names : list of class names
        """
        self.model = load_model(model_path)
        self.input_shape = input_shape
        self.class_names = class_names

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocesses the image to match the model's input requirements.

        """
        img_array = img_to_array(image)
        img_array = tf.image.resize(
            img_array, (self.input_shape[0], self.input_shape[1])
        )
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0
        return img_array

    def predict_image(self, image_path: str) -> str:
        """
        Predicts the class of the given image using the loaded model.

        """
        image = load_img(image_path)
        preprocessed_image = self.preprocess_image(image)
        predictions = self.model.predict(preprocessed_image)

        predicted_class_index = np.argmax(predictions[0])
        predicted_class_name = self.class_names[predicted_class_index]

        return predicted_class_name

    def predict_video(self, video_path: str, frame_skip: int = 10) -> str:
        """
        Predicts the class of the given video using loaded model.

        """
        cap = cv2.VideoCapture(video_path)
        predictions = []
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_skip == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                preprocessed_frame = self.preprocess_image(frame_rgb)
                prediction = self.model.predict(preprocessed_frame)
                predictions.append(prediction)

            frame_count += 1

        cap.release()

        predictions = np.array(predictions)
        average_prediction = np.mean(predictions, axis=0)
        predicted_class_index = np.argmax(average_prediction)
        predicted_class_name = self.class_names[predicted_class_index]

        return predicted_class_name
