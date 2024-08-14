from config import IMG_HEIGHT, IMG_WIDTH
import numpy as np
from keras.api.models import load_model
from keras.api.preprocessing.image import img_to_array, load_img
import tensorflow as tf


class ModelPredictor:
    def __init__(self, model_path, input_shape, class_names):
        self.model = load_model(model_path)
        self.input_shape = input_shape
        self.class_names = class_names

    def preprocess_image(self, image_path):
        image = load_img(
            image_path, target_size=(self.input_shape[0], self.input_shape[1])
        )
        img_array = img_to_array(image)
        img_array = tf.image.resize(
            img_array, (self.input_shape[0], self.input_shape[1])
        )
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0
        return img_array

    def predict_image(self, image_path):
        preprocessed_image = self.preprocess_image(image_path)
        predictions = self.model.predict(preprocessed_image)

        predicted_class_index = np.argmax(predictions[0])
        predicted_class_name = self.class_names[predicted_class_index]

        return predicted_class_name

    # to do:
    def preprocess_video(self, video_path):
        print("videło")

    def predict_video(self, video_path):
        print("kłot")
