from Predictor import Predictor
from Data_loader import DataLoader
from Trainer import Trainer
import tensorflow as tf

from config import (
    IMAGE_PATH,
    VIDEO_PATH,
)

if __name__ == "__main__":
    # Loading data
    data_loader = DataLoader()
    data_loader.load_data()

    train_ds, val_ds, test_ds, class_names, num_classes = data_loader.get_data()

    # Training
    trainer = Trainer(train_ds, val_ds, test_ds)
    trainer.train(num_classes, class_names, True, True)

    # Prediction
    predictor = Predictor()
    predictor.predict_image(IMAGE_PATH)
    predictor.predict_video(VIDEO_PATH)
