import tensorflow as tf
from Model_trainer import ModelTrainer
from Prediction_visualizer import PredictionVisualizer
from Data_augumentor import Data_augmentor

from config import (
    EPOCHS,
    BASE_MODEL_NAME,
)


class Trainer:
    def __init__(
        self,
        train_ds: tf.data.Dataset,
        val_ds: tf.data.Dataset,
        test_ds: tf.data.Dataset,
    ) -> None:
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.test_ds = test_ds

    def train(
        self,
        num_classes: int,
        class_names: list[str],
        visualize: bool,
        augment_data: bool,
    ):
        """
        Build and train the model with optional data augmentation.
        """

        augmentor = Data_augmentor(augment=augment_data)
        self.train_ds = augmentor.augment_dataset(self.train_ds)
        self.val_ds = self.val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
        self.test_ds = self.test_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

        model_trainer = ModelTrainer(
            num_classes=num_classes, epochs=EPOCHS, base_model_name=BASE_MODEL_NAME
        )
        model_trainer.build_model()
        model_trainer.compile_model()
        model_trainer.train_model(self.train_ds, self.val_ds)

        val_metrics = model_trainer.validate_model(self.val_ds)
        print("Validation metrics:", val_metrics)

        test_metrics = model_trainer.test_model(self.test_ds)
        print("Test metrics:", test_metrics)

        if visualize:
            visualizer = PredictionVisualizer(model_trainer.get_model(), class_names)
            visualizer.show_prediction(self.val_ds)
            visualizer.show_prediction(self.test_ds)
