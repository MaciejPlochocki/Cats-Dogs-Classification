import tensorflow as tf
from Model_trainer import ModelTrainer
from Prediction_visualizer import PredictionVisualizer

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

    def train(self, num_classes: int, class_names, visualize: bool):
        # Build and train model
        model_trainer = ModelTrainer(
            num_classes=num_classes, epochs=EPOCHS, base_model_name=BASE_MODEL_NAME
        )
        model_trainer.build_model()
        model_trainer.compile_model()
        model_trainer.train_model(self.train_ds, self.val_ds)

        # Validation
        val_metrics = model_trainer.validate_model(self.val_ds)
        print("Validation metrics:", val_metrics)

        # Testing
        test_metrics = model_trainer.test_model(self.test_ds)
        print("Test metrics:", test_metrics)

        if visualize:
            # Vizualization
            visualizer = PredictionVisualizer(model_trainer.get_model(), class_names)
            visualizer.show_prediction(self.val_ds)
            visualizer.show_prediction(self.test_ds)
