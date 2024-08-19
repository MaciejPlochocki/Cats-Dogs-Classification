from config import (
    BASE_MODEL_NAME,
    OPTIMIZER,
    LOSS,
    METRICS,
    PATIENCE,
    BASE_MODELS,
    INPUT_SHAPE,
)
from keras import models, layers
from keras.api.callbacks import ModelCheckpoint, EarlyStopping


class ModelTrainer:
    """
    A class to build, compile, and train  model.

    Attributes:

    input_shape : tuple
        The shape of the input data for the model.
    num_classes : int
        The number of output classes for the model.
    model : object
        The model to be trained.
    """

    def __init__(self, num_classes, epochs, base_model_name):
        """
        Initializes the ModelTrainer with the number of classes.

        Parameters:

        num_classes : int
            The number of output classes for the model.
        """
        self.input_shape = INPUT_SHAPE
        self.num_classes = num_classes
        self.base_model_name = base_model_name
        self.model = None
        self.EPOCHS = epochs

    def build_model(self):
        """
            Builds  model by customizing a pre-trained base model.

        This method initializes a base model from a predefined set of models
        (e.g., ResNet, Inception, list of available models: https://keras.io/api/applications/)  with weights pre-trained on the ImageNet dataset.
        Custom layers are added on top of the base model to adapt it for the specific
        classification task.

        Raises:

        ValueError:
            If the specified base model name is not in the predefined set of base models.

        Parameters:

        base_model_name : str
            The name of the pre-trained base model to be used.
        num_classes : int
            The number of output classes for the model.
        model : tensorflow.keras.Model
            The constructed Keras model ready for training.

        """
        if self.base_model_name not in BASE_MODELS:
            raise ValueError(
                f"This base model is not supported: {self.base_model_name}"
            )

        base_model_fn = BASE_MODELS[self.base_model_name]
        base_model = base_model_fn(
            weights="imagenet",
            include_top=False,
            input_shape=self.input_shape,
        )

        print("BASE MODEL IS: ", BASE_MODEL_NAME)

        x = base_model.output
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(256, activation="relu")(x)
        x = layers.Dropout(0.5)(x)
        predictions = layers.Dense(self.num_classes, activation="softmax")(x)

        self.model = models.Model(inputs=base_model.input, outputs=predictions)

    def compile_model(self):
        """
        Compiles the model with the specified optimizer, loss function, and metrics.
        """
        self.model.compile(optimizer=OPTIMIZER, loss=LOSS, metrics=METRICS)

    def train_model(self, train_ds, val_ds):
        """
        Trains the model using the provided training and validation datasets.

        Parameters:
        train_ds : tf.data.Dataset
            The dataset used for training the model.
        val_ds : tf.data.Dataset
            The dataset used for validating the model during training.
        """

        checkpoint = ModelCheckpoint("best_model.keras", save_best_only=True)
        callbacks = [
            EarlyStopping(PATIENCE),
        ]
        self.model.fit(
            train_ds, epochs=self.EPOCHS, validation_data=val_ds, callbacks=[checkpoint]
        )

    def validate_model(self, val_ds):
        """
        Validates the model using the validation ds.

        Parameters:
        val_ds : tf.data.Dataset
            The dataset used for validating the model.

        Returns:
        validation_metrics : dict
            The validation metrics of the model on the validation dataset.
        """
        return self.model.evaluate(val_ds, verbose=1)

    def test_model(self, test_ds):
        """
        Tests the model using test ds.

        Parameters:
        test_ds : tf.data.Dataset
            The dataset used for testing the model.

        Returns:
        test_metrics : dict
            The test metrics of the model on the test dataset.
        """
        return self.model.evaluate(test_ds, verbose=1)

    def get_model(self):
        """
        Returns the trained model.

        Returns:

        model : object
            The trained machine learning model.
        """
        return self.model

    def save_model(self, model_path="model.keras"):
        """
        Saves the entire model to a file.

        Parameters:
        model_path : str
            The path where the model will be saved.
        """
        self.model.save(model_path)

    def load_model(self, model_path="model.keras"):
        """
        Loads a model from a file.

        Parameters:
        model_path : str
            The path from where the model will be loaded.
        """
        self.model = models.load_model(model_path)

    def save_weights(self, weights_path="model_weights.h5"):
        """
        Saves the model's weights to a file.

        Parameters:
        weights_path : str
            The path where the weights will be saved.
        """
        self.model.save_weights(weights_path)

    def load_weights(self, weights_path="model_weights.h5"):
        """
        Loads the model's weights from a file.

        Parameters:
        weights_path : str
            The path from where the weights will be loaded.
        """
        self.model.load_weights(weights_path)
