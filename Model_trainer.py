from config import (
    IMG_HEIGHT,
    IMG_WIDTH,
    BASE_MODEL_NAME,
    INCLUDE_TOP,
    EPOCHS,
    OPTIMIZER,
    LOSS,
    METRICS,
)
from keras import models, layers, applications


class ModelTrainer:
    def __init__(self, num_classes):
        self.input_shape = (IMG_HEIGHT, IMG_WIDTH, 3)
        self.num_classes = num_classes
        self.model = None

    def build_model(self):
        if BASE_MODEL_NAME == "VGG16":
            base_model = applications.VGG16(
                weights="imagenet",
                include_top=INCLUDE_TOP,
                input_shape=self.input_shape,
            )
        elif BASE_MODEL_NAME == "Xception":
            base_model = applications.Xception(
                weights=None, input_shape=self.input_shape, classes=self.num_classes
            )

        self.model = models.Sequential(
            [
                base_model,
                layers.Flatten(),
                layers.Dense(256, activation="relu"),
                layers.Dropout(0.5),
                layers.Dense(self.num_classes, activation="softmax"),
            ]
        )

    def compile_model(self):
        self.model.compile(optimizer=OPTIMIZER, loss=LOSS, metrics=METRICS)

    def train_model(self, train_ds, val_ds):
        self.model.fit(train_ds, epochs=EPOCHS, validation_data=val_ds)

    def get_model(self):
        return self.model
