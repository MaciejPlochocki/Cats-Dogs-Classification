from config import (
    SRC_PATH_TRAIN,
    SRC_PATH_VALID,
    IMG_HEIGHT,
    IMG_WIDTH,
    BATCH_SIZE,
    LABEL_MODE,
    SHUFFLE,
)
import tensorflow as tf
from keras import utils


class DataLoader:
    def __init__(self):
        self.train_dir = SRC_PATH_TRAIN
        self.valid_dir = SRC_PATH_VALID
        self.img_height = IMG_HEIGHT
        self.img_width = IMG_WIDTH
        self.batch_size = BATCH_SIZE
        self.label_mode = LABEL_MODE
        self.shuffle = SHUFFLE
        self.train_ds = None
        self.val_ds = None
        self.class_names = []
        self.num_classes = 0

    def load_data(self):
        self.train_ds = utils.image_dataset_from_directory(
            self.train_dir,
            labels="inferred",
            label_mode=self.label_mode,
            batch_size=self.batch_size,
            image_size=(self.img_height, self.img_width),
            shuffle=self.shuffle,
        )
        self.val_ds = utils.image_dataset_from_directory(
            self.valid_dir,
            labels="inferred",
            label_mode=self.label_mode,
            batch_size=self.batch_size,
            image_size=(self.img_height, self.img_width),
            shuffle=self.shuffle,
        )
        self.class_names = self.train_ds.class_names
        self.num_classes = len(self.class_names)

    def get_data(self):
        return self.train_ds, self.val_ds, self.class_names, self.num_classes
