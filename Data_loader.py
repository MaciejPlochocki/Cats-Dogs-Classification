from config import (
    SRC_PATH_TRAIN,
    SRC_PATH_VALID,
    SRC_PATH_TEST,
    IMG_HEIGHT,
    IMG_WIDTH,
    BATCH_SIZE,
    LABEL_MODE,
    SHUFFLE,
)

from keras import utils
import tensorflow as tf


class DataLoader:
    """
    A class to load and prepare training and validation datasets for a model.

    Attributes:

    train_dir : str
        The path to the directory containing training data.
    valid_dir : str
        The path to the directory containing validation data.
    img_height : int
        The height of the images to be loaded.
    img_width : int
        The width of the images to be loaded.
    batch_size : int
        The number of images per batch.
    label_mode : str
        The mode for labels, e.g., 'int', 'categorical', or 'binary'.
    shuffle : bool
        Whether to shuffle the dataset.
    train_ds : tf.data.Dataset
        The dataset for training the model.
    val_ds : tf.data.Dataset
        The dataset for validating the model.
    class_names : list of str
        The names of the classes in the dataset.
    num_classes : int
        The number of classes in the dataset.
    """

    def __init__(self) -> None:
        """
        Initializes the DataLoader with specified configurations for image data loading.
        """
        self.train_dir = SRC_PATH_TRAIN
        self.valid_dir = SRC_PATH_VALID
        self.test_dir = SRC_PATH_TEST
        self.img_height = IMG_HEIGHT
        self.img_width = IMG_WIDTH
        self.batch_size = BATCH_SIZE
        self.label_mode = LABEL_MODE
        self.shuffle = SHUFFLE
        self.train_ds = None
        self.val_ds = None
        self.class_names: list[str] = []
        self.num_classes: int = 0

    def load_data(self) -> None:
        """
        Loads the training and validation datasets from the specified directories.
        """
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
        self.test_ds = utils.image_dataset_from_directory(
            self.valid_dir,
            labels="inferred",
            label_mode=self.label_mode,
            batch_size=self.batch_size,
            image_size=(self.img_height, self.img_width),
            shuffle=False,
        )
        self.class_names = self.train_ds.class_names
        self.num_classes = len(self.class_names)

    def get_data(self):
        """
        Returns the loaded datasets and class information.

        Returns:

        train_ds : tf.data.Dataset
            The dataset for training the model.
        val_ds : tf.data.Dataset
            The dataset for validating the model.
        class_names : list of str
            The names of the classes in the dataset.
        num_classes : int
            The number of classes in the dataset.
        """
        return (
            self.train_ds,
            self.val_ds,
            self.test_ds,
            self.class_names,
            self.num_classes,
        )
