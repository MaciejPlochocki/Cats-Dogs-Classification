import tensorflow as tf


class Data_augmentor:
    def __init__(self, augment: bool = True):
        """
        Initializes the DataAugmentor with the option to enable or disable augmentation.

        Parameters:
        - augment (bool): Whether to apply data augmentation or not.
        """
        self.augment = augment
        if self.augment:
            self.data_augmentation = tf.keras.Sequential(
                [
                    tf.keras.layers.RandomFlip("horizontal_and_vertical"),
                    tf.keras.layers.RandomRotation(0.2),
                    tf.keras.layers.RandomZoom(0.2),
                ]
            )

    def augment_dataset(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        """
        Applies augmentation to the dataset if enabled.

        Parameters:
        - dataset (tf.data.Dataset): The dataset to be augmented.

        Returns:
        - tf.data.Dataset: The augmented dataset.
        """
        if self.augment:
            dataset = dataset.map(
                lambda x, y: (self.data_augmentation(x, training=True), y),
                num_parallel_calls=tf.data.AUTOTUNE,
            )
        return dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
