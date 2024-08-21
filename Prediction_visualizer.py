import numpy as np
import matplotlib.pyplot as plt


class PredictionVisualizer:
    """
    A class to visualize predictions of a trained  model on image data.

    Attributes:

    model : object
        The trained model used for making predictions on the input data.
    class_names : list of str
        A list of class names corresponding to the model's output labels.
    """

    def __init__(self, model: object, class_names: list) -> None:
        """
        Initializes the PredictionVisualizer with a model and class names.

        Parameters:

        model : The trained model used to predict labels for the images.
        class_names : The list of class names that correspond to the model's output classes.
        """
        self.model = model
        self.class_names = class_names

    def show_prediction(self, data) -> None:
        """
        Displays images with their predicted and true labels.

        Parameters:

        data : generator or batch
            A data generator or batch that yields a tuple of (images, labels), where:
            images : array-like
                A batch of images to be predicted.
            labels : array-like
                The true labels for the batch of images.
        """
        images, labels = next(iter(data))
        num_images: int = len(images)
        plt.figure(figsize=(12, 12))

        for i in range(num_images):
            image = images[i]
            true_label = np.argmax(labels[i].numpy())

            image = np.expand_dims(image, axis=0)
            predictions = self.model.predict(image)
            predicted_label = np.argmax(predictions, axis=1)[0]

            plt.subplot(int(np.sqrt(num_images)), int(np.sqrt(num_images)) + 1, i + 1)
            plt.imshow(image[0].astype("uint8"))
            plt.axis("off")
            plt.title(
                f"Pred label: {self.class_names[predicted_label]}\nTrue label: {self.class_names[true_label]}"
            )

        plt.tight_layout()
        plt.show()
