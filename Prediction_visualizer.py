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

    def __init__(self, model, class_names):
        """
        Initializes the PredictionVisualizer with a model and class names.

        Parameters:

        model : object
            The trained model used to predict labels for the images.
        class_names : list of str
            The list of class names that correspond to the model's output classes.
        """
        self.model = model
        self.class_names = class_names

    def show_prediction(self, data):
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
        num_images = len(images)
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
                f"Pred: {self.class_names[predicted_label]}\nTrue: {self.class_names[true_label]}"
            )

        plt.tight_layout()
        plt.show()

    def show_test_predictions(self, dataset):
        plt.figure(figsize=(10, 10))
        for images, labels in dataset.take(1):
            predictions = self.model.predict(images)
            for i in range(9):
                ax = plt.subplot(3, 3, i + 1)
                plt.imshow(images[i].numpy().astype("uint8"))
                pred_class = np.argmax(predictions[i])
                true_class = np.argmax(labels[i])
                confidence = np.max(predictions[i]) * 100
                plt.title(
                    f"True: {self.class_names[true_class]}\nPred: {self.class_names[pred_class]} ({confidence:.2f}%)"
                )
                plt.axis("off")
        plt.show()
