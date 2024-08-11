import numpy as np
import matplotlib.pyplot as plt


class PredictionVisualizer:
    def __init__(self, model, class_names):
        self.model = model
        self.class_names = class_names

    def show_prediction(self, data):
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
            plt.title(f"Pred: {self.class_names[predicted_label]}\nTrue: {self.class_names[true_label]}")
        
        plt.tight_layout()
        plt.show()
