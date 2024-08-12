from Data_loader import DataLoader
from Model_trainer import ModelTrainer
from Prediction_visualizer import PredictionVisualizer
from config import EPOCHS

if __name__ == "__main__":
    # Loading data
    data_loader = DataLoader()
    data_loader.load_data()
    train_ds, val_ds, test_ds, class_names, num_classes = data_loader.get_data()

    # Build and train model
    model_trainer = ModelTrainer(num_classes=num_classes, epochs=EPOCHS)
    model_trainer.build_model()
    model_trainer.compile_model()
    model_trainer.train_model(train_ds, val_ds)

    # Walidacja
    val_metrics = model_trainer.validate_model(val_ds)
    print("Validation metrics:", val_metrics)

    # Testowanie
    test_metrics = model_trainer.test_model(test_ds)
    print("Test metrics:", test_metrics)

    # Wizualizacja
    visualizer = PredictionVisualizer(model_trainer.get_model(), class_names)
    visualizer.show_prediction(val_ds)
    visualizer.show_test_predictions(test_ds)
