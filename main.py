from Data_loader import DataLoader
from Model_trainer import ModelTrainer
from Prediction_visualizer import PredictionVisualizer

if __name__ == "__main__":
    # Loading data
    data_loader = DataLoader()
    data_loader.load_data()
    train_ds, val_ds, class_names, num_classes = data_loader.get_data()

    # Build and train model
    model_trainer = ModelTrainer(num_classes=num_classes)
    model_trainer.build_model()
    model_trainer.compile_model()
    model_trainer.train_model(train_ds, val_ds) 

    # Visualisation
    visualizer = PredictionVisualizer(model_trainer.get_model(), class_names)
    visualizer.show_prediction(val_ds)
