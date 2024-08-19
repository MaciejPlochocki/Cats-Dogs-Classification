from Data_loader import DataLoader
from Model_trainer import ModelTrainer
from Prediction_visualizer import PredictionVisualizer
from Model_predictor import ModelPredictor
from keras.api.models import load_model
from config import (
    EPOCHS,
    SRC_MODEL_PATH,
    IMAGE_PATH,
    INPUT_SHAPE,
    VIDEO_PATH,
    BASE_MODEL_NAME,
)

if __name__ == "__main__":
    # Loading data
    data_loader = DataLoader()
    data_loader.load_data()
    train_ds, val_ds, test_ds, class_names, num_classes = data_loader.get_data()

    # # Build and train model
    # model_trainer = ModelTrainer(
    #     num_classes=num_classes, epochs=EPOCHS, base_model_name=BASE_MODEL_NAME
    # )
    # model_trainer.build_model()
    # model_trainer.compile_model()
    # model_trainer.train_model(train_ds, val_ds)

    # # load model
    # # model = load_model(SRC_MODEL_PATH)

    # # Validation
    # val_metrics = model_trainer.validate_model(val_ds)
    # print("Validation metrics:", val_metrics)

    # # # Testing
    # test_metrics = model_trainer.test_model(test_ds)
    # print("Test metrics:", test_metrics)

    # # # Vizualization
    # visualizer = PredictionVisualizer(model_trainer.get_model(), class_names)
    # visualizer.show_prediction(val_ds)
    # visualizer.show_prediction(test_ds)

    # predict image
    predictor = ModelPredictor(
        SRC_MODEL_PATH, input_shape=INPUT_SHAPE, class_names=class_names
    )

    image_prediction = predictor.predict_image(IMAGE_PATH)

    print("Predicted class for image :", image_prediction)

    # predict video
    video_prediction = predictor.predict_video(VIDEO_PATH)
    print("Predicted class for the video:", video_prediction)
