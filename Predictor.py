from config import INPUT_SHAPE, SRC_MODEL_PATH
from Model_predictor import ModelPredictor
from Data_loader import DataLoader

data_loader = DataLoader()
data_loader.load_data()
train_ds, val_ds, test_ds, class_names, num_classes = data_loader.get_data()


class Predictor:
    def __init__(self) -> None:
        self.predictor = ModelPredictor(
            SRC_MODEL_PATH, input_shape=INPUT_SHAPE, class_names=class_names
        )

    def predict_image(self, img_path: str):
        # predict image
        image_prediction = self.predictor.predict_image(img_path)

        print("Predicted class for image :", image_prediction)

    def predict_video(self, vid_path: str):
        # predict video
        video_prediction = self.predictor.predict_video(vid_path)
        print("Predicted class for the video:", video_prediction)
