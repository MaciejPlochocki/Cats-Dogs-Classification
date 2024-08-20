from config import INPUT_SHAPE, SRC_MODEL_PATH
from Model_predictor import ModelPredictor
from Data_loader import DataLoader


class Predictor:
    def __init__(self, class_names: str) -> None:
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
