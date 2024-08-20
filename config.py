from keras.api import applications

# Path to ds
SRC_PATH_TRAIN = "ds/catsvsdogs/imgs/trainv2/"
SRC_PATH_VALID = "ds/catsvsdogs/imgs/validation/"
SRC_PATH_TEST = "ds/catsvsdogs/imgs/test/"
SRC_MODEL_PATH = "models/best_model.keras"
IMAGE_PATH = "ds/catsvsdogs/imgs/pred/19.jpg"
VIDEO_PATH = "ds/catsvsdogs/imgs/pred/cat1.mp4"

# Model and training params
BATCH_SIZE = 20
IMG_HEIGHT = 256
IMG_WIDTH = 256
EPOCHS = 10
BASE_MODEL_NAME = "Xception"
INCLUDE_TOP = True
PATIENCE = 3

# Compilation model params
OPTIMIZER = "adam"
LOSS = "categorical_crossentropy"
METRICS = ["accuracy"]
# others
LABEL_MODE = "categorical"
SHUFFLE = True
INPUT_SHAPE = (IMG_HEIGHT, IMG_WIDTH, 3)

# Base models dict
BASE_MODELS = {
    "VGG16": applications.VGG16,
    "VGG19": applications.VGG19,
    "Xception": applications.Xception,
    "ResNet50": applications.ResNet50,
    "ResNet50V2": applications.ResNet50V2,
    "ResNet101": applications.ResNet101,
    "ResNet101V2": applications.ResNet101V2,
    "ResNet152": applications.ResNet152,
    "ResNet152V2": applications.ResNet152V2,
    "InceptionV3": applications.InceptionV3,
    "InceptionResNetV2": applications.InceptionResNetV2,
    "MobileNet": applications.MobileNet,
    "MobileNetV2": applications.MobileNetV2,
    "DenseNet121": applications.DenseNet121,
    "DenseNet169": applications.DenseNet169,
    "DenseNet201": applications.DenseNet201,
    "NASNetMobile": applications.NASNetMobile,
    "NASNetLarge": applications.NASNetLarge,
    "EfficientNetB0": applications.EfficientNetB0,
    "EfficientNetB1": applications.EfficientNetB1,
    "EfficientNetB2": applications.EfficientNetB2,
    "EfficientNetB3": applications.EfficientNetB3,
    "EfficientNetB4": applications.EfficientNetB4,
    "EfficientNetB5": applications.EfficientNetB5,
    "EfficientNetB6": applications.EfficientNetB6,
    "EfficientNetB7": applications.EfficientNetB7,
    "EfficientNetV2B0": applications.EfficientNetV2B0,
    "EfficientNetV2B1": applications.EfficientNetV2B1,
    "EfficientNetV2B2": applications.EfficientNetV2B2,
    "EfficientNetV2B3": applications.EfficientNetV2B3,
    "EfficientNetV2S": applications.EfficientNetV2S,
    "EfficientNetV2M": applications.EfficientNetV2M,
    "EfficientNetV2L": applications.EfficientNetV2L,
    "ConvNeXtTiny": applications.ConvNeXtTiny,
    "ConvNeXtSmall": applications.ConvNeXtSmall,
    "ConvNeXtBase": applications.ConvNeXtBase,
    "ConvNeXtLarge": applications.ConvNeXtLarge,
    "ConvNeXtXLarge": applications.ConvNeXtXLarge,
}
