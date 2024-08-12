# Path to ds


SRC_PATH_TRAIN = "ds/catsvsdogs/imgs/train/"
SRC_PATH_VALID = "ds/catsvsdogs/imgs/validation/"
SRC_PATH_TEST = "ds/catsvsdogs/imgs/test/"

# Model and training params
BATCH_SIZE = 20
IMG_HEIGHT = 256
IMG_WIDTH = 256
EPOCHS = 10
BASE_MODEL_NAME = "Xception"
INCLUDE_TOP = True

# Compilation model params
OPTIMIZER = "adam"
LOSS = "categorical_crossentropy"
METRICS = ["accuracy"]
# others
LABEL_MODE = "categorical"
SHUFFLE = True  # Czy mieszać dane
