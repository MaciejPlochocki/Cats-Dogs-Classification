# config.py

# Ścieżki do danych
SRC_PATH_TRAIN = "ds/catsvsdogs/imgs/train/"
SRC_PATH_VALID = "ds/catsvsdogs/imgs/validation/"

# Parametry modelu i treningu
BATCH_SIZE = 20
IMG_HEIGHT = 256
IMG_WIDTH = 256
EPOCHS = 10
BASE_MODEL_NAME = "Xception"  # Można zmienić na "VGG16", "ResNet50", etc.
INCLUDE_TOP = True

# Parametry kompilacji modelu
OPTIMIZER = "rmsprop"
LOSS = "categorical_crossentropy"
METRICS = ["accuracy"]
# others
LABEL_MODE = "categorical"  # Może być "int", "categorical", "binary"
SHUFFLE = True  # Czy mieszać dane
