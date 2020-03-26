import tensorflow as tf
import numpy as np
from src.config import TEST_DIR, TRAIN_DIR, WEIGHTS_PATH
from src.preprocess import process_single_image
from src.model import AlexNet

def load_model():
    model = AlexNet()
    model.load_weights(WEIGHTS_PATH)
    return model

if __name__ == "__main__":
    model = load_model()
    test_ds = tf.data.Dataset.list_files(str(TEST_DIR / '*/'))
    CLASS_NAMES = np.array([item.name for item in TRAIN_DIR.glob('*')])
    print(CLASS_NAMES)
    for img_path in test_ds.take(1):
        print(img_path)
        image = process_single_image(img_path)
        prediction = model(image)
        class_prediction = tf.nn.softmax(prediction)
        print(class_prediction)