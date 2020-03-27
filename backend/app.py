from flask import Flask, request
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import tensorflow as tf
import numpy as np
import json
from PIL import Image
from src.config import WEIGHTS_PATH, IMG_WIDTH, IMG_HEIGHT
from src.preprocess import process_single_image
from src.model import AlexNet

# physical_devices = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[0], True)

app = Flask(__name__)

# Allow
CORS(app)

# Path for uploaded images
UPLOAD_FOLDER = 'data/uploads/'

# Allowed file extransions
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def load_model():
    model = AlexNet()
    model.load_weights(WEIGHTS_PATH)
    return model

def convert_image(file):
    img = Image.open(file)
    print(type(img))
    img = np.array(img)
    print(img)
    img = tf.convert_to_tensor(
        img, dtype=None, dtype_hint=None, name=None
    )
    print(img)
    img = tf.dtypes.cast(img, tf.float32)
    img = tf.image.resize(img, [IMG_WIDTH, IMG_HEIGHT])
    img = tf.expand_dims(img, axis=0)
    return img

@app.route("/")
def hello():
    return "Hello World!"

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        print("request files", request.files)
        file = request.files['file']
        img = convert_image(file)

        print(f"Shape of image {img.shape}")
        model = load_model()
        predicted = model(img)
        predicted_image_class = tf.nn.softmax(predicted)
        predicted_image_class = predicted_image_class.numpy()
        print("predicted_image_class", predicted_image_class)
        # return json.dumps({"class": str(predicted_image_class)})
        return json.dumps({"class": "buildings"})

if __name__ == "__main__":
    app.run(debug=True)