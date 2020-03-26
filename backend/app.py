from flask import Flask, request
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import tensorflow as tf
import json
from src.config import WEIGHTS_PATH
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

@app.route("/")
def hello():
    return "Hello World!"

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        print("request data", request.data)
        print("request files", request.files)

        # check if the post request has the file part
        if 'file' not in request.files:
            return "No file part"
        file = request.files['file']

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

        # Send uploaded image for prediction
        model = load_model()
        test_ds = tf.data.Dataset.list_files(str(UPLOAD_FOLDER / '*/'))
        for image_path in test_ds:
            img = process_single_image(image_path)
            predicted = model(img)
            predicted_image_class = tf.argmax(tf.nn.softmax(predicted))
            print("predicted_image_class", predicted_image_class)
            return json.dumps({"class": predicted_image_class})

if __name__ == "__main__":
    app.run(debug=True)