from flask import Flask, request
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import json

from PIL import Image

import requests

app = Flask(__name__)

# Allow
CORS(app)

# Path for uploaded images
UPLOAD_FOLDER = 'data/uploads/'

# Allowed file extransions
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


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
        # predicted_image_class = predict_img(UPLOAD_FOLDER+filename)
        # print("predicted_image_class", predicted_image_class)

    return json.dumps({"class": "12"})


if __name__ == "__main__":
    app.run(debug=True)