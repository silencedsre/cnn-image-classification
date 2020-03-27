from flask import Flask, request
from flask_cors import CORS
import json
from src.preprocess import convert_image
from src.predict import load_model, make_predictions, check_class

# physical_devices = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[0], True)

app = Flask(__name__)

# Allow
CORS(app)

@app.route("/")
def hello():
    return "Hello World!"

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        print("request files", request.files)
        file = request.files['file']
        img = convert_image(file)
        print(f"Shape of image {img.shape}")
        model = load_model()
        predicted = model(img)
        class_name = make_predictions(predicted)
        print("predicted_image_class", class_name)
        return json.dumps({"class": class_name})

if __name__ == "__main__":
    app.run(debug=True)