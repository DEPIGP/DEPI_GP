from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

app = Flask(__name__)
model=load_model('CoronaModel.keras')
# Specify the upload folder
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/upload', methods=['POST'])
def upload_file():
    # Check if the post request has the file part
    if 'image' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['image']

    # If the user does not select a file, the browser submits an empty file without a filename
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file:
        # Ensure a secure filename to prevent any potential security risks
        filename = secure_filename(file.filename)
        # Save the file to the upload folder
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        target_size = (100, 100,3)
        img=load_img(file_path,target_size=target_size)
        img_array = img_to_array(img)
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        predictions = model.predict(img_array)
        predicted_class = 1 if predictions[0] > 0.5 else 0
        if predicted_class==1:
            return f'your result is {predicted_class} which means you have corona'
        else:
            return f'your result is {predicted_class} which means you are fine'


if __name__ == '__main__':
    app.run(debug=True)
