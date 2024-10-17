from flask import Flask, render_template, request, send_from_directory
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import os

app = Flask(__name__)
model = load_model('F:/FlaskApp/dataset/m.h5')  # Ensure the path to your model is correct

class_names = {
    0: 'fresh_apple',
    1: 'fresh_banana',
    2: 'fresh_bitter_gourd',
    3: 'fresh_capsicum',
    4: 'fresh_orange',
    5: 'fresh_tomato',
    6: 'stale_apple',
    7: 'stale_banana',
    8: 'stale_bitter_gourd',
    9: 'stale_capsicum',
    10: 'stale_orange',
    11: 'stale_tomato'
}

def preprocess_image(image_path):
    img = load_img(image_path, target_size=(128, 128))  # Resize to (128, 128)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize image
    return img_array

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/freshness', methods=['GET', 'POST'])
def freshness():
    if request.method == 'POST':
        if 'image' not in request.files:
            return "No file provided", 400
        file = request.files['image']
        if file.filename == '':
            return "No selected file", 400

        # Save the uploaded file
        filepath = os.path.join('uploads', file.filename)
        os.makedirs('uploads', exist_ok=True)
        file.save(filepath)

        # Preprocess the image and make a prediction
        img_array = preprocess_image(filepath)
        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction, axis=-1)[0]  # Get predicted class index
        class_name = class_names.get(predicted_class, "Unknown Class")

        # Display both the image and the predicted class name
        image_url = f'/uploads/{file.filename}'
        return render_template('upload.html', title='Check Freshness', image_url=image_url, result=f'Predicted Class: {class_name}')
    else:
        return render_template('upload.html', title='Check Freshness')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory('uploads', filename)

if __name__ == '__main__':
    app.run(debug=True)
