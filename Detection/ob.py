from flask import Flask, render_template_string, request, send_from_directory
import os
import torch
import cv2
from matplotlib import pyplot as plt
import numpy as np

app = Flask(__name__)

# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    
# Path to save uploaded images
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template_string(index_html)

@app.route('/detect', methods=['POST'])
def detect_objects():
    if 'image' not in request.files:
        return "No file uploaded", 400
    file = request.files['image']
    if file.filename == '':
        return "No selected file", 400

    # Save the uploaded file
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    # Perform object detection
    results = model(filepath)
    
    # Get the number of objects detected
    object_count = len(results.xyxy[0])

    # Extract the predicted class indices and map them to class names
    detected_class_indices = results.pred[0][:, -1].tolist()
    detected_objects = [results.names[int(idx)] for idx in detected_class_indices]

    # Convert image to numpy array and display with bounding boxes
    img = cv2.imread(filepath)
    results.render()  # Updates results with bounding boxes

    # Convert BGR to RGB for displaying in Matplotlib
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Save the resulting image with bounding boxes
    result_image_path = os.path.join(UPLOAD_FOLDER, 'detected_' + file.filename)
    plt.imsave(result_image_path, img_rgb)

    return render_template_string(result_html, image_url=f'/uploads/detected_{file.filename}', object_count=object_count, detected_objects=detected_objects)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

# HTML for the main index page
index_html = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Object Detection</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            background-color: #f4f4f9;
            margin: 0;
            padding: 0;
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
            flex-direction: column;
        }
        h1 {
            color: #333;
        }
        form {
            background-color: #fff;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 0 15px rgba(0, 0, 0, 0.1);
            text-align: center;
        }
        input[type="file"] {
            margin: 10px 0;
        }
        input[type="submit"] {
            padding: 10px 20px;
            background-color: #4CAF50;
            border: none;
            color: white;
            border-radius: 5px;
            cursor: pointer;
        }
        input[type="submit"]:hover {
            background-color: #45a049;
        }
        footer {
            margin-top: 20px;
            color: #777;
        }
    </style>
</head>
<body>
    <h1>Object Detection</h1>
    <form action="/detect" method="POST" enctype="multipart/form-data">
        <input type="file" name="image" accept="image/*" required>
        <br>
        <input type="submit" value="Detect Objects">
    </form>
    <footer>&copy; 2024 Object Detection App</footer>
</body>
</html>
'''

# HTML for displaying detection results
result_html = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Object Detection Result</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            background-color: #f4f4f9;
            margin: 0;
            padding: 0;
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
            flex-direction: column;
        }
        h1 {
            color: #333;
        }
        img {
            max-width: 80%;
            height: auto;
            border: 5px solid #333;
            margin-top: 20px;
        }
        a {
            text-decoration: none;
            color: #4CAF50;
            margin-top: 20px;
            font-weight: bold;
        }
    </style>
</head>
<body>
    <h1>Detection Result</h1>
    <p>Number of objects detected: {{ object_count }}</p>
    <p>Objects detected: {{ detected_objects }}</p>
    <img src="{{ image_url }}" alt="Detected Objects">
    <br>
    <a href="/">Upload another image</a>
</body>
</html>
'''

if __name__ == '__main__':
    app.run(debug=True, port=5001)
