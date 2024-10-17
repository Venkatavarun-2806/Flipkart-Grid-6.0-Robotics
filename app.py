from flask import Flask, render_template, request
import cloudinary
import cloudinary.uploader
import requests
from PIL import Image
from io import BytesIO
from paddleocr import PaddleOCR
import cv2
import numpy as np
import os
import re
from datetime import datetime 

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Initialize the PaddleOCR tool (language can be changed as needed)
ocr = PaddleOCR(use_angle_cls=True, lang='en',cpu_threads=1)

# Regular expression patterns for expiry dates
expiry_date_patterns = [
    r'consume\s*before\s*[:\-]?\s*.*?(\d{4}[\/\-]\d{2}[\/\-]\d{2})',  # Consume Before: 2024/07/20
    r'exp\s*[:\-]?\s*.*?(\d{2}[\/\-]\d{2}[\/\-]\d{4})',  # Exp: 20/07/2024
    r'exp\s*[:\-]?\s*.*?(\d{2}\s[A-Za-z]{3,}\s*\d{4})',  # Exp: 20 MAY 2024
    r'(\d{2}[\/\-]\d{2}[\/\-]\d{4})',  # 20/07/2024
    r'(\d{2}\s*[A-Za-z]{3,}\s*\d{4})',  # 20 MAY 2024
    r'(\d{4}[\/\-]\d{2}[\/\-]\d{2})',  # 2024/07/20
]

# Initialize Flask app
app = Flask(__name__)

# Configure Cloudinary
cloudinary.config(
    cloud_name='dyeyolqjw',
    api_key='634952695785272',
    api_secret='yZ638kOoevqbsOo80BWNodKpPCU'
)

# Function to fetch image from URL
def fetch_image_from_url(image_url):
    """
    Fetch image from URL and return it as a PIL Image.
    """
    response = requests.get(image_url)
    img = Image.open(BytesIO(response.content))
    return img

# Function to perform OCR on the image
def perform_ocr_on_image(image_url):
    """
    Fetch the image from the URL, run PaddleOCR, and return extracted text.
    """
    # Fetch image
    img = fetch_image_from_url(image_url)

    # Convert PIL image to OpenCV format (PaddleOCR works with OpenCV images)
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    # Perform OCR on the image
    ocr_results = ocr.ocr(img_cv)

    # Extract text from the OCR results
    extracted_text = ""
    for line in ocr_results:
        for word_info in line:
            extracted_text += word_info[1][0] + " "

    return extracted_text.strip()


def find_expiry_date(extracted_text):
    """
    Find expiry date in the extracted text using regular expressions.
    """
    for pattern in expiry_date_patterns:
        match = re.search(pattern, extracted_text, re.IGNORECASE)
        if match:
            return match.group(1)  # Return the first matched date pattern
    return None

def parse_date(date_str):
    """
    Try to parse the detected expiry date string into a datetime object.
    """
    # Define different date formats that we might encounter
    date_formats = ['%d/%m/%Y', '%Y/%m/%d', '%d-%m-%Y', '%Y-%m-%d', '%d %b %Y', '%d %B %Y']
    for date_format in date_formats:
        try:
            return datetime.strptime(date_str, date_format)
        except ValueError:
            continue
    return None

def check_if_expired(expiry_date):
    """
    Check if the product is expired based on the expiry date.
    """
    if expiry_date:
        parsed_expiry_date = parse_date(expiry_date)
        if parsed_expiry_date:
            current_date = datetime.now()
            if parsed_expiry_date < current_date:
                return True  # Product is expired
            else:
                return False  # Product is not expired
        else:
            return "Invalid date format"
    return "No expiry date found"

# Home Page with 4 buttons
@app.route('/')
def index():
    return render_template('index.html')

# Route for Text Extraction using PaddleOCR
@app.route('/text_extraction', methods=['GET', 'POST'])
def text_extraction():
    if request.method == 'POST':
        # Get the uploaded file
        image_file = request.files['image']
        
        # Upload to Cloudinary
        upload_result = cloudinary.uploader.upload(image_file)
        
        # Get the URL of the uploaded image
        image_url = upload_result['url']
        
        # Perform OCR using the PaddleOCR function
        extracted_text = perform_ocr_on_image(image_url)
        
        # Return the result to the user
        return render_template('upload.html', result=extracted_text, image_url=image_url)
    
    return render_template('upload.html', title="Text Extraction")

# Route for Expiry Date (you can integrate OCR logic similarly if needed)
@app.route('/expiry_date', methods=['GET', 'POST'])
def expiry_date():
    if request.method == 'POST':
        image_file = request.files['image']
        
        # Upload to Cloudinary
        upload_result = cloudinary.uploader.upload(image_file)
        
        # Get the URL of the uploaded image
        image_url = upload_result['url']

        # Add expiry date extraction logic here using the image URL
        extracted_text = perform_ocr_on_image(image_url)

        expiry_date = find_expiry_date(extracted_text)
        if expiry_date:
            is_expired = check_if_expired(expiry_date)
            if isinstance(is_expired, bool):
                result="Is the product expired?", "Yes" if is_expired else "No"
            else:
                result=is_expired
        else:
            result="Expiry date not found"


        return render_template('upload.html', result=result,image_url=image_url)
    return render_template('upload.html', title="Expiry Date Detection")

# Route for Freshness Detection (you can integrate similar logic for freshness)
@app.route('/freshness', methods=['GET', 'POST'])
def freshness():
    if request.method == 'POST':
        image_file = request.files['image']
        
        # Upload to Cloudinary
        upload_result = cloudinary.uploader.upload(image_file)
        
        # Get the URL of the uploaded image
        image_url = upload_result['url']

        # Add freshness detection logic here using the image URL
        result = f"Freshness status detected in the image at URL: {image_url}"

        return render_template('upload.html', result=result)
    return render_template('upload.html', title="Freshness Detection")

import os
import torch
from flask import Flask, request, render_template
from matplotlib import pyplot as plt
import cv2
from werkzeug.utils import secure_filename

app = Flask(__name__)

# YOLOv5 model loading for object detection
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Set the upload folder path and allowed extensions
UPLOAD_FOLDER = 'uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Route for object detection (can be adapted for freshness detection)
@app.route('/detect', methods=['GET', 'POST'])
def detect():
    if request.method == 'POST':
        # Check if the image is part of the request
        if 'image' not in request.files:
            return "No file uploaded"
        
        image_file = request.files['image']
        if image_file.filename == '':
            return "No selected file"
        
        # Save the uploaded image
        filename = secure_filename(image_file.filename)
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        image_file.save(image_path)

        # Perform object detection using YOLOv5
        results = model(image_path)
        num_objects = len(results.xyxy[0])  # Number of objects detected

        # Render the bounding boxes on the image
        img = cv2.imread(image_path)
        results.render()  # Update the image with bounding boxes
        
        # Convert BGR image to RGB for Matplotlib display
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Save the result image (with bounding boxes)
        result_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'result_' + filename)
        cv2.imwrite(result_image_path, img)

        # Plot the image with bounding boxes using Matplotlib (optional)
        plt.imshow(img_rgb)
        plt.axis('off')
        plt.savefig(os.path.join(app.config['UPLOAD_FOLDER'], 'plot_' + filename))  # Save the plotted image
        
        # Return detection result and image in response
        return render_template('result.html', image_url=result_image_path, num_objects=num_objects)
    
    return render_template('upload.html')

# Flask entry point
if __name__ == '__main__':
    app.run(debug=True)