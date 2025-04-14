import os
from flask import Flask, render_template, request, jsonify
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms, models
import pandas as pd

# Load the disease and supplement information
disease_info = pd.read_csv('disease_info.csv', encoding='cp1252')
supplement_info = pd.read_csv('supplement_info.csv', encoding='cp1252')

# Define the model path
model_path = 'App/models/ResNet50.pt'

# Load the pre-trained ResNet50 model (ImageNet weights) and adjust for binary classification
try:
    # Load ResNet50 model with ImageNet pre-trained weights
    model = models.resnet50(weights='IMAGENET1K_V1')

    # Load the saved state_dict for the model (if it exists)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))  # Load custom weights if available
        print(f"Model weights loaded successfully from {model_path}")
    else:
        print("No custom model weights found, using pre-trained ResNet50 with ImageNet weights")

    # Adjust the final fully connected layer for binary classification (2 output classes)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.eval()  # Set the model to evaluation mode
except Exception as e:
    print(f"Error loading the model: {e}")
    exit(1)

# Define the image transformation (resize and normalize)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ImageNet normalization
])

# Flask app setup
app = Flask(__name__)

@app.route('/')
def preloader_page():
    return render_template('preloader.html')

@app.route('/home')
def home_page():
    return render_template('home.html')

@app.route('/contact')
def contact():
    return render_template('contact-us.html')

@app.route('/index')
def ai_engine_page():
    return render_template('index.html')

@app.route('/mobile-device')
def mobile_device_detected_page():
    return render_template('mobile-device.html')
@app.route('/submit', methods=['POST'])
def submit():
    try:
        # Get the uploaded image
        image = request.files['image']
        filename = image.filename
        file_path = os.path.join('App/static/uploads', filename)

        # Save the uploaded image
        os.makedirs(os.path.dirname(file_path), exist_ok=True)  # Ensure the directory exists
        image.save(file_path)

        # Open the image and apply transformations
        img = Image.open(file_path).convert('RGB')
        img = transform(img).unsqueeze(0)  # Add batch dimension

        # Make a prediction
        with torch.no_grad():
            output = model(img)
            print(f"Raw output: {output}")  # Print raw logits
            probabilities = torch.nn.functional.softmax(output, dim=1)
            print(f"Probabilities: {probabilities}")
        pred_idx = torch.argmax(probabilities, dim=1).item()


        # Retrieve data for the prediction
        title = disease_info['disease_name'][pred_idx]
        description = disease_info['description'][pred_idx]
        prevent = disease_info['Possible Steps'][pred_idx]
        image_url = disease_info['image_url'][pred_idx]
        supplement_name = supplement_info['supplement name'][pred_idx]
        supplement_image_url = supplement_info['supplement image'][pred_idx]
        supplement_buy_link = supplement_info['buy link'][pred_idx]

        # Render the result page with the prediction details
        return render_template(
            'submit.html', title=title, desc=description, prevent=prevent,
            image_url=image_url, sname=supplement_name,
            simage=supplement_image_url, buy_link=supplement_buy_link
        )
    except Exception as e:
        return jsonify({'error': str(e)})


@app.route('/market', methods=['GET', 'POST'])
def market():
    return render_template('market.html', supplement_image=list(supplement_info['supplement image']),
                        supplement_name=list(supplement_info['supplement name']), disease=list(disease_info['disease_name']), buy=list(supplement_info['buy link']))


if __name__ == '__main__':
    app.run(debug=True, port=5001)
