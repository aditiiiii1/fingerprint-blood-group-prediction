from flask import Flask, request, render_template, jsonify
from PIL import Image
import torch
from torchvision import transforms, models
import os
import requests

app = Flask(__name__)

blood_group_labels = ['A', 'A-', 'AB', 'AB-', 'B', 'B-', 'O', 'O-']
IMG_SIZE = 224
MODEL_PATH = 'bloodgroup_model.pt'  # Make sure this matches your model filename

# Load model only once
def get_model():
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, len(blood_group_labels))
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    model.eval()
    return model

model = get_model()

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

def predict_image(img_path):
    image = Image.open(img_path)
    input_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(input_tensor)
    predicted_idx = torch.argmax(outputs, 1).item()
    confidence = torch.softmax(outputs, 1)[0][predicted_idx].item()
    predicted_label = blood_group_labels[predicted_idx]
    return predicted_label, round(float(confidence), 3)

def fetch_external_image(external_url, save_path='external_image.png'):
    r = requests.get(external_url)
    if r.status_code == 200:
        with open(save_path, 'wb') as f:
            f.write(r.content)
        return save_path
    else:
        return None

@app.route('/', methods=['GET', 'POST'])
def index():
    predicted = None
    confidence = None
    err = None
    if request.method == 'POST':
        # Option 1: upload file
        if 'image' in request.files and request.files['image'].filename:
            img_file = request.files['image']
            img_path = "uploaded_image.png"
            img_file.save(img_path)
            predicted, confidence = predict_image(img_path)
        # Option 2: supply an external image URL
        elif 'external_url' in request.form and request.form['external_url']:
            img_path = fetch_external_image(request.form['external_url'])
            if img_path:
                predicted, confidence = predict_image(img_path)
            else:
                err = "Could not fetch image from URL"
        else:
            err = "No image or URL provided!"
    return render_template('index.html',
                            predicted=predicted,
                            confidence=confidence,
                            error=err,
                            predicted_group=predicted)

@app.route('/api/predict', methods=['POST'])
def api_predict():
    if 'image' in request.files:
        img_file = request.files['image']
        img_path = "uploaded_image.png"
        img_file.save(img_path)
    elif 'external_url' in request.form:
        img_path = fetch_external_image(request.form['external_url'])
        if img_path is None:
            return jsonify({'error': 'Could not fetch external image'}), 400
    else:
        return jsonify({'error': 'No image or external_url provided'}), 400

    predicted, confidence = predict_image(img_path)
    result = {
        "predicted_group": predicted,
        "confidence": confidence,
        "status": "OK"
    }
    return jsonify(result), 200

if __name__ == '__main__':
    app.run(debug=True)
