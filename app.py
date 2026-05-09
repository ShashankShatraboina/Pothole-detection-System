from flask import Flask, render_template, request
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import os
import torch.nn as nn

app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Define class names
classes = ["good", "poor", "satisfactory", "very_poor"]

def load_model(model_path, device):
    model = models.resnet101(weights=None)
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, len(classes))
    )
    state_dict = torch.load(model_path, map_location=device)
    if "module." in list(state_dict.keys())[0]:
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k.replace("module.", "")
            new_state_dict[name] = v
        state_dict = new_state_dict
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

# Define Image Transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def predict_image(model, image_path, device):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image)
        _, predicted_class = torch.max(output, 1)
        predicted_label = classes[predicted_class.item()]
    return predicted_label

# Load model
model_path = "model/best_resnet101_model.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if os.path.exists(model_path):
    model = load_model(model_path, device)
else:
    model = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('index.html', error="No file uploaded")
    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', error="No selected file")
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)
    
    if model:
        prediction = predict_image(model, file_path, device)
        return render_template('index.html', image_path=file_path, prediction=prediction)
    else:
        return render_template('index.html', error="Model file not found")

if __name__ == '__main__':
    app.run(debug=True)
