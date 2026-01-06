from flask import Flask, render_template, request, jsonify
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v2
from torchvision import transforms
from PIL import Image
import os
import cv2
import numpy as np
import requests
from dotenv import load_dotenv
from groq import Groq
# Load environment variables
load_dotenv()

# -----------------------------
# Flask app
# -----------------------------
app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# -----------------------------
# Device
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Class labels
# -----------------------------
class_names = [
    "Mild",
    "Moderate",
    "No_DR",
    "Proliferate_DR",
    "Severe"
]

# -----------------------------
# Load MobileNetV2 model
# -----------------------------
device = torch.device("cpu")

mobilenet = mobilenet_v2(pretrained=False)
mobilenet.classifier[1] = nn.Linear(
    mobilenet.classifier[1].in_features, 5
)

checkpoint = torch.load("mobilenetv2_dr_checkpoint.pth", map_location=device)

if "state_dict" in checkpoint:
    mobilenet.load_state_dict(checkpoint["state_dict"])
else:
    mobilenet.load_state_dict(checkpoint)

mobilenet.to(device)
mobilenet.eval()

# -----------------------------
# Image preprocessing
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def preprocess_image(image):
    """Preprocess image for model inference"""
    image = image.convert("RGB")
    image = transform(image)
    image = image.unsqueeze(0)
    return image.to(device)

# -----------------------------
# Prediction function
# -----------------------------
def predict(image):
    """
    Perform DR stage prediction
    Returns: (class_name, confidence_score)
    """
    image = preprocess_image(image)

    with torch.no_grad():
        outputs = mobilenet(image)
        probs = F.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, 1)

    return class_names[pred.item()], conf.item()

# -----------------------------
# Grad-CAM Implementation
# -----------------------------
class GradCAM:
    """Generate Grad-CAM heatmap for model explainability"""
    
    def __init__(self, model):
        self.model = model
        self.feature_maps = None
        self.gradients = None
        
        # Hook the last convolutional layer
        self.target_layer = model.features[-1]
        self.target_layer.register_forward_hook(self.save_feature_maps)
        self.target_layer.register_full_backward_hook(self.save_gradients)
    
    def save_feature_maps(self, module, input, output):
        self.feature_maps = output
    
    def save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]
    
    def generate_cam(self, input_image, target_class=None):
        """Generate Grad-CAM heatmap"""
        self.model.eval()
        
        # Forward pass
        output = self.model(input_image)
        
        if target_class is None:
            target_class = output.argmax(dim=1)
        
        # Backward pass
        self.model.zero_grad()
        output[0, target_class].backward()
        
        # Generate CAM
        gradients = self.gradients.cpu().data.numpy()[0]
        feature_maps = self.feature_maps.cpu().data.numpy()[0]
        
        weights = np.mean(gradients, axis=(1, 2))
        cam = np.zeros(feature_maps.shape[1:], dtype=np.float32)
        
        for i, w in enumerate(weights):
            cam += w * feature_maps[i]
        
        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (224, 224))
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        
        return cam

def generate_gradcam_visualization(image_path):
    """
    Generate Grad-CAM heatmap and overlay
    Returns: (heatmap_path, overlay_path)
    """
    # Load and preprocess image
    image = Image.open(image_path)
    input_tensor = preprocess_image(image)
    
    # Generate Grad-CAM
    gradcam = GradCAM(mobilenet)
    cam = gradcam.generate_cam(input_tensor)
    
    # Load original image
    original_img = cv2.imread(image_path)
    original_img = cv2.resize(original_img, (224, 224))
    
    # Create heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    
    # Create overlay
    overlay = cv2.addWeighted(original_img, 0.6, heatmap, 0.4, 0)
    
    # Save images
    filename = os.path.basename(image_path)
    name, ext = os.path.splitext(filename)
    
    heatmap_path = os.path.join(app.config["UPLOAD_FOLDER"], f"{name}_heatmap{ext}")
    overlay_path = os.path.join(app.config["UPLOAD_FOLDER"], f"{name}_overlay{ext}")
    
    cv2.imwrite(heatmap_path, heatmap)
    cv2.imwrite(overlay_path, overlay)
    
    return heatmap_path, overlay_path

# -----------------------------
# Groq API Configuration
# -----------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
print("DEBUG → GROQ_API_KEY loaded:", bool(GROQ_API_KEY))

groq_client = Groq(api_key=GROQ_API_KEY)



# System prompt for DR-specific chatbot
DR_SYSTEM_PROMPT = """
You are a knowledgeable assistant specializing ONLY in Diabetic Retinopathy (DR).

You may answer questions related to:
- What is Diabetic Retinopathy
- DR stages (No DR, Mild, Moderate, Severe, Proliferative DR)
- Symptoms and risk factors
- Prevention and precautions for diabetic patients
- Diet recommendations (what to eat and avoid)
- Eye care tips for diabetics
- General information about diabetes and eye health

STRICT RULES:
- Do NOT provide emergency medical advice
- Do NOT diagnose diseases
- Do NOT recommend medications or dosages
- If asked about non-DR topics, politely redirect the user to DR-related questions
- Always advise consulting a healthcare professional

Tone: concise, empathetic, and easy to understand.
"""



def get_dr_chatbot_response(user_message):

    if not GROQ_API_KEY:
        return "⚠️ Chatbot service is not configured."

    if not user_message or not user_message.strip():
        return "Please ask a question related to Diabetic Retinopathy."

    try:
        completion = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",   # ✅ WORKING MODEL
            messages=[
                {"role": "system", "content": DR_SYSTEM_PROMPT},
                {"role": "user", "content": user_message.strip()}
            ],
            temperature=0.4,
            max_tokens=300
        )

        return completion.choices[0].message.content

    except Exception as e:
        print("CHATBOT ERROR:", e)
        return "❌ Chatbot service is temporarily unavailable."

# -----------------------------
# Routes
# -----------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    """Main page - handles image upload and prediction"""
    prediction = None
    confidence = None
    image_path = None

    if request.method == "POST":
        file = request.files.get("image")

        if file and file.filename:
            # Save uploaded image
            image_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(image_path)

            # Make prediction
            image = Image.open(image_path)
            prediction, confidence = predict(image)

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        image_path=image_path
    )

@app.route("/predict", methods=["POST"])
def predict_route():
    """
    API endpoint for DR prediction
    Returns JSON with prediction and confidence
    """
    try:
        file = request.files.get("image")
        
        if not file or not file.filename:
            return jsonify({"error": "No image uploaded"}), 400
        
        # Save image
        image_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(image_path)
        
        # Make prediction
        image = Image.open(image_path)
        prediction, confidence = predict(image)
        
        return jsonify({
            "success": True,
            "prediction": prediction,
            "confidence": round(confidence * 100, 2),
            "image_path": image_path
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/gradcam", methods=["POST"])
def gradcam_route():
    """
    Generate Grad-CAM visualization
    Returns JSON with paths to heatmap and overlay images
    """
    try:
        data = request.get_json()
        image_path = data.get("image_path")
        
        if not image_path or not os.path.exists(image_path):
            return jsonify({"error": "Image not found"}), 400
        
        # Generate Grad-CAM
        heatmap_path, overlay_path = generate_gradcam_visualization(image_path)
        
        return jsonify({
            "success": True,
            "heatmap": heatmap_path,
            "overlay": overlay_path
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_message = data.get("message", "")

    print("DEBUG → user_message:", user_message)

    reply = get_dr_chatbot_response(user_message)

    return jsonify({"reply": reply})


# -----------------------------
# Run app
# -----------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
