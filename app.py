import gradio as gr
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# Define class mappings for both crops
CLASS_NAMES_RICE = [
    "Bacterial Leaf Blight", 
    "Brown Spot", 
    "Healthy", 
    "Leaf Blast", 
    "Leaf Scald", 
    "Narrow Brown Spot"
]

CLASS_NAMES_CHILI = [
    "Healthy", 
    "Leaf Curl", 
    "Leaf Spot", 
    "Whitefly", 
    "Yellowish"
]

# Image preprocessing pipeline (Standard ImageNet normalization)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load models lazily or on startup
def load_model(model_path, num_classes):
    model = models.mobilenet_v3_large(weights=None)
    num_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(num_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

# Load both models into memory
rice_model = load_model("rice_disease_mobilenetv3_6class.pth", len(CLASS_NAMES_RICE))
chili_model = load_model("chili_disease_mobilenetv3.pth", len(CLASS_NAMES_CHILI))

def predict_disease(image, crop_type):
    if image is None:
        return "Please upload an image."
    
    # Select model and classes based on crop choice
    if crop_type == "Rice":
        model = rice_model
        classes = CLASS_NAMES_RICE
    elif crop_type == "Chili Pepper":
        model = chili_model
        classes = CLASS_NAMES_CHILI
    else:
        return "Invalid crop selection."

    # Preprocess image
    image_tensor = transform(image).unsqueeze(0)

    # Inference
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        
    # Format top predictions into a dictionary for Gradio Label component
    confidences = {classes[i]: float(probabilities[i]) for i in range(len(classes))}
    return confidences

# Gradio Interface UI
interface = gr.Interface(
    fn=predict_disease,
    inputs=[
        gr.Image(type="pil", label="Upload Leaf Image"),
        gr.Dropdown(choices=["Rice", "Chili Pepper"], label="Select Crop Type", value="Rice")
    ],
    outputs=gr.Label(num_top_classes=3, label="Diagnosis Result"),
    title="🌱 CropDoctor-AI: Multi-Crop Disease Diagnostic Platform",
    description="Upload a leaf image of either Rice or Chili Pepper to detect diseases instantly using customized MobileNetV3 AI models.",
    examples=[
        ["testImage1.jpg", "Rice"],
        ["testImage2.jpg", "Rice"]
    ] if "testImage1.jpg" in __import__("os").listdir(".") else None
)

if __name__ == "__main__":
    interface.launch()