import gradio as gr
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

# Define class mappings for each crop
CROP_CLASSES = {
    "Rice": [
        "Bacterial Leaf Blight", 
        "Brown Spot", 
        "Healthy", 
        "Leaf Blast", 
        "Leaf Scald", 
        "Narrow Brown Spot"
    ],
    "Chili Pepper": [
        "Healthy", 
        "Leaf Curl", 
        "Leaf Spot", 
        "Whitefly", 
        "Yellowish"
    ],
    "Tea": [
        "Algal Leaf", 
        "Anthracnose", 
        "Bird Eye Spot", 
        "Brown Blight", 
        "Healthy", 
        "Red Leaf Spot"
    ]
}

# Image transformation pipeline (standard for MobileNetV3)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def load_model(crop_name):
    """Loads the appropriate pre-trained MobileNetV3 model based on crop selection."""
    if crop_name == "Rice":
        model_path = "rice_disease_mobilenetv3_6class.pth"
        num_classes = len(CROP_CLASSES["Rice"])
    elif crop_name == "Chili Pepper":
        model_path = "chili_disease_mobilenetv3.pth"
        num_classes = len(CROP_CLASSES["Chili Pepper"])
    elif crop_name == "Tea":
        model_path = "tea_disease_mobilenetv3.pth"
        num_classes = len(CROP_CLASSES["Tea"])
    else:
        raise ValueError("Invalid crop selected.")

    # Initialize MobileNetV3-Large architecture
    model = models.mobilenet_v3_large(weights=None)
    num_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(num_features, num_classes)
    
    # Load trained weights
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'), weights_only=True))
    model.eval()
    return model

def predict_disease(crop_type, image):
    """Performs inference and returns class probabilities/predictions."""
    if image is None:
        return "Please upload an image first."
    
    try:
        # Load model for the selected crop
        model = load_model(crop_type)
        class_names = CROP_CLASSES[crop_type]

        # Preprocess image
        image = image.convert("RGB")
        input_tensor = transform(image).unsqueeze(0) # Add batch dimension

        # Predict
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            
        # Format results into a dictionary for Gradio's Label component
        confidences = {class_names[i]: float(probabilities[i]) for i in range(len(class_names))}
        return confidences

    except Exception as e:
        return f"Error during prediction: {str(e)}"

# Build Gradio UI Dashboard
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🌱 CropDoctor-AI: Multi-Crop Disease Diagnostic Platform")
    gr.Markdown("Select your crop type, upload a leaf image, and let our AI diagnose potential diseases instantly.")
    
    with gr.Row():
        with gr.Column(scale=1):
            crop_dropdown = gr.Dropdown(
                choices=["Rice", "Chili Pepper", "Tea"], 
                label="Select Crop Type", 
                value="Rice"
            )
            image_input = gr.Image(type="pil", label="Upload Leaf Image")
            submit_btn = gr.Button("Diagnose Disease", variant="primary")
            
        with gr.Column(scale=1):
            output_label = gr.Label(label="Diagnosis Results & Confidence Scores")

    submit_btn.click(
        fn=predict_disease, 
        inputs=[crop_dropdown, image_input], 
        outputs=output_label
    )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)