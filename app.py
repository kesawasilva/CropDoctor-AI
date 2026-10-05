import gradio as gr
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import sys

# 1. Setup device and clean class names
class_names = ['Bacterial Leaf Blight', 'Brown Spot', 'Healthy', 'Leaf Blast', 'Leaf Scald', 'Narrow Brown Spot']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Path to your saved weights
weights_path = "rice_disease_mobilenetv3_6class.pth"

# Safe check: Make sure the weights file exists
if not os.path.exists(weights_path):
    print(f"❌ Error: '{weights_path}' not found!")
    print("Please make sure the trained model weights file is in the same folder.")
    sys.exit(1)

# 3. Rebuild model structure and load weights
model = models.mobilenet_v3_large()
model.classifier[3] = nn.Linear(model.classifier[3].in_features, len(class_names))

# map_location allows users without a GPU to run this smoothly on their CPU
model.load_state_dict(torch.load(weights_path, map_location=device))
model = model.to(device).eval()
print("⚡ AI Model loaded successfully and ready for inference!")

# 4. Image Preprocessing
app_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 5. Prediction function
def classify_leaf(image):
    if image is None:
        return "Please upload an image."
    
    # Convert numpy array from Gradio interface to PIL Image
    pil_img = Image.fromarray(image.astype('uint8'), 'RGB')
    tensor_img = app_transform(pil_img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(tensor_img)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
    
    # Return dictionary mapping class name to probability score
    return {class_names[i]: float(probabilities[i]) for i in range(len(class_names))}

# 6. Build the UI Layout
ui = gr.Interface(
    fn=classify_leaf,
    inputs=gr.Image(),
    outputs=gr.Label(num_top_classes=3),
    title="🌾 Rice Leaf Disease Identifier",
    description="Upload a photo of a rice leaf to diagnose its health status instantly using deep learning.",
    theme="soft"
)

# 7. Launch the app locally
if __name__ == "__main__":
    ui.launch(share=False)