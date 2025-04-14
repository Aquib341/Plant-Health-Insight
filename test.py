import torch
from torchvision.models import resnet50
import torch.nn as nn

# Load the pre-trained ResNet50 model
model_path = 'models/ResNet50.pt'
try:
    model = resnet50()  # Initialize ResNet50 architecture
    model.fc = nn.Linear(model.fc.in_features, 2)  # Update the FC layer for 2 classes
    model.load_state_dict(torch.load(model_path))  # Load weights
    model.eval()
    print(f"Model loaded successfully from {model_path}")
except FileNotFoundError as fnf_error:
    print(f"Model file not found: {fnf_error}")
    exit(1)
except RuntimeError as load_error:
    print(f"Error loading the model: {load_error}")
    exit(1)
