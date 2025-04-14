import torch
from torchvision import models

# Load the pre-trained ResNet50 model
model = models.resnet50(pretrained=True)

# Save the model's state dictionary
torch.save(model.state_dict(), 'models/ResNet50.pt')

print("Pre-trained ResNet50 model saved at 'App/models/ResNet50.pt'")
