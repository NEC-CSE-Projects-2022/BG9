import torch
import timm

# Path to your model checkpoint
ckpt_path = 'models/checkpoint_epoch36.pt'

# Create the exact model architecture you used in training
# (binary classification, 2 output classes)
model = timm.create_model("swinv2_tiny_window8_256", pretrained=False, num_classes=2)

# Load checkpoint
checkpoint = torch.load(ckpt_path, map_location='cpu')
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Dummy input (same size as your training images)
dummy = torch.randn(1, 3, 256, 256)
out = model(dummy)

print("Output shape:", out.shape)
