import json
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import base64
import io
import os

# -----------------------
# Load model at startup
# -----------------------

MODEL_PATH = os.path.join("model", "efficientnet_b0_best.pth")
device = "cpu"

# Build model architecture
model = models.efficientnet_b0(weights=None)
model.classifier = nn.Sequential(
    nn.Dropout(p=0.2),
    nn.Linear(model.classifier[1].in_features, 2)
)

# Load checkpoint
checkpoint = torch.load(MODEL_PATH, map_location=device)

# Handle both raw state_dict and dict with "model_state_dict"
state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
model.load_state_dict(state_dict)

model.eval()
model.to(device)

# -----------------------
# Image transforms
# -----------------------

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

print("************** Lambda container started ****************")

# -----------------------
# Lambda handler
# -----------------------

def handler(event, context):
    print("---------- Event received ----------")
    print("EVENT:", json.dumps(event))

    try:
        # Case 1: HTTP API (v2) with double-encoded body
        if "body" in event:
            raw_body = event["body"]

            # First decode: event["body"] is a JSON string
            if isinstance(raw_body, str):
                body = json.loads(raw_body)
            else:
                body = raw_body

            # Second decode: body["body"] is another JSON string
            if isinstance(body, dict) and "body" in body:
                inner = body["body"]
                if isinstance(inner, str):
                    body = json.loads(inner)

        # Case 2: HTTP API rawBody
        elif "rawBody" in event:
            body = json.loads(event["rawBody"])

        # Case 3: Direct invocation
        elif "image" in event:
            body = event

        else:
            return {
                "statusCode": 400,
                "body": json.dumps({"error": "Invalid request format"})
            }

        # Validate
        if "image" not in body:
            return {
                "statusCode": 400,
                "body": json.dumps({"error": "No image provided"})
            }

        # Decode image
        image_bytes = base64.b64decode(body["image"])
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Preprocess
        x = transform(image).unsqueeze(0).to(device)

        # Inference
        with torch.no_grad():
            outputs = model(x)
            probs = torch.softmax(outputs, dim=1)
            pred = torch.argmax(probs, dim=1).item()
            confidence = probs[0][pred].item()

        return {
            "statusCode": 200,
            "body": json.dumps({
                "prediction": int(pred),
                "confidence": float(confidence)
            })
        }

    except Exception as e:
        print("ERROR:", str(e))
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }
