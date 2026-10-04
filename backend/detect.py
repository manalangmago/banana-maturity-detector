"""
detect.py
---------
Loads the trained YOLO model (best.pt) once, and runs detection on incoming images.

Since your model was trained to both find the banana AND classify its ripeness
(this is common when you train YOLO with ripeness stages as the class names,
e.g. "unripe", "ripe", "overripe"), this file is responsible ONLY for running
the model and returning the raw detection results (boxes, labels, confidence).

classification.py will take those raw results and turn them into something
clean/friendly to send to the frontend.
"""

import sys
import base64
import torch
import torch.nn as nn
import cv2
from ultralytics import YOLO
from PIL import Image
import io

# ---------------------------------------------------------------------------
# CBAM (Convolutional Block Attention Module) — custom layer used during
# training. This must match the ORIGINAL training notebook exactly, or the
# saved weights won't line up with the layer shapes.
#
# best.pt was saved while CBAM lived in "__main__" (the training notebook),
# so PyTorch's unpickler goes looking for CBAM in __main__ when loading the
# checkpoint — even in this file. We define it here and manually attach it
# to the __main__ module so the loader can find it.
# ---------------------------------------------------------------------------
class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, max(1, channels // reduction), 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(max(1, channels // reduction), channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return self.sigmoid(avg + max_out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        return self.sigmoid(self.conv(torch.cat([avg, max_out], dim=1)))


class CBAM(nn.Module):
    def __init__(self, c1=1024, reduction=16, kernel_size=7):
        super().__init__()
        self.reduction = reduction
        self.kernel_size = kernel_size
        self.ca = ChannelAttention(c1, reduction)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        # Auto-fix channel mismatch caused by YOLOv8 width scaling
        if x.shape[1] != self.ca.fc[0].in_channels:
            self.ca = ChannelAttention(x.shape[1], self.reduction).to(x.device)
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x


# Register CBAM into __main__ (matches how it was defined during training)
# AND into ultralytics' own modules, so the model-building code that looks
# up layer types by name can also find it.
_main_module = sys.modules["__main__"]
_main_module.CBAM = CBAM
_main_module.ChannelAttention = ChannelAttention
_main_module.SpatialAttention = SpatialAttention

import ultralytics.nn.modules as modules_pkg
import ultralytics.nn.tasks as tasks_module
modules_pkg.CBAM = CBAM
tasks_module.__dict__["CBAM"] = CBAM

# Load the model ONE time when the server starts (not on every request).
# Loading a model is slow/expensive, so this must live at module level.
MODEL_PATH = "model/best.pt"
model = YOLO(MODEL_PATH)


def run_detection(image_bytes: bytes, confidence_threshold: float = 0.4):
    """
    Runs the YOLO model on a single image.

    Args:
        image_bytes: raw bytes of the uploaded image (from the FastAPI upload)
        confidence_threshold: ignore detections below this confidence score

    Returns:
        A tuple: (detections, annotated_image_base64)

        detections is a list of dicts like:
        {
            "label": "ripe",         # class name from your model's training
            "confidence": 0.93,      # how sure the model is (0 to 1)
            "box": [x1, y1, x2, y2]  # pixel coordinates of the bounding box
        }

        annotated_image_base64 is a base64-encoded PNG (as a string) of the
        original image with bounding boxes + labels drawn on it — ready to
        send straight to the frontend and display in an <img> tag.
    """
    # Convert the raw bytes into something the model can read
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Run inference. YOLO returns a list (one entry per image); we only sent one.
    results = model.predict(source=image, conf=confidence_threshold, verbose=False)
    result = results[0]

    # --- Extract the raw detection data (same as before) ---
    detections = []
    for box in result.boxes:
        class_id = int(box.cls[0])
        label = model.names[class_id]          # e.g. "ripe", "unripe", "overripe"
        confidence = float(box.conf[0])
        x1, y1, x2, y2 = box.xyxy[0].tolist()   # bounding box corners

        detections.append({
            "label": label,
            "confidence": round(confidence, 4),
            "box": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
        })

    # --- Build the annotated image ---
    # result.plot() returns a numpy array (BGR, OpenCV's default color order)
    # with the boxes/labels already drawn on it.
    annotated_bgr = result.plot()

    # Encode that array into PNG bytes, then base64 so it can travel as JSON text.
    success, buffer = cv2.imencode(".png", annotated_bgr)
    if not success:
        raise RuntimeError("Failed to encode annotated image.")

    annotated_image_base64 = base64.b64encode(buffer).decode("utf-8")

    return detections, annotated_image_base64