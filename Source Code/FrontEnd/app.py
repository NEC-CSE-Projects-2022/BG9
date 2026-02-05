import os
from datetime import datetime
from werkzeug.utils import secure_filename
from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory

import numpy as np
from PIL import Image
import cv2

import torch
import timm
from torchvision import transforms


# ---------------- Flask Setup ----------------
app = Flask(__name__)
app.secret_key = "6f9d1b8a0a21cba12d934caabb3e72ac7a9937ef41b78a4e9d32769d2cf500af"   # your secure key


BASE_DIR = os.path.dirname(__file__)
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
MODEL_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(UPLOAD_DIR, exist_ok=True)


# ---------------- Allowed Extensions ----------------
ALLOWED_EXT = {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}


# ---------------- Load Model (SwinV2 Tiny) ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = timm.create_model("swinv2_tiny_window8_256", pretrained=False, num_classes=2)
ckpt_path = os.path.join(MODEL_DIR, "checkpoint_epoch36.pt")

checkpoint = torch.load(ckpt_path, map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
model.eval()

val_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])


# ---------------- Helper Functions ----------------
def allowed(filename: str) -> bool:
    _, ext = os.path.splitext(filename.lower())
    return ext in ALLOWED_EXT


def ensure_grayscale(pil_img: Image.Image) -> Image.Image:
    if pil_img.mode != "L":
        return pil_img.convert("L")
    return pil_img


def filenames_look_ok(amp_name: str, phase_name: str) -> bool:
    base_a = os.path.splitext(amp_name)[0].upper()
    base_p = os.path.splitext(phase_name)[0].upper()
    return ("_A" in base_a) and ("_P" in base_p)


def fuse_hsl_exact(amp_img: Image.Image, phase_img: Image.Image) -> Image.Image:

    amp = np.array(ensure_grayscale(amp_img), dtype=np.float32)
    phase = np.array(ensure_grayscale(phase_img), dtype=np.float32)

    # Fix: Resize if shapes don't match
    if amp.shape != phase.shape:
        phase = cv2.resize(
            phase,
            (amp.shape[1], amp.shape[0]),
            interpolation=cv2.INTER_LINEAR
        )

    H = (amp / 255.0) * 360.0
    S = (phase / 255.0)
    L = np.full_like(S, 0.5, dtype=np.float32)

    hls = cv2.merge([H.astype(np.float32), L.astype(np.float32), S.astype(np.float32)])
    rgb = cv2.cvtColor(hls, cv2.COLOR_HLS2RGB_FULL)
    rgb = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)

    return Image.fromarray(rgb)


def predict_from_pair(amp_img: Image.Image, phase_img: Image.Image):

    fused = fuse_hsl_exact(amp_img, phase_img)
    tensor = val_transform(fused).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    prob_non = float(probs[0])
    prob_micro = float(probs[1])

    confidence = max(prob_non, prob_micro)

    # STRICT UNCERTAIN RULE
    if confidence >= 0.99:
        label = "Microplastic" if prob_micro > prob_non else "Non-Microplastic"
    else:
        label = "Uncertain"

    return label, prob_micro, fused



# ---------------- ROUTES ----------------

# HOME PAGE
@app.route("/")
@app.route("/home")
def home():
    return render_template("home.html")


# ABOUT PAGE
@app.route("/about")
def about():
    return render_template("about.html")


# PREDICT PAGE
@app.route("/predict", methods=["GET", "POST"])
def predict():

    # If page opened normally (GET)
    if request.method == "GET":
        return render_template("predict.html")

    # If form submitted (POST)
    amp_file = request.files.get("amp_file")
    phase_file = request.files.get("phase_file")

    if not amp_file or not phase_file:
        flash("Please upload both Amplitude and Phase images.")
        return redirect(url_for("predict"))

    if not (allowed(amp_file.filename) and allowed(phase_file.filename)):
        flash("Unsupported file type. Use BMP/PNG/JPG/TIFF.")
        return redirect(url_for("predict"))

    if not filenames_look_ok(amp_file.filename, phase_file.filename):
        flash("Incorrect filenames! Amplitude should contain '_A' & Phase should contain '_P'.")
        return redirect(url_for("predict"))

    # Save uploaded files
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    amp_name = secure_filename(f"{stamp}_A{os.path.splitext(amp_file.filename)[1].lower()}")
    phase_name = secure_filename(f"{stamp}_P{os.path.splitext(phase_file.filename)[1].lower()}")

    amp_path = os.path.join(UPLOAD_DIR, amp_name)
    phase_path = os.path.join(UPLOAD_DIR, phase_name)

    amp_file.save(amp_path)
    phase_file.save(phase_path)

    # Open images
    amp_img = ensure_grayscale(Image.open(amp_path))
    phase_img = ensure_grayscale(Image.open(phase_path))

    # Run prediction
    label, prob_micro, fused_img = predict_from_pair(amp_img, phase_img)

    fused_url = None
    prob_to_show = None

    # Only show image + probability if NOT uncertain
    if label != "Uncertain":
        fused_name = f"{stamp}_HSL.png"
        fused_path = os.path.join(UPLOAD_DIR, fused_name)
        fused_img.save(fused_path)
        fused_url = url_for("get_upload", filename=fused_name)
        prob_to_show = round(prob_micro, 4)

    return render_template(
        "result.html",
        pred_label=label,
        prob=prob_to_show,
        fused_url=fused_url
    )


# SERVE SAVED FILES
@app.route("/uploads/<path:filename>")
def get_upload(filename):
    return send_from_directory(UPLOAD_DIR, filename)



# HEALTH CHECK
@app.route("/health")
def health():
    return {"status": "ok", "device": str(device)}



# RUN APP
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
