import os
import cv2
import json
import hashlib
import pandas as pd
from datetime import datetime

# === CONFIG ===
CSV_PATH = "./processed/fd_features.csv"
IMAGE_FOLDER = "./processed/edges"  # or textures, depending on use case
CERT_DIR = "./certificates"

# --- Load FD Data from CSV ---
def load_fd_data_from_csv(csv_path):
    df = pd.read_csv(csv_path)

    # Normalize filename
    df['filename'] = df['filename'].str.strip().str.lower()

    fd_data = {
        row['filename']: {
            "geometrical_fd": row['fd_geometrical'],
            "textural_fd": row['fd_textural'],
            "health_status": row.get('health_status', None)
        }
        for _, row in df.iterrows()
    }
    return fd_data

# --- Generate Leaf Certificate with Identity Hash ---
def generate_leaf_certificate(image_path, meta):
    filename = os.path.basename(image_path)

    # Read and hash image content
    with open(image_path, "rb") as f:
        image_bytes = f.read()
    image_hash = hashlib.sha256(image_bytes).hexdigest()

    fd_string = f"{meta['geometrical_fd']:.6f}_{meta['textural_fd']:.6f}"
    combined = image_hash + fd_string
    identity_hash = hashlib.sha256(combined.encode()).hexdigest()

    certificate = {
        "filename": filename,
        "geometrical_fd": meta['geometrical_fd'],
        "textural_fd": meta['textural_fd'],
        "health_status": meta.get("health_status"),
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "image_hash": image_hash,
        "identity_hash": identity_hash,
        "metadata": {
            "creator": "FD-Driven Crypto Leaf System",
            "notes": "Cryptographic certificate binding FD biometric and image hash."
        }
    }

    # Save JSON
    os.makedirs(CERT_DIR, exist_ok=True)
    cert_path = os.path.join(CERT_DIR, os.path.splitext(filename)[0] + ".json")
    with open(cert_path, "w") as json_file:
        json.dump(certificate, json_file, indent=4)

    return certificate

# --- Main Processing Loop ---
def process_all_images(image_folder, fd_data):
    for image_filename in os.listdir(image_folder):
        if not image_filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        image_path = os.path.join(image_folder, image_filename)
        normalized_name = image_filename.strip().lower()

        if normalized_name in fd_data:
            meta = fd_data[normalized_name]
            cert = generate_leaf_certificate(image_path, meta)
            print(json.dumps(cert, indent=2))
        else:
            print(f"[SKIP] No FD data found for: {image_filename}")

# === RUN ===
if __name__ == "__main__":
    fd_data = load_fd_data_from_csv(CSV_PATH)
    process_all_images(IMAGE_FOLDER, fd_data)
