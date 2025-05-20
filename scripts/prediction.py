import os
import cv2
import hashlib
import pandas as pd
import joblib
import numpy as np
from datetime import datetime
from skimage.feature import graycomatrix, graycoprops

# === CONFIG ===
MODEL_PATH = "./processed/leaf_fd_classifier.pkl"
LABEL_ENCODER_PATH = "./processed/leaf_fd_classifier_label_encoder.pkl"
LOG_PATH = "./processed/prediction_log.csv"

# === FD Feature Extraction Functions ===

def extract_features(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Image could not be loaded.")

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_bin = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    # Geometric FD
    def boxcount(img_bin, k):
        S = np.add.reduceat(
            np.add.reduceat(img_bin, np.arange(0, img_bin.shape[0], k), axis=0),
            np.arange(0, img_bin.shape[1], k), axis=1)
        return np.count_nonzero(S)

    sizes = 2 ** np.arange(1, int(np.log2(min(img_bin.shape))) - 1)
    counts = [boxcount(img_bin, k) for k in sizes]
    coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
    fd_geometrical = -coeffs[0]

    # GLCM Texture Features
    glcm = graycomatrix(img_gray, distances=[5], angles=[0], levels=256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    dissimilarity = graycoprops(glcm, 'dissimilarity')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]
    ASM = graycoprops(glcm, 'ASM')[0, 0]

    fd_textural = np.log1p(contrast + dissimilarity + homogeneity + energy + correlation + ASM)

    features = [
        fd_geometrical,
        fd_textural,
        contrast,
        dissimilarity,
        homogeneity,
        energy,
        correlation,
        ASM
    ]
    return features, fd_geometrical, fd_textural

# === Sustainability Functions ===

def calculate_sustainability_score(soil_health, water_usage, pesticide_usage, farming_techniques):
    # Normalize all inputs to be between 0 and 1
    normalized_soil_health = soil_health / 100
    normalized_water_usage = water_usage / 100
    normalized_pesticide_usage = pesticide_usage / 100
    normalized_farming_techniques = farming_techniques / 100

    # Weighted average to calculate sustainability score (adjust weights as needed)
    sustainability_score = (0.25 * normalized_soil_health +
                            0.25 * normalized_water_usage +
                            0.25 * (1 - normalized_pesticide_usage) +  # Inverse because less pesticide is better
                            0.25 * normalized_farming_techniques)

    # Return sustainability score (scaled to 100 for easier interpretation)
    return sustainability_score * 100

# Function to get farming data (this could come from sensors or manual input)
def get_farming_data():
    # Manual input for soil health, water usage, pesticide usage, and farming techniques
    print("Please enter the following sustainability data (0 to 100 scale):")
    
    soil_health = float(input("Enter soil health (0-100): "))
    water_usage = float(input("Enter water usage (0-100): "))
    pesticide_usage = float(input("Enter pesticide usage (0-100): "))
    farming_techniques = float(input("Enter sustainability of farming techniques (0-100): "))
    
    return soil_health, water_usage, pesticide_usage, farming_techniques

# === Main prediction flow ===

def main():
    image_path = input("📂 Enter the full path to the leaf image: ").strip()

    if not os.path.exists(image_path):
        print("❌ File does not exist.")
        return

    # Load model
    clf = joblib.load(MODEL_PATH)
    le = joblib.load(LABEL_ENCODER_PATH)

    # Extract features
    try:
        features, fd_geo, fd_text = extract_features(image_path)
    except Exception as e:
        print(f"❌ Error extracting features: {e}")
        return

    # Get farming data and calculate sustainability score
    soil_health, water_usage, pesticide_usage, farming_techniques = get_farming_data()
    sustainability_score = calculate_sustainability_score(soil_health, water_usage, pesticide_usage, farming_techniques)

    # Predict plant health based on FD and sustainability score
    features_array = np.array(features).reshape(1, -1)
    prediction = clf.predict(features_array)[0]
    predicted_label = le.inverse_transform([prediction])[0]

    # Combine FD and sustainability data into final prediction score (for example)
    final_health_score = (fd_geo * 0.5) + (fd_text * 0.5) + (sustainability_score * 0.5)

    print("\n Prediction Result")
    print("----------------------------")
    print(f" Image: {os.path.basename(image_path)}")
    print(f" Geometrical FD: {fd_geo:.4f}")
    print(f" Textural FD: {fd_text:.4f}")
    print(f" Predicted Health Status: {predicted_label}")
    print(f" Sustainability Score: {sustainability_score:.2f}")
    print(f" Final Health Score: {final_health_score:.2f}")

    # Optional: Log the prediction
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    log_row = {
        "timestamp": datetime.utcnow().isoformat(),
        "image": os.path.basename(image_path),
        "fd_geometrical": fd_geo,
        "fd_textural": fd_text,
        "predicted_health": predicted_label,
        "sustainability_score": sustainability_score,
        "final_health_score": final_health_score
    }

    if os.path.exists(LOG_PATH):
        pd.DataFrame([log_row]).to_csv(LOG_PATH, mode='a', index=False, header=False)
    else:
        pd.DataFrame([log_row]).to_csv(LOG_PATH, index=False)

    print(f"\n📁 Logged to: {LOG_PATH}")

# === Run ===
if __name__ == "__main__":
    main()
