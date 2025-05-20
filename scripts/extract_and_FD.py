import os
import cv2
import numpy as np
import pandas as pd
from skimage.feature import graycomatrix, graycoprops

# ========== FOLDERS ==========
DATA_DIR = "./PlantVillage"
EDGE_DIR = "./processed/edges"
TEXTURE_DIR = "./processed/textures"
CSV_PATH = "./processed/fd_features.csv"

# ========== UTILITY FUNCTIONS ==========

def ensure_dirs():
    os.makedirs(EDGE_DIR, exist_ok=True)
    os.makedirs(TEXTURE_DIR, exist_ok=True)

def extract_edges(img_gray):
    blurred = cv2.GaussianBlur(img_gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    return edges

def enhance_veins(img_gray):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    tophat = cv2.morphologyEx(img_gray, cv2.MORPH_TOPHAT, kernel)
    return tophat

def boxcount(img_bin, k):
    S = np.add.reduceat(
        np.add.reduceat(img_bin, np.arange(0, img_bin.shape[0], k), axis=0),
        np.arange(0, img_bin.shape[1], k), axis=1)
    return np.count_nonzero(S)

# ========== MAIN PROCESS ==========
def process_images():
    ensure_dirs()
    rows = []

    valid_exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

    for root, dirs, files in os.walk(DATA_DIR):
        for fname in files:
            if not fname.lower().endswith(valid_exts):
                continue

            img_path = os.path.join(root, fname)
            print(f"Processing: {img_path}")
            img = cv2.imread(img_path)
            if img is None:
                print(f"Could not load {img_path}")
                continue

            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_bin = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

            # Geometrical Fractal Dimension (FD) using Box Counting
            sizes = 2 ** np.arange(1, int(np.log2(min(img_bin.shape))) - 1)
            counts = [boxcount(img_bin, k) for k in sizes]
            coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
            fd_geometrical = -coeffs[0]

            # Save edge image
            edges = extract_edges(img_gray)
            edge_path = os.path.join(EDGE_DIR, fname)
            cv2.imwrite(edge_path, edges)

            # Save texture image
            veins = enhance_veins(img_gray)
            texture_path = os.path.join(TEXTURE_DIR, fname)
            cv2.imwrite(texture_path, veins)

            # GLCM Texture Features
            glcm = graycomatrix(img_gray, distances=[5], angles=[0], levels=256, symmetric=True, normed=True)
            contrast = graycoprops(glcm, 'contrast')[0, 0]
            dissimilarity = graycoprops(glcm, 'dissimilarity')[0, 0]
            homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
            energy = graycoprops(glcm, 'energy')[0, 0]
            correlation = graycoprops(glcm, 'correlation')[0, 0]
            ASM = graycoprops(glcm, 'ASM')[0, 0]

            # Textural Fractal Dimension (FD) from GLCM
            fd_textural = np.log1p(contrast + dissimilarity + homogeneity + energy + correlation + ASM)

            # ✅ Extract health_status from the parent directory name
            health_status = os.path.basename(root)

            rows.append({
                "filename": fname,
                "fd_geometrical": fd_geometrical,
                "fd_textural": fd_textural,
                "contrast": contrast,
                "dissimilarity": dissimilarity,
                "homogeneity": homogeneity,
                "energy": energy,
                "correlation": correlation,
                "ASM": ASM,
                "health_status": health_status  # ✅ Added label here
            })

    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(CSV_PATH, index=False)
        print(f"\n✅ Done. CSV saved at: {CSV_PATH}")
    else:
        print("⚠️ No valid images found or processed.")

# ========== RUN ==========
if __name__ == "__main__":
    process_images()
