# LeafFD: Plant Disease Detection Using Fractal Dimension Analysis

A machine learning system that uses fractal dimension (FD) analysis and texture features to detect plant diseases from leaf images. The system combines geometric and textural fractal dimensions with sustainability metrics to provide comprehensive plant health assessment.

## 🌱 Overview

This project analyzes plant leaf images using advanced image processing techniques to extract fractal dimension features and texture characteristics. It employs machine learning to classify plant health status and includes sustainability scoring for comprehensive agricultural assessment.

## 📁 Project Structure

```
leafFD/
├── PlantVillage/                    # Dataset directory with plant images
│   ├── Pepper__bell___Bacterial_spot/
│   ├── Pepper__bell___healthy/
│   ├── Potato___Early_blight/
│   ├── Potato___Late_blight/
│   ├── Potato___healthy/
│   ├── Tomato_Bacterial_spot/
│   ├── Tomato_Early_blight/
│   ├── Tomato_Late_blight/
│   ├── Tomato_Leaf_Mold/
│   ├── Tomato_Septoria_leaf_spot/
│   ├── Tomato_Spider_mites_Two_spotted_spider_mite/
│   ├── Tomato__Target_Spot/
│   ├── Tomato__Tomato_YellowLeaf__Curl_Virus/
│   ├── Tomato__Tomato_mosaic_virus/
│   └── Tomato_healthy/
├── scripts/                         # Main processing scripts
│   ├── extract_and_FD.py           # Feature extraction and FD calculation
│   ├── model.py                    # Machine learning model training
│   ├── prediction.py               # Disease prediction with sustainability
│   └── hash.py                     # Cryptographic certificate generation
├── requirements.txt                 # Python dependencies
└── README.md                       # This file
```

## 🚀 Features

- **Fractal Dimension Analysis**: Calculates geometric and textural fractal dimensions
- **Texture Feature Extraction**: Uses Gray-Level Co-occurrence Matrix (GLCM) for texture analysis
- **Machine Learning Classification**: Random Forest classifier for disease detection
- **Sustainability Assessment**: Incorporates farming practices into health scoring
- **Cryptographic Certificates**: Generates secure certificates for leaf identity verification
- **Edge and Vein Enhancement**: Advanced image processing for better feature extraction

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/darkraider01/leafFD.git
cd leafFD
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## 📊 Usage

### Step 1: Feature Extraction

Extract fractal dimension features from your plant images:

```bash
python scripts/extract_and_FD.py
```

This script will:
- Process all images in the [`PlantVillage/`](#project-structure) directory
- Calculate geometric and textural fractal dimensions
- Extract GLCM texture features (contrast, dissimilarity, homogeneity, energy, correlation, ASM)
- Save processed edge and texture images
- Generate a CSV file with all extracted features

### Step 2: Model Training

Train the machine learning classifier:

```bash
python scripts/model.py
```

This will:
- Load the feature data from the CSV file
- Split data into training and testing sets
- Train a Random Forest classifier
- Evaluate model performance
- Save the trained model and label encoder

### Step 3: Disease Prediction

Make predictions on new leaf images:

```bash
python scripts/prediction.py
```

The prediction system will:
- Prompt for an image path
- Extract fractal dimension features
- Request sustainability data (soil health, water usage, pesticide usage, farming techniques)
- Provide comprehensive health assessment
- Log predictions for future reference

### Step 4: Generate Certificates (Optional)

Create cryptographic certificates for leaf verification:

```bash
python scripts/hash.py
```

This generates secure JSON certificates containing:
- Image hash and identity verification
- Fractal dimension biometrics
- Timestamp and metadata
- Cryptographic binding of image and FD data

## 🔬 Technical Details

### Fractal Dimension Calculation

The system calculates two types of fractal dimensions:

1. **Geometric FD**: Uses box-counting method on binary images
2. **Textural FD**: Derived from GLCM texture features

### Feature Set

Each image is characterized by 8 features:
- `fd_geometrical`: Geometric fractal dimension
- `fd_textural`: Textural fractal dimension  
- `contrast`: GLCM contrast measure
- `dissimilarity`: GLCM dissimilarity measure
- `homogeneity`: GLCM homogeneity measure
- `energy`: GLCM energy measure
- `correlation`: GLCM correlation measure
- `ASM`: Angular Second Moment from GLCM

### Sustainability Scoring

The system incorporates sustainability metrics:
- Soil health (0-100 scale)
- Water usage efficiency (0-100 scale)
- Pesticide usage (0-100 scale, lower is better)
- Farming technique sustainability (0-100 scale)

## 📈 Model Performance

The Random Forest classifier provides:
- Accuracy metrics on test data
- Detailed classification report
- Support for multiple plant species and disease types

## 🔒 Security Features

The cryptographic certificate system provides:
- SHA-256 image hashing
- Fractal dimension biometric binding
- Tamper-evident certificates
- Unique identity verification

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests to improve the system.

## 📄 License

This project is open source. Please check the repository for license details.

## 🙏 Acknowledgments

- PlantVillage dataset for providing plant disease images
- OpenCV and scikit-image communities for image processing tools
- scikit-learn for machine learning capabilities

---

*For questions or support, please open an issue in the repository.*