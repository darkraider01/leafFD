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
│   ├── extract_and_FD.py           # Basic feature extraction and FD calculation
│   ├── advanced_fd_analysis.py     # Advanced multi-scale fractal analysis
│   ├── model.py                    # Traditional machine learning model training
│   ├── deep_learning_model.py      # Hybrid CNN + traditional ML with explainable AI
│   ├── prediction.py               # Disease prediction with sustainability
│   ├── hash.py                     # Cryptographic certificate generation
│   ├── web_api.py                  # Real-time web API with FastAPI
│   ├── mlops_pipeline.py           # MLOps pipeline with monitoring and versioning
│   └── mobile_optimization.py      # Mobile and edge deployment optimization
├── requirements.txt                 # Python dependencies (enhanced)
├── pipeline_config.yaml            # MLOps pipeline configuration
└── README.md                       # This file
```

## 🚀 Features

### Core Analysis
- **Advanced Multi-Scale Fractal Analysis**: Multiple FD calculation methods (box counting, differential box counting, blanket method, lacunarity, multifractal)
- **Texture Feature Extraction**: Uses Gray-Level Co-occurrence Matrix (GLCM) for comprehensive texture analysis
- **Morphological Feature Analysis**: Shape complexity, skeleton analysis, and geometric properties

### Machine Learning & AI
- **Hybrid Deep Learning**: Combines CNN features with traditional fractal analysis
- **Explainable AI**: LIME-based explanations for model predictions
- **Ensemble Methods**: Multiple model approaches for robust predictions
- **Knowledge Distillation**: Optimized student models for mobile deployment

### MLOps & Production
- **Model Versioning**: Complete model registry with version control
- **Performance Monitoring**: Real-time model performance tracking
- **Data Drift Detection**: Automatic detection of data distribution changes
- **Automated Retraining**: Trigger-based model retraining pipeline
- **A/B Testing**: Compare model versions in production

### Web & Mobile
- **Real-time Web API**: FastAPI-based service with interactive web interface
- **Mobile Optimization**: TensorFlow Lite conversion with quantization and pruning
- **Edge Deployment**: Optimized inference for resource-constrained devices
- **Batch Processing**: Handle multiple images simultaneously

### Security & Traceability
- **Cryptographic Certificates**: Generates secure certificates for leaf identity verification
- **Blockchain Integration**: Immutable record keeping for agricultural traceability
- **Audit Trails**: Complete logging of predictions and model decisions

### Sustainability & Analytics
- **Sustainability Assessment**: Incorporates farming practices into health scoring
- **Environmental Impact**: Carbon footprint tracking for AI operations
- **Comprehensive Reporting**: Detailed analysis reports with visualizations

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

### Quick Start with Web API

Launch the real-time web interface:

```bash
python scripts/web_api.py
```

Then open your browser to `http://localhost:12000` for an interactive web interface where you can:
- Upload leaf images for instant analysis
- Input sustainability data
- View detailed results with fractal analysis
- Download comprehensive reports

### Advanced Usage

#### 1. Advanced Feature Extraction

Extract comprehensive fractal features using multiple methods:

```bash
python scripts/advanced_fd_analysis.py
```

This provides:
- Multiple fractal dimension calculation methods
- Morphological feature analysis
- Enhanced preprocessing techniques
- Detailed feature visualization

#### 2. Hybrid Deep Learning Model

Train the advanced hybrid model combining CNN and traditional features:

```bash
python scripts/deep_learning_model.py
```

Features:
- EfficientNet-based feature extraction
- Hybrid architecture combining CNN and fractal features
- LIME-based explainable AI
- Comprehensive model evaluation

#### 3. MLOps Pipeline

Set up production-ready ML pipeline:

```bash
python scripts/mlops_pipeline.py
```

Capabilities:
- Model versioning and registry
- Performance monitoring
- Data drift detection
- Automated retraining triggers
- A/B testing support

#### 4. Mobile Optimization

Optimize models for mobile deployment:

```bash
python scripts/mobile_optimization.py
```

This provides:
- TensorFlow Lite conversion
- Model quantization and pruning
- Performance benchmarking
- Mobile deployment packages

#### 5. Traditional Workflow

For the original workflow:

**Step 1: Basic Feature Extraction**
```bash
python scripts/extract_and_FD.py
```

**Step 2: Traditional Model Training**
```bash
python scripts/model.py
```

**Step 3: Prediction**
```bash
python scripts/prediction.py
```

**Step 4: Certificate Generation**
```bash
python scripts/hash.py
```

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