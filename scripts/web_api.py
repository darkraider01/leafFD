"""
Real-time Web API for Plant Disease Detection
FastAPI-based service for real-time leaf analysis
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import cv2
import numpy as np
import pandas as pd
import joblib
import json
import uuid
from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel
import asyncio
import aiofiles
from pathlib import Path

# Import our custom modules
from advanced_fd_analysis import AdvancedFractalAnalyzer
from deep_learning_model import HybridPlantDiseaseClassifier

app = FastAPI(
    title="LeafFD Plant Disease Detection API",
    description="Advanced plant disease detection using fractal dimension analysis and deep learning",
    version="2.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global models
fractal_analyzer = None
ml_classifier = None
dl_classifier = None

# Data models
class SustainabilityData(BaseModel):
    soil_health: float
    water_usage: float
    pesticide_usage: float
    farming_techniques: float

class PredictionResponse(BaseModel):
    prediction_id: str
    predicted_disease: str
    confidence: float
    fractal_features: dict
    sustainability_score: float
    processing_time: float
    timestamp: str

class BatchPredictionRequest(BaseModel):
    image_urls: List[str]
    sustainability_data: Optional[SustainabilityData] = None

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize models on startup"""
    global fractal_analyzer, ml_classifier, dl_classifier
    
    print("🚀 Starting LeafFD API...")
    
    # Initialize fractal analyzer
    fractal_analyzer = AdvancedFractalAnalyzer()
    
    # Load traditional ML model if available
    try:
        ml_classifier = joblib.load("./processed/leaf_fd_classifier.pkl")
        print("✅ Traditional ML model loaded")
    except:
        print("⚠️ Traditional ML model not found")
    
    # Initialize deep learning classifier
    try:
        dl_classifier = HybridPlantDiseaseClassifier()
        dl_classifier.load_models("./models")
        print("✅ Deep learning model loaded")
    except:
        print("⚠️ Deep learning model not found")
    
    # Create necessary directories
    os.makedirs("./uploads", exist_ok=True)
    os.makedirs("./results", exist_ok=True)
    os.makedirs("./reports", exist_ok=True)

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main web interface"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>LeafFD - Plant Disease Detection</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
            .container { max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            h1 { color: #2c5530; text-align: center; }
            .upload-area { border: 2px dashed #4CAF50; padding: 40px; text-align: center; margin: 20px 0; border-radius: 10px; }
            .upload-area:hover { background: #f9f9f9; }
            input[type="file"] { margin: 10px 0; }
            button { background: #4CAF50; color: white; padding: 12px 24px; border: none; border-radius: 5px; cursor: pointer; font-size: 16px; }
            button:hover { background: #45a049; }
            .result { margin-top: 20px; padding: 20px; background: #e8f5e8; border-radius: 5px; }
            .sustainability { margin: 20px 0; padding: 15px; background: #f0f8ff; border-radius: 5px; }
            .feature-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin: 10px 0; }
            .feature-item { padding: 8px; background: #f9f9f9; border-radius: 3px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🌱 LeafFD Plant Disease Detection</h1>
            <p style="text-align: center; color: #666;">Upload a leaf image for advanced disease analysis using fractal dimension and AI</p>
            
            <div class="upload-area">
                <h3>📸 Upload Leaf Image</h3>
                <input type="file" id="imageFile" accept="image/*" />
                <br><br>
                <button onclick="uploadImage()">Analyze Leaf</button>
            </div>
            
            <div class="sustainability">
                <h3>🌍 Sustainability Data (Optional)</h3>
                <div class="feature-grid">
                    <div>Soil Health (0-100): <input type="number" id="soilHealth" value="75" min="0" max="100"></div>
                    <div>Water Usage (0-100): <input type="number" id="waterUsage" value="60" min="0" max="100"></div>
                    <div>Pesticide Usage (0-100): <input type="number" id="pesticideUsage" value="30" min="0" max="100"></div>
                    <div>Farming Techniques (0-100): <input type="number" id="farmingTech" value="80" min="0" max="100"></div>
                </div>
            </div>
            
            <div id="result" class="result" style="display: none;">
                <h3>📊 Analysis Results</h3>
                <div id="resultContent"></div>
            </div>
        </div>
        
        <script>
            async function uploadImage() {
                const fileInput = document.getElementById('imageFile');
                const file = fileInput.files[0];
                
                if (!file) {
                    alert('Please select an image file');
                    return;
                }
                
                const formData = new FormData();
                formData.append('file', file);
                formData.append('soil_health', document.getElementById('soilHealth').value);
                formData.append('water_usage', document.getElementById('waterUsage').value);
                formData.append('pesticide_usage', document.getElementById('pesticideUsage').value);
                formData.append('farming_techniques', document.getElementById('farmingTech').value);
                
                try {
                    document.getElementById('result').style.display = 'block';
                    document.getElementById('resultContent').innerHTML = '<p>🔄 Analyzing image... Please wait.</p>';
                    
                    const response = await fetch('/predict', {
                        method: 'POST',
                        body: formData
                    });
                    
                    const result = await response.json();
                    
                    if (response.ok) {
                        displayResult(result);
                    } else {
                        document.getElementById('resultContent').innerHTML = `<p style="color: red;">❌ Error: ${result.detail}</p>`;
                    }
                } catch (error) {
                    document.getElementById('resultContent').innerHTML = `<p style="color: red;">❌ Error: ${error.message}</p>`;
                }
            }
            
            function displayResult(result) {
                const content = `
                    <div style="background: white; padding: 15px; border-radius: 5px; margin: 10px 0;">
                        <h4>🎯 Prediction: ${result.predicted_disease}</h4>
                        <p><strong>Confidence:</strong> ${(result.confidence * 100).toFixed(1)}%</p>
                        <p><strong>Sustainability Score:</strong> ${result.sustainability_score.toFixed(1)}/100</p>
                        <p><strong>Processing Time:</strong> ${result.processing_time.toFixed(2)}s</p>
                    </div>
                    
                    <div style="background: white; padding: 15px; border-radius: 5px; margin: 10px 0;">
                        <h4>🔬 Fractal Features</h4>
                        <div class="feature-grid">
                            ${Object.entries(result.fractal_features).map(([key, value]) => 
                                `<div class="feature-item"><strong>${key}:</strong> ${typeof value === 'number' ? value.toFixed(4) : JSON.stringify(value)}</div>`
                            ).join('')}
                        </div>
                    </div>
                    
                    <p style="font-size: 12px; color: #666;">Prediction ID: ${result.prediction_id}</p>
                `;
                
                document.getElementById('resultContent').innerHTML = content;
            }
        </script>
    </body>
    </html>
    """

@app.post("/predict", response_model=PredictionResponse)
async def predict_disease(
    file: UploadFile = File(...),
    soil_health: float = 75.0,
    water_usage: float = 60.0,
    pesticide_usage: float = 30.0,
    farming_techniques: float = 80.0
):
    """Predict plant disease from uploaded image"""
    start_time = datetime.now()
    prediction_id = str(uuid.uuid4())
    
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Save uploaded file
        file_path = f"./uploads/{prediction_id}_{file.filename}"
        async with aiofiles.open(file_path, 'wb') as f:
            content = await file.read()
            await f.write(content)
        
        # Extract fractal features
        fractal_features = fractal_analyzer.analyze_image(file_path)
        
        # Calculate sustainability score
        sustainability_score = calculate_sustainability_score(
            soil_health, water_usage, pesticide_usage, farming_techniques
        )
        
        # Make prediction (using traditional ML for now)
        if ml_classifier:
            # Prepare features for ML model
            feature_vector = prepare_feature_vector(fractal_features)
            prediction = ml_classifier.predict([feature_vector])[0]
            confidence = max(ml_classifier.predict_proba([feature_vector])[0])
        else:
            # Fallback prediction
            prediction = "healthy" if fractal_features.get('box_counting', 0) > 1.5 else "diseased"
            confidence = 0.75
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Save result
        result = {
            "prediction_id": prediction_id,
            "predicted_disease": prediction,
            "confidence": float(confidence),
            "fractal_features": fractal_features,
            "sustainability_score": sustainability_score,
            "processing_time": processing_time,
            "timestamp": datetime.now().isoformat()
        }
        
        # Save to database/log
        await save_prediction_result(result, file_path)
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/batch_predict")
async def batch_predict(request: BatchPredictionRequest):
    """Batch prediction for multiple images"""
    results = []
    
    for i, image_url in enumerate(request.image_urls):
        try:
            # Download image (simplified - in production, add proper validation)
            # This is a placeholder for batch processing
            result = {
                "image_url": image_url,
                "prediction_id": str(uuid.uuid4()),
                "status": "pending"
            }
            results.append(result)
        except Exception as e:
            results.append({
                "image_url": image_url,
                "error": str(e),
                "status": "failed"
            })
    
    return {"batch_id": str(uuid.uuid4()), "results": results}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "models_loaded": {
            "fractal_analyzer": fractal_analyzer is not None,
            "ml_classifier": ml_classifier is not None,
            "dl_classifier": dl_classifier is not None
        }
    }

@app.get("/stats")
async def get_stats():
    """Get API usage statistics"""
    # In production, this would query a database
    return {
        "total_predictions": 0,
        "accuracy_metrics": {},
        "popular_diseases": [],
        "average_processing_time": 0.0
    }

@app.get("/download_report/{prediction_id}")
async def download_report(prediction_id: str):
    """Download detailed analysis report"""
    report_path = f"./reports/{prediction_id}_report.json"
    
    if os.path.exists(report_path):
        return FileResponse(
            report_path,
            media_type='application/json',
            filename=f"leaf_analysis_{prediction_id}.json"
        )
    else:
        raise HTTPException(status_code=404, detail="Report not found")

# Helper functions
def calculate_sustainability_score(soil_health, water_usage, pesticide_usage, farming_techniques):
    """Calculate sustainability score"""
    normalized_soil_health = soil_health / 100
    normalized_water_usage = water_usage / 100
    normalized_pesticide_usage = pesticide_usage / 100
    normalized_farming_techniques = farming_techniques / 100
    
    sustainability_score = (0.25 * normalized_soil_health +
                           0.25 * normalized_water_usage +
                           0.25 * (1 - normalized_pesticide_usage) +
                           0.25 * normalized_farming_techniques)
    
    return sustainability_score * 100

def prepare_feature_vector(fractal_features):
    """Prepare feature vector for ML model"""
    # Extract relevant features for the model
    features = [
        fractal_features.get('box_counting', 0),
        fractal_features.get('differential_box_counting', 0),
        fractal_features.get('blanket', 0),
        fractal_features.get('lacunarity', 0),
        fractal_features.get('multifractal', 0),
    ]
    
    # Add morphological features if available
    if 'morphological' in fractal_features:
        morph = fractal_features['morphological']
        features.extend([
            morph.get('compactness', 0),
            morph.get('solidity', 0),
            morph.get('form_factor', 0)
        ])
    
    return features

async def save_prediction_result(result, file_path):
    """Save prediction result to log"""
    log_path = "./results/prediction_log.jsonl"
    
    # Add file info to result
    result['file_path'] = file_path
    result['file_size'] = os.path.getsize(file_path)
    
    # Append to log file
    async with aiofiles.open(log_path, 'a') as f:
        await f.write(json.dumps(result) + '\n')

if __name__ == "__main__":
    uvicorn.run(
        "web_api:app",
        host="0.0.0.0",
        port=12000,
        reload=True,
        log_level="info"
    )