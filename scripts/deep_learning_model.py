"""
Deep Learning Model with Explainable AI for Plant Disease Detection
Combines CNN features with traditional fractal dimension analysis
"""

import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import lime
from lime import lime_image
from skimage.segmentation import mark_boundaries

class HybridPlantDiseaseClassifier:
    """Hybrid model combining CNN and traditional ML features"""
    
    def __init__(self, img_size=(224, 224)):
        self.img_size = img_size
        self.cnn_model = None
        self.hybrid_model = None
        self.label_encoder = LabelEncoder()
        self.feature_extractor = None
        
    def build_cnn_feature_extractor(self, num_classes):
        """Build CNN for feature extraction"""
        base_model = keras.applications.EfficientNetB0(
            weights='imagenet',
            include_top=False,
            input_shape=(*self.img_size, 3)
        )
        
        # Freeze base model layers
        base_model.trainable = False
        
        # Add custom layers
        inputs = keras.Input(shape=(*self.img_size, 3))
        x = base_model(inputs, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.2)(x)
        
        # Feature extraction layer
        features = layers.Dense(256, activation='relu', name='feature_layer')(x)
        outputs = layers.Dense(num_classes, activation='softmax')(features)
        
        self.cnn_model = keras.Model(inputs, outputs)
        self.feature_extractor = keras.Model(inputs, features)
        
        return self.cnn_model
    
    def build_hybrid_model(self, cnn_features_dim, traditional_features_dim, num_classes):
        """Build hybrid model combining CNN and traditional features"""
        # CNN features input
        cnn_input = keras.Input(shape=(cnn_features_dim,), name='cnn_features')
        
        # Traditional features input
        traditional_input = keras.Input(shape=(traditional_features_dim,), name='traditional_features')
        
        # Process CNN features
        cnn_branch = layers.Dense(128, activation='relu')(cnn_input)
        cnn_branch = layers.Dropout(0.3)(cnn_branch)
        cnn_branch = layers.Dense(64, activation='relu')(cnn_branch)
        
        # Process traditional features
        trad_branch = layers.Dense(64, activation='relu')(traditional_input)
        trad_branch = layers.Dropout(0.2)(trad_branch)
        trad_branch = layers.Dense(32, activation='relu')(trad_branch)
        
        # Combine features
        combined = layers.concatenate([cnn_branch, trad_branch])
        combined = layers.Dense(128, activation='relu')(combined)
        combined = layers.Dropout(0.3)(combined)
        combined = layers.Dense(64, activation='relu')(combined)
        
        # Output layer
        outputs = layers.Dense(num_classes, activation='softmax')(combined)
        
        self.hybrid_model = keras.Model(
            inputs=[cnn_input, traditional_input],
            outputs=outputs
        )
        
        return self.hybrid_model
    
    def preprocess_image(self, img_path):
        """Preprocess image for CNN"""
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Cannot load image: {img_path}")
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.img_size)
        img = img.astype(np.float32) / 255.0
        
        return img
    
    def extract_cnn_features(self, images):
        """Extract features using CNN feature extractor"""
        if self.feature_extractor is None:
            raise ValueError("Feature extractor not built yet")
        
        return self.feature_extractor.predict(images)
    
    def train_models(self, image_paths, traditional_features, labels):
        """Train both CNN and hybrid models"""
        # Prepare data
        images = np.array([self.preprocess_image(path) for path in image_paths])
        labels_encoded = self.label_encoder.fit_transform(labels)
        num_classes = len(np.unique(labels_encoded))
        
        # Split data
        X_img_train, X_img_test, X_trad_train, X_trad_test, y_train, y_test = train_test_split(
            images, traditional_features, labels_encoded, test_size=0.2, random_state=42, stratify=labels_encoded
        )
        
        # Build and train CNN
        self.build_cnn_feature_extractor(num_classes)
        self.cnn_model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Train CNN
        history_cnn = self.cnn_model.fit(
            X_img_train, y_train,
            validation_data=(X_img_test, y_test),
            epochs=20,
            batch_size=32,
            verbose=1
        )
        
        # Extract CNN features
        cnn_features_train = self.extract_cnn_features(X_img_train)
        cnn_features_test = self.extract_cnn_features(X_img_test)
        
        # Build and train hybrid model
        self.build_hybrid_model(
            cnn_features_train.shape[1],
            X_trad_train.shape[1],
            num_classes
        )
        
        self.hybrid_model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Train hybrid model
        history_hybrid = self.hybrid_model.fit(
            [cnn_features_train, X_trad_train], y_train,
            validation_data=([cnn_features_test, X_trad_test], y_test),
            epochs=30,
            batch_size=32,
            verbose=1
        )
        
        return history_cnn, history_hybrid
    
    def predict_with_explanation(self, img_path, traditional_features):
        """Make prediction with LIME explanation"""
        # Preprocess image
        img = self.preprocess_image(img_path)
        img_batch = np.expand_dims(img, axis=0)
        
        # Extract CNN features
        cnn_features = self.extract_cnn_features(img_batch)
        
        # Make prediction
        prediction = self.hybrid_model.predict([cnn_features, traditional_features.reshape(1, -1)])
        predicted_class = np.argmax(prediction)
        confidence = np.max(prediction)
        predicted_label = self.label_encoder.inverse_transform([predicted_class])[0]
        
        # Generate LIME explanation
        explainer = lime_image.LimeImageExplainer()
        
        def predict_fn(images):
            """Prediction function for LIME"""
            processed_images = []
            for image in images:
                # Ensure image is in correct format
                if image.max() <= 1.0:
                    processed_images.append(image)
                else:
                    processed_images.append(image / 255.0)
            
            processed_images = np.array(processed_images)
            cnn_feats = self.extract_cnn_features(processed_images)
            
            # Use average traditional features for explanation
            trad_feats = np.tile(traditional_features, (len(processed_images), 1))
            
            return self.hybrid_model.predict([cnn_feats, trad_feats])
        
        # Generate explanation
        explanation = explainer.explain_instance(
            img,
            predict_fn,
            top_labels=3,
            hide_color=0,
            num_samples=1000
        )
        
        return {
            'predicted_label': predicted_label,
            'confidence': confidence,
            'prediction_probabilities': prediction[0],
            'explanation': explanation
        }
    
    def visualize_explanation(self, explanation, img_path, save_path=None):
        """Visualize LIME explanation"""
        img = self.preprocess_image(img_path)
        
        # Get explanation for top predicted class
        temp, mask = explanation.get_image_and_mask(
            explanation.top_labels[0],
            positive_only=True,
            num_features=10,
            hide_rest=False
        )
        
        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        axes[0].imshow(img)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Explanation overlay
        axes[1].imshow(mark_boundaries(temp, mask))
        axes[1].set_title('LIME Explanation')
        axes[1].axis('off')
        
        # Heatmap
        axes[2].imshow(mask, cmap='RdYlBu')
        axes[2].set_title('Feature Importance Heatmap')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def generate_comprehensive_report(self, img_path, traditional_features, save_dir="./reports"):
        """Generate comprehensive analysis report"""
        os.makedirs(save_dir, exist_ok=True)
        
        # Get prediction with explanation
        result = self.predict_with_explanation(img_path, traditional_features)
        
        # Create report
        report = {
            'image_path': img_path,
            'predicted_disease': result['predicted_label'],
            'confidence': float(result['confidence']),
            'all_probabilities': {
                self.label_encoder.inverse_transform([i])[0]: float(prob)
                for i, prob in enumerate(result['prediction_probabilities'])
            },
            'traditional_features': {
                f'feature_{i}': float(val) for i, val in enumerate(traditional_features)
            }
        }
        
        # Save explanation visualization
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        explanation_path = os.path.join(save_dir, f"{img_name}_explanation.png")
        self.visualize_explanation(result['explanation'], img_path, explanation_path)
        
        # Save report as JSON
        import json
        report_path = os.path.join(save_dir, f"{img_name}_report.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        return report, explanation_path, report_path
    
    def save_models(self, save_dir="./models"):
        """Save trained models"""
        os.makedirs(save_dir, exist_ok=True)
        
        if self.cnn_model:
            self.cnn_model.save(os.path.join(save_dir, "cnn_model.h5"))
        
        if self.hybrid_model:
            self.hybrid_model.save(os.path.join(save_dir, "hybrid_model.h5"))
        
        # Save label encoder
        import joblib
        joblib.dump(self.label_encoder, os.path.join(save_dir, "label_encoder.pkl"))
    
    def load_models(self, save_dir="./models"):
        """Load trained models"""
        import joblib
        
        self.cnn_model = keras.models.load_model(os.path.join(save_dir, "cnn_model.h5"))
        self.hybrid_model = keras.models.load_model(os.path.join(save_dir, "hybrid_model.h5"))
        self.label_encoder = joblib.load(os.path.join(save_dir, "label_encoder.pkl"))
        
        # Recreate feature extractor
        self.feature_extractor = keras.Model(
            self.cnn_model.input,
            self.cnn_model.get_layer('feature_layer').output
        )

def main():
    """Example usage of hybrid classifier"""
    print("🚀 Initializing Hybrid Plant Disease Classifier...")
    
    classifier = HybridPlantDiseaseClassifier()
    
    # This would be used with actual training data
    print("📊 For training, you would need:")
    print("1. Image paths list")
    print("2. Traditional features array (from fractal analysis)")
    print("3. Disease labels")
    print("\nExample training call:")
    print("history_cnn, history_hybrid = classifier.train_models(image_paths, traditional_features, labels)")
    
    print("\n🔍 For prediction with explanation:")
    print("result = classifier.predict_with_explanation(img_path, traditional_features)")
    print("classifier.generate_comprehensive_report(img_path, traditional_features)")

if __name__ == "__main__":
    main()