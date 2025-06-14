"""
MLOps Pipeline for Plant Disease Detection
Includes model versioning, monitoring, and automated retraining
"""

import os
import json
import pickle
import hashlib
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import mlflow
import mlflow.sklearn
import mlflow.tensorflow
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import joblib
import sqlite3
from pathlib import Path
import logging
from dataclasses import dataclass, asdict
import yaml

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ModelMetrics:
    """Model performance metrics"""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    confusion_matrix: List[List[int]]
    timestamp: str
    model_version: str

@dataclass
class DataDriftMetrics:
    """Data drift detection metrics"""
    feature_drift_scores: Dict[str, float]
    overall_drift_score: float
    drift_detected: bool
    timestamp: str

class ModelRegistry:
    """Model registry for version control and metadata management"""
    
    def __init__(self, registry_path: str = "./model_registry"):
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(exist_ok=True)
        self.db_path = self.registry_path / "registry.db"
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database for model registry"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS models (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT NOT NULL,
                version TEXT NOT NULL,
                model_type TEXT NOT NULL,
                file_path TEXT NOT NULL,
                metrics TEXT,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                is_active BOOLEAN DEFAULT FALSE,
                UNIQUE(model_name, version)
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_version TEXT NOT NULL,
                input_hash TEXT NOT NULL,
                prediction TEXT NOT NULL,
                confidence REAL,
                actual_label TEXT,
                feedback_score INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS drift_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                drift_score REAL NOT NULL,
                feature_drifts TEXT,
                drift_detected BOOLEAN,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        conn.commit()
        conn.close()
    
    def register_model(self, model_name: str, version: str, model_type: str, 
                      model_object, metrics: ModelMetrics, metadata: Dict = None):
        """Register a new model version"""
        # Save model file
        model_filename = f"{model_name}_v{version}.pkl"
        model_path = self.registry_path / model_filename
        
        if model_type == "sklearn":
            joblib.dump(model_object, model_path)
        elif model_type == "tensorflow":
            model_object.save(str(model_path.with_suffix('.h5')))
            model_path = model_path.with_suffix('.h5')
        else:
            with open(model_path, 'wb') as f:
                pickle.dump(model_object, f)
        
        # Store in database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                INSERT INTO models (model_name, version, model_type, file_path, metrics, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                model_name, version, model_type, str(model_path),
                json.dumps(asdict(metrics)), json.dumps(metadata or {})
            ))
            conn.commit()
            logger.info(f"Model {model_name} v{version} registered successfully")
        except sqlite3.IntegrityError:
            logger.error(f"Model {model_name} v{version} already exists")
        finally:
            conn.close()
    
    def get_active_model(self, model_name: str):
        """Get the currently active model"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT file_path, version, model_type FROM models 
            WHERE model_name = ? AND is_active = TRUE
        """, (model_name,))
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            file_path, version, model_type = result
            if model_type == "sklearn":
                model = joblib.load(file_path)
            elif model_type == "tensorflow":
                import tensorflow as tf
                model = tf.keras.models.load_model(file_path)
            else:
                with open(file_path, 'rb') as f:
                    model = pickle.load(f)
            
            return model, version
        
        return None, None
    
    def promote_model(self, model_name: str, version: str):
        """Promote a model version to active"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Deactivate all versions
        cursor.execute("""
            UPDATE models SET is_active = FALSE WHERE model_name = ?
        """, (model_name,))
        
        # Activate specified version
        cursor.execute("""
            UPDATE models SET is_active = TRUE 
            WHERE model_name = ? AND version = ?
        """, (model_name, version))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Model {model_name} v{version} promoted to active")

class DataDriftDetector:
    """Detect data drift in incoming predictions"""
    
    def __init__(self, reference_data: np.ndarray, feature_names: List[str]):
        self.reference_data = reference_data
        self.feature_names = feature_names
        self.reference_stats = self._calculate_stats(reference_data)
    
    def _calculate_stats(self, data: np.ndarray) -> Dict:
        """Calculate statistical properties of data"""
        return {
            'mean': np.mean(data, axis=0),
            'std': np.std(data, axis=0),
            'min': np.min(data, axis=0),
            'max': np.max(data, axis=0),
            'percentiles': np.percentile(data, [25, 50, 75], axis=0)
        }
    
    def detect_drift(self, new_data: np.ndarray, threshold: float = 0.1) -> DataDriftMetrics:
        """Detect drift in new data compared to reference"""
        new_stats = self._calculate_stats(new_data)
        
        feature_drift_scores = {}
        
        for i, feature_name in enumerate(self.feature_names):
            # Calculate drift score using statistical distance
            ref_mean, ref_std = self.reference_stats['mean'][i], self.reference_stats['std'][i]
            new_mean, new_std = new_stats['mean'][i], new_stats['std'][i]
            
            # Normalized difference in means
            mean_diff = abs(new_mean - ref_mean) / (ref_std + 1e-8)
            
            # Ratio of standard deviations
            std_ratio = max(new_std, ref_std) / (min(new_std, ref_std) + 1e-8)
            
            # Combined drift score
            drift_score = (mean_diff + std_ratio - 1) / 2
            feature_drift_scores[feature_name] = drift_score
        
        overall_drift_score = np.mean(list(feature_drift_scores.values()))
        drift_detected = overall_drift_score > threshold
        
        return DataDriftMetrics(
            feature_drift_scores=feature_drift_scores,
            overall_drift_score=overall_drift_score,
            drift_detected=drift_detected,
            timestamp=datetime.now().isoformat()
        )

class ModelMonitor:
    """Monitor model performance and trigger retraining"""
    
    def __init__(self, registry: ModelRegistry):
        self.registry = registry
        self.performance_threshold = 0.8  # Minimum acceptable accuracy
        self.drift_threshold = 0.1
        self.retraining_window = timedelta(days=7)
    
    def log_prediction(self, model_version: str, input_data: np.ndarray, 
                      prediction: str, confidence: float, actual_label: str = None):
        """Log a prediction for monitoring"""
        input_hash = hashlib.md5(input_data.tobytes()).hexdigest()
        
        conn = sqlite3.connect(self.registry.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO predictions (model_version, input_hash, prediction, confidence, actual_label)
            VALUES (?, ?, ?, ?, ?)
        """, (model_version, input_hash, prediction, confidence, actual_label))
        
        conn.commit()
        conn.close()
    
    def calculate_model_performance(self, model_version: str, days: int = 7) -> Optional[ModelMetrics]:
        """Calculate model performance over recent predictions"""
        conn = sqlite3.connect(self.registry.db_path)
        
        query = """
            SELECT prediction, actual_label, confidence 
            FROM predictions 
            WHERE model_version = ? 
            AND actual_label IS NOT NULL 
            AND created_at >= datetime('now', '-{} days')
        """.format(days)
        
        df = pd.read_sql_query(query, conn, params=(model_version,))
        conn.close()
        
        if len(df) < 10:  # Need minimum samples
            return None
        
        y_true = df['actual_label'].values
        y_pred = df['prediction'].values
        
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
        cm = confusion_matrix(y_true, y_pred).tolist()
        
        return ModelMetrics(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            confusion_matrix=cm,
            timestamp=datetime.now().isoformat(),
            model_version=model_version
        )
    
    def should_retrain(self, model_version: str) -> Tuple[bool, str]:
        """Determine if model should be retrained"""
        # Check performance degradation
        metrics = self.calculate_model_performance(model_version)
        if metrics and metrics.accuracy < self.performance_threshold:
            return True, f"Performance degraded: accuracy {metrics.accuracy:.3f} < {self.performance_threshold}"
        
        # Check data drift
        # This would require implementing drift detection on recent data
        
        # Check time since last training
        # This would check model age
        
        return False, "No retraining needed"

class AutoMLPipeline:
    """Automated ML pipeline for continuous learning"""
    
    def __init__(self, config_path: str = "./pipeline_config.yaml"):
        self.config = self._load_config(config_path)
        self.registry = ModelRegistry(self.config.get('registry_path', './model_registry'))
        self.monitor = ModelMonitor(self.registry)
        
        # Initialize MLflow
        mlflow.set_tracking_uri(self.config.get('mlflow_uri', './mlruns'))
        mlflow.set_experiment(self.config.get('experiment_name', 'plant_disease_detection'))
    
    def _load_config(self, config_path: str) -> Dict:
        """Load pipeline configuration"""
        default_config = {
            'registry_path': './model_registry',
            'mlflow_uri': './mlruns',
            'experiment_name': 'plant_disease_detection',
            'retraining_schedule': '0 2 * * 0',  # Weekly at 2 AM
            'performance_threshold': 0.8,
            'drift_threshold': 0.1
        }
        
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                user_config = yaml.safe_load(f)
            default_config.update(user_config)
        
        return default_config
    
    def train_and_register_model(self, X_train: np.ndarray, y_train: np.ndarray,
                                X_test: np.ndarray, y_test: np.ndarray,
                                model_name: str = "plant_disease_classifier"):
        """Train and register a new model version"""
        
        with mlflow.start_run():
            # Train model (using Random Forest as example)
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.preprocessing import LabelEncoder
            
            # Encode labels
            le = LabelEncoder()
            y_train_encoded = le.fit_transform(y_train)
            y_test_encoded = le.transform(y_test)
            
            # Train model
            model = RandomForestClassifier(
                n_estimators=self.config.get('n_estimators', 100),
                random_state=42
            )
            model.fit(X_train, y_train_encoded)
            
            # Evaluate
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test_encoded, y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_test_encoded, y_pred, average='weighted'
            )
            cm = confusion_matrix(y_test_encoded, y_pred).tolist()
            
            # Log metrics to MLflow
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1_score", f1)
            
            # Log model
            mlflow.sklearn.log_model(model, "model")
            
            # Create version
            version = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Create metrics object
            metrics = ModelMetrics(
                accuracy=accuracy,
                precision=precision,
                recall=recall,
                f1_score=f1,
                confusion_matrix=cm,
                timestamp=datetime.now().isoformat(),
                model_version=version
            )
            
            # Register model
            self.registry.register_model(
                model_name=model_name,
                version=version,
                model_type="sklearn",
                model_object=model,
                metrics=metrics,
                metadata={
                    "training_samples": len(X_train),
                    "test_samples": len(X_test),
                    "features": X_train.shape[1],
                    "mlflow_run_id": mlflow.active_run().info.run_id
                }
            )
            
            # Auto-promote if better than current active model
            current_model, current_version = self.registry.get_active_model(model_name)
            if current_model is None or accuracy > self.config.get('auto_promote_threshold', 0.85):
                self.registry.promote_model(model_name, version)
                logger.info(f"Auto-promoted model {model_name} v{version}")
            
            return model, version, metrics
    
    def run_monitoring_cycle(self):
        """Run a complete monitoring cycle"""
        logger.info("Starting monitoring cycle...")
        
        # Get all active models
        conn = sqlite3.connect(self.registry.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT model_name FROM models WHERE is_active = TRUE")
        active_models = [row[0] for row in cursor.fetchall()]
        conn.close()
        
        for model_name in active_models:
            _, version = self.registry.get_active_model(model_name)
            
            # Check if retraining is needed
            should_retrain, reason = self.monitor.should_retrain(version)
            
            if should_retrain:
                logger.info(f"Retraining triggered for {model_name}: {reason}")
                # Here you would implement the retraining logic
                # self.retrain_model(model_name)
            else:
                logger.info(f"Model {model_name} v{version} performing well")
    
    def generate_model_report(self, model_name: str) -> Dict:
        """Generate comprehensive model report"""
        model, version = self.registry.get_active_model(model_name)
        
        if not model:
            return {"error": f"No active model found for {model_name}"}
        
        # Get recent performance
        metrics = self.monitor.calculate_model_performance(version)
        
        # Get prediction statistics
        conn = sqlite3.connect(self.registry.db_path)
        
        # Recent predictions count
        recent_predictions = pd.read_sql_query("""
            SELECT COUNT(*) as count 
            FROM predictions 
            WHERE model_version = ? 
            AND created_at >= datetime('now', '-7 days')
        """, conn, params=(version,))
        
        # Confidence distribution
        confidence_stats = pd.read_sql_query("""
            SELECT AVG(confidence) as avg_confidence, 
                   MIN(confidence) as min_confidence,
                   MAX(confidence) as max_confidence
            FROM predictions 
            WHERE model_version = ?
            AND created_at >= datetime('now', '-7 days')
        """, conn, params=(version,))
        
        conn.close()
        
        report = {
            "model_name": model_name,
            "version": version,
            "recent_predictions": recent_predictions.iloc[0]['count'],
            "performance_metrics": asdict(metrics) if metrics else None,
            "confidence_stats": confidence_stats.to_dict('records')[0],
            "generated_at": datetime.now().isoformat()
        }
        
        return report

def main():
    """Example usage of MLOps pipeline"""
    print("🚀 MLOps Pipeline for Plant Disease Detection")
    
    # Initialize pipeline
    pipeline = AutoMLPipeline()
    
    # Example: Load some dummy data for demonstration
    # In practice, this would load your actual training data
    X_train = np.random.rand(1000, 8)  # 8 features
    y_train = np.random.choice(['healthy', 'diseased'], 1000)
    X_test = np.random.rand(200, 8)
    y_test = np.random.choice(['healthy', 'diseased'], 200)
    
    print("📊 Training and registering model...")
    model, version, metrics = pipeline.train_and_register_model(
        X_train, y_train, X_test, y_test
    )
    
    print(f"✅ Model registered: version {version}")
    print(f"📈 Accuracy: {metrics.accuracy:.3f}")
    
    # Run monitoring
    print("🔍 Running monitoring cycle...")
    pipeline.run_monitoring_cycle()
    
    # Generate report
    print("📋 Generating model report...")
    report = pipeline.generate_model_report("plant_disease_classifier")
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()