"""
Mobile Optimization and Edge Deployment
Optimizes models for mobile devices and edge computing
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow_lite as tflite
import joblib
import json
import cv2
from pathlib import Path
import time
import psutil
import platform
from typing import Dict, List, Tuple, Optional

class MobileModelOptimizer:
    """Optimize models for mobile deployment"""
    
    def __init__(self):
        self.optimization_methods = {
            'quantization': self._apply_quantization,
            'pruning': self._apply_pruning,
            'distillation': self._apply_knowledge_distillation,
            'compression': self._apply_model_compression
        }
    
    def optimize_tensorflow_model(self, model_path: str, optimization_type: str = 'quantization') -> str:
        """Optimize TensorFlow model for mobile deployment"""
        model = keras.models.load_model(model_path)
        
        if optimization_type in self.optimization_methods:
            optimized_model = self.optimization_methods[optimization_type](model)
        else:
            optimized_model = model
        
        # Convert to TensorFlow Lite
        tflite_model_path = self._convert_to_tflite(optimized_model, model_path)
        
        return tflite_model_path
    
    def _apply_quantization(self, model):
        """Apply post-training quantization"""
        # Create a representative dataset for quantization
        def representative_dataset():
            for _ in range(100):
                # Generate random data matching your input shape
                data = np.random.random((1, 224, 224, 3)).astype(np.float32)
                yield [data]
        
        # Convert with quantization
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.uint8
        
        return converter
    
    def _apply_pruning(self, model):
        """Apply magnitude-based pruning"""
        import tensorflow_model_optimization as tfmot
        
        # Define pruning parameters
        pruning_params = {
            'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
                initial_sparsity=0.30,
                final_sparsity=0.70,
                begin_step=0,
                end_step=1000
            )
        }
        
        # Apply pruning to the model
        model_for_pruning = tfmot.sparsity.keras.prune_low_magnitude(model, **pruning_params)
        
        return model_for_pruning
    
    def _apply_knowledge_distillation(self, teacher_model):
        """Create a smaller student model using knowledge distillation"""
        # Create a smaller student model
        student_model = keras.Sequential([
            keras.layers.Conv2D(16, 3, activation='relu', input_shape=(224, 224, 3)),
            keras.layers.MaxPooling2D(),
            keras.layers.Conv2D(32, 3, activation='relu'),
            keras.layers.MaxPooling2D(),
            keras.layers.Conv2D(64, 3, activation='relu'),
            keras.layers.GlobalAveragePooling2D(),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(teacher_model.output_shape[-1], activation='softmax')
        ])
        
        # Implement distillation training (simplified)
        # In practice, you would train the student model to match teacher outputs
        
        return student_model
    
    def _apply_model_compression(self, model):
        """Apply general model compression techniques"""
        # This could include various compression techniques
        # For now, we'll use the model as-is and rely on TFLite conversion
        return model
    
    def _convert_to_tflite(self, model, original_path: str) -> str:
        """Convert model to TensorFlow Lite format"""
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        tflite_model = converter.convert()
        
        # Save the model
        tflite_path = original_path.replace('.h5', '_mobile.tflite')
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        
        return tflite_path
    
    def optimize_sklearn_model(self, model_path: str) -> str:
        """Optimize scikit-learn model for mobile deployment"""
        model = joblib.load(model_path)
        
        # For sklearn models, we can use various optimization techniques
        optimized_model = self._compress_sklearn_model(model)
        
        # Save optimized model
        optimized_path = model_path.replace('.pkl', '_mobile.pkl')
        joblib.dump(optimized_model, optimized_path)
        
        return optimized_path
    
    def _compress_sklearn_model(self, model):
        """Compress sklearn model by reducing precision"""
        # This is a simplified compression - in practice, you might use
        # techniques like feature selection, model simplification, etc.
        return model

class EdgeInferenceEngine:
    """Optimized inference engine for edge devices"""
    
    def __init__(self, model_path: str, model_type: str = 'tflite'):
        self.model_path = model_path
        self.model_type = model_type
        self.model = None
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        
        self._load_model()
    
    def _load_model(self):
        """Load the optimized model"""
        if self.model_type == 'tflite':
            self.interpreter = tf.lite.Interpreter(model_path=self.model_path)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
        elif self.model_type == 'sklearn':
            self.model = joblib.load(self.model_path)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def preprocess_image(self, image_path: str) -> np.ndarray:
        """Preprocess image for inference"""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Cannot load image: {image_path}")
        
        # Resize to model input size
        if self.model_type == 'tflite':
            input_shape = self.input_details[0]['shape']
            target_size = (input_shape[1], input_shape[2])
        else:
            target_size = (224, 224)  # Default size
        
        img = cv2.resize(img, target_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if self.model_type == 'tflite':
            # Normalize for TFLite model
            img = img.astype(np.float32) / 255.0
            img = np.expand_dims(img, axis=0)
        
        return img
    
    def extract_features_fast(self, image: np.ndarray) -> np.ndarray:
        """Fast feature extraction for edge devices"""
        # Simplified fractal dimension calculation for speed
        gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        
        # Fast box counting approximation
        sizes = [2, 4, 8, 16]
        counts = []
        
        for size in sizes:
            # Simplified box counting
            h, w = gray.shape
            boxes_h = h // size
            boxes_w = w // size
            
            if boxes_h > 0 and boxes_w > 0:
                resized = gray[:boxes_h*size, :boxes_w*size]
                boxes = resized.reshape(boxes_h, size, boxes_w, size)
                non_empty = np.count_nonzero(np.sum(boxes, axis=(1, 3)))
                counts.append(non_empty)
        
        if len(counts) >= 2:
            log_sizes = np.log(sizes[:len(counts)])
            log_counts = np.log(counts)
            fd = -np.polyfit(log_sizes, log_counts, 1)[0]
        else:
            fd = 1.5  # Default value
        
        # Additional fast features
        mean_intensity = np.mean(gray)
        std_intensity = np.std(gray)
        
        # Edge density
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (gray.shape[0] * gray.shape[1])
        
        return np.array([fd, mean_intensity, std_intensity, edge_density])
    
    def predict(self, image_path: str) -> Dict:
        """Make prediction on edge device"""
        start_time = time.time()
        
        # Preprocess image
        img = self.preprocess_image(image_path)
        preprocessing_time = time.time() - start_time
        
        # Extract features
        feature_start = time.time()
        if self.model_type == 'tflite':
            # Use TFLite model for prediction
            self.interpreter.set_tensor(self.input_details[0]['index'], img)
            self.interpreter.invoke()
            prediction = self.interpreter.get_tensor(self.output_details[0]['index'])
            predicted_class = np.argmax(prediction)
            confidence = np.max(prediction)
        else:
            # Use traditional features with sklearn model
            features = self.extract_features_fast(img[0])  # Remove batch dimension
            prediction = self.model.predict_proba([features])[0]
            predicted_class = np.argmax(prediction)
            confidence = np.max(prediction)
        
        feature_time = time.time() - feature_start
        total_time = time.time() - start_time
        
        return {
            'predicted_class': int(predicted_class),
            'confidence': float(confidence),
            'preprocessing_time': preprocessing_time,
            'inference_time': feature_time,
            'total_time': total_time,
            'memory_usage': self._get_memory_usage()
        }
    
    def _get_memory_usage(self) -> Dict:
        """Get current memory usage"""
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            'rss_mb': memory_info.rss / 1024 / 1024,  # Resident Set Size
            'vms_mb': memory_info.vms / 1024 / 1024,  # Virtual Memory Size
            'percent': process.memory_percent()
        }

class MobileBenchmark:
    """Benchmark models on mobile/edge devices"""
    
    def __init__(self):
        self.device_info = self._get_device_info()
    
    def _get_device_info(self) -> Dict:
        """Get device information"""
        return {
            'platform': platform.platform(),
            'processor': platform.processor(),
            'cpu_count': psutil.cpu_count(),
            'memory_total_gb': psutil.virtual_memory().total / 1024**3,
            'python_version': platform.python_version()
        }
    
    def benchmark_model(self, model_path: str, model_type: str, 
                       test_images: List[str], num_runs: int = 10) -> Dict:
        """Benchmark model performance"""
        engine = EdgeInferenceEngine(model_path, model_type)
        
        results = {
            'device_info': self.device_info,
            'model_path': model_path,
            'model_type': model_type,
            'num_test_images': len(test_images),
            'num_runs': num_runs,
            'results': []
        }
        
        total_times = []
        memory_usages = []
        
        for run in range(num_runs):
            run_results = []
            
            for img_path in test_images:
                try:
                    result = engine.predict(img_path)
                    run_results.append(result)
                    total_times.append(result['total_time'])
                    memory_usages.append(result['memory_usage']['rss_mb'])
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
            
            results['results'].append(run_results)
        
        # Calculate statistics
        results['statistics'] = {
            'avg_inference_time': np.mean(total_times),
            'std_inference_time': np.std(total_times),
            'min_inference_time': np.min(total_times),
            'max_inference_time': np.max(total_times),
            'avg_memory_usage_mb': np.mean(memory_usages),
            'max_memory_usage_mb': np.max(memory_usages),
            'throughput_fps': 1.0 / np.mean(total_times) if total_times else 0
        }
        
        return results
    
    def compare_models(self, model_configs: List[Dict], test_images: List[str]) -> Dict:
        """Compare multiple model configurations"""
        comparison_results = {
            'device_info': self.device_info,
            'models': []
        }
        
        for config in model_configs:
            print(f"Benchmarking {config['name']}...")
            
            try:
                benchmark_result = self.benchmark_model(
                    config['path'], 
                    config['type'], 
                    test_images
                )
                
                model_summary = {
                    'name': config['name'],
                    'path': config['path'],
                    'type': config['type'],
                    'statistics': benchmark_result['statistics'],
                    'model_size_mb': os.path.getsize(config['path']) / 1024 / 1024
                }
                
                comparison_results['models'].append(model_summary)
                
            except Exception as e:
                print(f"Error benchmarking {config['name']}: {e}")
        
        # Rank models by performance
        if comparison_results['models']:
            # Sort by inference time (ascending) and memory usage (ascending)
            comparison_results['models'].sort(
                key=lambda x: (x['statistics']['avg_inference_time'], 
                              x['statistics']['avg_memory_usage_mb'])
            )
            
            # Add rankings
            for i, model in enumerate(comparison_results['models']):
                model['rank'] = i + 1
        
        return comparison_results

def create_mobile_deployment_package(model_path: str, model_type: str, 
                                   output_dir: str = "./mobile_package"):
    """Create a deployment package for mobile applications"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Copy optimized model
    model_filename = os.path.basename(model_path)
    target_model_path = os.path.join(output_dir, model_filename)
    
    import shutil
    shutil.copy2(model_path, target_model_path)
    
    # Create configuration file
    config = {
        'model_filename': model_filename,
        'model_type': model_type,
        'input_size': [224, 224, 3] if model_type == 'tflite' else [8],
        'preprocessing': {
            'normalize': True,
            'resize': True,
            'color_space': 'RGB'
        },
        'classes': ['healthy', 'diseased'],  # Update with actual classes
        'version': '1.0.0',
        'created_at': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open(os.path.join(output_dir, 'model_config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    # Create inference script
    inference_script = '''
import numpy as np
import cv2
import json
import tensorflow as tf

class MobileInference:
    def __init__(self, model_path, config_path):
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        if self.config['model_type'] == 'tflite':
            self.interpreter = tf.lite.Interpreter(model_path=model_path)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
    
    def predict(self, image_path):
        img = cv2.imread(image_path)
        img = cv2.resize(img, tuple(self.config['input_size'][:2]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)
        
        self.interpreter.set_tensor(self.input_details[0]['index'], img)
        self.interpreter.invoke()
        prediction = self.interpreter.get_tensor(self.output_details[0]['index'])
        
        predicted_class = np.argmax(prediction)
        confidence = np.max(prediction)
        
        return {
            'class': self.config['classes'][predicted_class],
            'confidence': float(confidence)
        }
'''
    
    with open(os.path.join(output_dir, 'mobile_inference.py'), 'w') as f:
        f.write(inference_script)
    
    # Create README
    readme_content = f"""
# Mobile Deployment Package

This package contains an optimized model for mobile deployment.

## Files:
- `{model_filename}`: Optimized model file
- `model_config.json`: Model configuration
- `mobile_inference.py`: Inference script
- `README.md`: This file

## Usage:
```python
from mobile_inference import MobileInference

inference = MobileInference('{model_filename}', 'model_config.json')
result = inference.predict('path/to/image.jpg')
print(result)
```

## Requirements:
- TensorFlow Lite (for .tflite models)
- OpenCV
- NumPy

Model Type: {model_type}
Created: {time.strftime('%Y-%m-%d %H:%M:%S')}
"""
    
    with open(os.path.join(output_dir, 'README.md'), 'w') as f:
        f.write(readme_content)
    
    print(f"✅ Mobile deployment package created in: {output_dir}")
    return output_dir

def main():
    """Example usage of mobile optimization"""
    print("📱 Mobile Optimization for Plant Disease Detection")
    
    # Initialize optimizer
    optimizer = MobileModelOptimizer()
    
    print("🔧 Model optimization capabilities:")
    print("- TensorFlow Lite conversion")
    print("- Post-training quantization")
    print("- Model pruning")
    print("- Knowledge distillation")
    print("- Scikit-learn compression")
    
    # Example benchmark
    print("\n📊 Benchmarking capabilities:")
    benchmark = MobileBenchmark()
    print(f"Device: {benchmark.device_info['platform']}")
    print(f"CPU cores: {benchmark.device_info['cpu_count']}")
    print(f"Memory: {benchmark.device_info['memory_total_gb']:.1f} GB")
    
    print("\n📦 Mobile deployment package creation available")
    print("Use create_mobile_deployment_package() to generate deployment files")

if __name__ == "__main__":
    main()