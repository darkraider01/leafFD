"""
Advanced Multi-Scale Fractal Dimension Analysis
Implements multiple FD calculation methods for more robust feature extraction
"""

import cv2
import numpy as np
from scipy import ndimage
from skimage.morphology import skeletonize
from skimage.measure import regionprops
import matplotlib.pyplot as plt

class AdvancedFractalAnalyzer:
    """Advanced fractal dimension analysis with multiple methods"""
    
    def __init__(self):
        self.methods = {
            'box_counting': self._box_counting_fd,
            'differential_box_counting': self._differential_box_counting_fd,
            'blanket': self._blanket_fd,
            'lacunarity': self._lacunarity_analysis,
            'multifractal': self._multifractal_analysis
        }
    
    def analyze_image(self, image_path):
        """Comprehensive fractal analysis of leaf image"""
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Cannot load image: {image_path}")
        
        # Preprocessing
        img_processed = self._preprocess_image(img)
        
        results = {}
        for method_name, method_func in self.methods.items():
            try:
                results[method_name] = method_func(img_processed)
            except Exception as e:
                print(f"Warning: {method_name} failed: {e}")
                results[method_name] = None
        
        # Additional morphological features
        results['morphological'] = self._morphological_features(img_processed)
        
        return results
    
    def _preprocess_image(self, img):
        """Advanced preprocessing for better FD analysis"""
        # Adaptive histogram equalization
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        img = clahe.apply(img)
        
        # Noise reduction
        img = cv2.bilateralFilter(img, 9, 75, 75)
        
        # Edge-preserving smoothing
        img = cv2.edgePreservingFilter(img, flags=1, sigma_s=50, sigma_r=0.4)
        
        return img
    
    def _box_counting_fd(self, img):
        """Traditional box counting method"""
        # Convert to binary
        _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Box counting
        sizes = 2 ** np.arange(1, int(np.log2(min(binary.shape))) - 1)
        counts = []
        
        for size in sizes:
            # Reshape image to fit box size
            h, w = binary.shape
            h_boxes = h // size
            w_boxes = w // size
            
            if h_boxes == 0 or w_boxes == 0:
                continue
                
            resized = binary[:h_boxes*size, :w_boxes*size]
            boxes = resized.reshape(h_boxes, size, w_boxes, size)
            box_sums = np.sum(boxes, axis=(1, 3))
            non_empty_boxes = np.count_nonzero(box_sums)
            counts.append(non_empty_boxes)
        
        if len(counts) < 2:
            return 0.0
            
        # Linear regression
        log_sizes = np.log(sizes[:len(counts)])
        log_counts = np.log(counts)
        coeffs = np.polyfit(log_sizes, log_counts, 1)
        
        return -coeffs[0]
    
    def _differential_box_counting_fd(self, img):
        """Differential box counting for grayscale images"""
        h, w = img.shape
        max_box_size = min(h, w) // 4
        box_sizes = [2**i for i in range(1, int(np.log2(max_box_size)))]
        
        counts = []
        for box_size in box_sizes:
            if box_size >= min(h, w):
                break
                
            # Calculate differential box count
            count = 0
            for i in range(0, h - box_size, box_size):
                for j in range(0, w - box_size, box_size):
                    box = img[i:i+box_size, j:j+box_size]
                    min_val = np.min(box)
                    max_val = np.max(box)
                    
                    # Number of boxes needed to cover the height difference
                    height_diff = max_val - min_val
                    if height_diff > 0:
                        count += int(np.ceil(height_diff / box_size)) + 1
                    else:
                        count += 1
            
            counts.append(count)
        
        if len(counts) < 2:
            return 0.0
            
        # Linear regression
        log_sizes = np.log(box_sizes[:len(counts)])
        log_counts = np.log(counts)
        coeffs = np.polyfit(log_sizes, log_counts, 1)
        
        return -coeffs[0]
    
    def _blanket_fd(self, img):
        """Blanket method for fractal dimension"""
        # Create upper and lower surfaces
        epsilon_values = [1, 2, 4, 8, 16]
        areas = []
        
        for eps in epsilon_values:
            # Morphological operations to create blanket surfaces
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (eps, eps))
            upper = cv2.morphologyEx(img, cv2.MORPH_DILATE, kernel)
            lower = cv2.morphologyEx(img, cv2.MORPH_ERODE, kernel)
            
            # Calculate blanket area
            area = np.sum(upper - lower)
            areas.append(area)
        
        if len(areas) < 2:
            return 0.0
            
        # Linear regression
        log_eps = np.log(epsilon_values)
        log_areas = np.log(areas)
        coeffs = np.polyfit(log_eps, log_areas, 1)
        
        return 2 - coeffs[0]
    
    def _lacunarity_analysis(self, img):
        """Lacunarity analysis for texture characterization"""
        _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        binary = binary / 255.0
        
        box_sizes = [2, 4, 8, 16, 32]
        lacunarities = []
        
        for box_size in box_sizes:
            if box_size >= min(binary.shape):
                break
                
            masses = []
            h, w = binary.shape
            
            for i in range(0, h - box_size + 1, box_size // 2):
                for j in range(0, w - box_size + 1, box_size // 2):
                    box = binary[i:i+box_size, j:j+box_size]
                    mass = np.sum(box)
                    masses.append(mass)
            
            if len(masses) > 0:
                masses = np.array(masses)
                mean_mass = np.mean(masses)
                var_mass = np.var(masses)
                
                if mean_mass > 0:
                    lacunarity = (var_mass / (mean_mass ** 2)) + 1
                    lacunarities.append(lacunarity)
        
        return np.mean(lacunarities) if lacunarities else 0.0
    
    def _multifractal_analysis(self, img):
        """Multifractal spectrum analysis"""
        # Convert to probability distribution
        img_norm = img.astype(float) / np.sum(img)
        
        # Box sizes
        box_sizes = [2, 4, 8, 16, 32]
        q_values = np.linspace(-5, 5, 21)  # Range of q values
        
        tau_q = []
        
        for q in q_values:
            log_sum = []
            
            for box_size in box_sizes:
                if box_size >= min(img.shape):
                    break
                    
                h, w = img_norm.shape
                partition_sum = 0
                
                for i in range(0, h, box_size):
                    for j in range(0, w, box_size):
                        box = img_norm[i:min(i+box_size, h), j:min(j+box_size, w)]
                        box_sum = np.sum(box)
                        
                        if box_sum > 0:
                            if q != 1:
                                partition_sum += box_sum ** q
                            else:
                                partition_sum += box_sum * np.log(box_sum)
                
                if partition_sum > 0:
                    if q != 1:
                        log_sum.append(np.log(partition_sum))
                    else:
                        log_sum.append(partition_sum)
            
            if len(log_sum) >= 2:
                log_sizes = np.log(box_sizes[:len(log_sum)])
                if q != 1:
                    coeffs = np.polyfit(log_sizes, log_sum, 1)
                    tau_q.append(coeffs[0])
                else:
                    coeffs = np.polyfit(log_sizes, log_sum, 1)
                    tau_q.append(coeffs[0])
        
        # Calculate multifractal spectrum width
        if len(tau_q) > 0:
            return np.max(tau_q) - np.min(tau_q)
        else:
            return 0.0
    
    def _morphological_features(self, img):
        """Extract morphological features"""
        # Binary image for shape analysis
        _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Skeleton analysis
        skeleton = skeletonize(binary > 0)
        skeleton_length = np.sum(skeleton)
        
        # Contour analysis
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            perimeter = cv2.arcLength(largest_contour, True)
            
            # Shape complexity metrics
            if perimeter > 0:
                compactness = (4 * np.pi * area) / (perimeter ** 2)
                form_factor = (4 * np.pi * area) / (perimeter ** 2)
            else:
                compactness = 0
                form_factor = 0
            
            # Convex hull analysis
            hull = cv2.convexHull(largest_contour)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0
            
            return {
                'skeleton_length': skeleton_length,
                'area': area,
                'perimeter': perimeter,
                'compactness': compactness,
                'form_factor': form_factor,
                'solidity': solidity
            }
        
        return {
            'skeleton_length': skeleton_length,
            'area': 0,
            'perimeter': 0,
            'compactness': 0,
            'form_factor': 0,
            'solidity': 0
        }

def main():
    """Example usage of advanced fractal analyzer"""
    analyzer = AdvancedFractalAnalyzer()
    
    # Example with a sample image (you would replace with actual image path)
    import os
    sample_images = []
    
    # Find sample images in PlantVillage directory
    for root, dirs, files in os.walk("./PlantVillage"):
        for file in files[:2]:  # Just analyze first 2 images as example
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                sample_images.append(os.path.join(root, file))
    
    for img_path in sample_images:
        print(f"\nAnalyzing: {img_path}")
        try:
            results = analyzer.analyze_image(img_path)
            
            print("Fractal Dimension Results:")
            for method, value in results.items():
                if method != 'morphological' and value is not None:
                    print(f"  {method}: {value:.4f}")
            
            if results['morphological']:
                print("Morphological Features:")
                for feature, value in results['morphological'].items():
                    print(f"  {feature}: {value:.4f}")
                    
        except Exception as e:
            print(f"Error analyzing {img_path}: {e}")

if __name__ == "__main__":
    main()