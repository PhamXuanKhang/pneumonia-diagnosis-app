"""
Validation script cho Pipeline 2 TFLite model
So sánh accuracy giữa Keras model và TFLite model
"""

import argparse
import os
import sys
import json
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.efficientnet_p2 import load_trained_p2_model, ConfigP2
from data.preprocessing_p2 import (
    PneumoniaDataGeneratorP2, 
    get_val_test_transforms,
    preprocess_single_image,
    Config as DataConfig
)


def create_test_dataframe(data_root):
    """
    Tạo test DataFrame từ test data
    
    Args:
        data_root: Đường dẫn đến thư mục data
    
    Returns:
        DataFrame với columns ['filepath', 'label']
    """
    import glob
    
    test_data_path = os.path.join(data_root, "test")
    
    if not os.path.exists(test_data_path):
        print(f"Cảnh báo: Không tìm thấy {test_data_path}")
        return None
    
    # Load test data
    test_normal_files = glob.glob(os.path.join(test_data_path, 'NORMAL', '*.jpeg'))
    test_pneumonia_files = glob.glob(os.path.join(test_data_path, 'PNEUMONIA', '*.jpeg'))
    
    test_filepaths = test_normal_files + test_pneumonia_files
    test_labels = ['NORMAL'] * len(test_normal_files) + ['PNEUMONIA'] * len(test_pneumonia_files)
    
    test_df = pd.DataFrame({'filepath': test_filepaths, 'label': test_labels})
    print(f"Test dataset: {len(test_df)} samples")
    print(f"  - NORMAL: {len(test_normal_files)}")
    print(f"  - PNEUMONIA: {len(test_pneumonia_files)}")
    
    return test_df


class TFLiteInferenceEngine:
    """TFLite inference engine"""
    
    def __init__(self, tflite_path):
        self.tflite_path = tflite_path
        self.interpreter = tf.lite.Interpreter(model_path=tflite_path)
        self.interpreter.allocate_tensors()
        
        # Get input and output details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        print(f"TFLite model loaded: {tflite_path}")
        print(f"Input shape: {self.input_details[0]['shape']}")
        print(f"Output shape: {self.output_details[0]['shape']}")
    
    def predict_single(self, input_data):
        """
        Predict single sample
        
        Args:
            input_data: Preprocessed input data shape (1, 224, 224, 3)
        
        Returns:
            prediction: Output từ model
        """
        # Set input tensor
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        
        # Run inference
        self.interpreter.invoke()
        
        # Get output
        output = self.interpreter.get_tensor(self.output_details[0]['index'])
        return output
    
    def predict_batch(self, image_paths, batch_size=32):
        """
        Predict batch of images
        
        Args:
            image_paths: List of image paths
            batch_size: Batch size for processing
        
        Returns:
            predictions: Array of predictions
            inference_times: List of inference times
        """
        predictions = []
        inference_times = []
        
        print(f"Predicting {len(image_paths)} images...")
        
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i+batch_size]
            
            for img_path in batch_paths:
                # Preprocess image
                try:
                    input_data = preprocess_single_image(img_path, target_size=(224, 224))
                    
                    # Measure inference time
                    start_time = time.time()
                    prediction = self.predict_single(input_data)
                    end_time = time.time()
                    
                    predictions.append(prediction[0][0])  # Binary classification
                    inference_times.append(end_time - start_time)
                    
                except Exception as e:
                    print(f"Lỗi xử lý {img_path}: {e}")
                    predictions.append(0.5)  # Default prediction
                    inference_times.append(0.0)
            
            if (i // batch_size + 1) % 10 == 0:
                print(f"  Processed {min(i + batch_size, len(image_paths))}/{len(image_paths)} images")
        
        return np.array(predictions), inference_times


def validate_models(keras_model_path, tflite_model_path, test_df, max_samples=None):
    """
    Validate và so sánh Keras model vs TFLite model
    
    Args:
        keras_model_path: Đường dẫn đến Keras model
        tflite_model_path: Đường dẫn đến TFLite model
        test_df: Test DataFrame
        max_samples: Giới hạn số samples để test (None = all)
    
    Returns:
        results: Dictionary chứa kết quả validation
    """
    print("=" * 80)
    print("MODEL VALIDATION")
    print("=" * 80)
    
    # Limit samples if specified
    if max_samples and len(test_df) > max_samples:
        test_df = test_df.sample(n=max_samples, random_state=42)
        print(f"Limited to {max_samples} samples for validation")
    
    # Prepare data
    image_paths = test_df['filepath'].tolist()
    true_labels = test_df['label'].map(DataConfig.CLASS_INDICES).values
    
    results = {
        'num_samples': len(test_df),
        'keras_model': {},
        'tflite_model': {},
        'comparison': {}
    }
    
    # 1. Keras model validation
    print("\n1. Keras Model Validation")
    print("-" * 40)
    
    try:
        keras_model = load_trained_p2_model(keras_model_path)
        
        # Create data generator
        transforms = get_val_test_transforms(DataConfig.IMG_SIZE)
        test_generator = PneumoniaDataGeneratorP2(
            df=test_df,
            batch_size=32,
            target_size=DataConfig.IMG_SIZE,
            preprocessing_function=tf.keras.applications.efficientnet.preprocess_input,
            augmentation=transforms,
            shuffle=False
        )
        
        # Predict
        print("Predicting với Keras model...")
        start_time = time.time()
        keras_predictions = keras_model.predict(test_generator, verbose=1)
        keras_time = time.time() - start_time
        
        keras_pred_binary = (keras_predictions > 0.5).astype(int).flatten()
        keras_accuracy = accuracy_score(true_labels, keras_pred_binary)
        
        results['keras_model'] = {
            'accuracy': keras_accuracy,
            'total_time': keras_time,
            'avg_time_per_sample': keras_time / len(test_df),
            'predictions': keras_predictions.flatten().tolist()
        }
        
        print(f"✅ Keras Model - Accuracy: {keras_accuracy:.4f}")
        print(f"   Total time: {keras_time:.2f}s")
        print(f"   Avg time per sample: {keras_time/len(test_df)*1000:.2f}ms")
        
    except Exception as e:
        print(f"❌ Lỗi Keras model validation: {e}")
        results['keras_model']['error'] = str(e)
    
    # 2. TFLite model validation
    print("\n2. TFLite Model Validation")
    print("-" * 40)
    
    try:
        tflite_engine = TFLiteInferenceEngine(tflite_model_path)
        
        # Predict
        tflite_predictions, inference_times = tflite_engine.predict_batch(image_paths)
        
        tflite_pred_binary = (tflite_predictions > 0.5).astype(int)
        tflite_accuracy = accuracy_score(true_labels, tflite_pred_binary)
        
        total_tflite_time = sum(inference_times)
        avg_tflite_time = np.mean(inference_times)
        
        results['tflite_model'] = {
            'accuracy': tflite_accuracy,
            'total_time': total_tflite_time,
            'avg_time_per_sample': avg_tflite_time,
            'predictions': tflite_predictions.tolist(),
            'inference_times': inference_times
        }
        
        print(f"✅ TFLite Model - Accuracy: {tflite_accuracy:.4f}")
        print(f"   Total time: {total_tflite_time:.2f}s")
        print(f"   Avg time per sample: {avg_tflite_time*1000:.2f}ms")
        
    except Exception as e:
        print(f"❌ Lỗi TFLite model validation: {e}")
        results['tflite_model']['error'] = str(e)
    
    # 3. Comparison
    print("\n3. Model Comparison")
    print("-" * 40)
    
    if 'error' not in results['keras_model'] and 'error' not in results['tflite_model']:
        accuracy_diff = abs(results['keras_model']['accuracy'] - results['tflite_model']['accuracy'])
        speed_ratio = results['keras_model']['avg_time_per_sample'] / results['tflite_model']['avg_time_per_sample']
        
        results['comparison'] = {
            'accuracy_difference': accuracy_diff,
            'speed_ratio': speed_ratio,
            'accuracy_retention': results['tflite_model']['accuracy'] / results['keras_model']['accuracy']
        }
        
        print(f"Accuracy difference: {accuracy_diff:.4f}")
        print(f"Accuracy retention: {results['comparison']['accuracy_retention']:.4f} ({results['comparison']['accuracy_retention']*100:.2f}%)")
        print(f"Speed improvement: {speed_ratio:.2f}x")
        
        # Classification report
        print("\n4. Classification Report (TFLite)")
        print("-" * 40)
        report = classification_report(
            true_labels, 
            tflite_pred_binary, 
            target_names=DataConfig.CLASS_NAMES
        )
        print(report)
        
        # Confusion matrix
        cm = confusion_matrix(true_labels, tflite_pred_binary)
        print("\nConfusion Matrix (TFLite):")
        print(f"                Predicted")
        print(f"              Normal  Pneumonia")
        print(f"Actual Normal    {cm[0,0]:4d}      {cm[0,1]:4d}")
        print(f"    Pneumonia    {cm[1,0]:4d}      {cm[1,1]:4d}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Validate Pipeline 2 TFLite model')
    parser.add_argument(
        '--keras_model', 
        type=str, 
        default='ml/models/saved_models/P2_EffNetB0_Baseline_final.keras',
        help='Path to Keras model'
    )
    parser.add_argument(
        '--tflite_model', 
        type=str, 
        default='ml/models/tflite/pneumonia_efficientnet_p2.tflite',
        help='Path to TFLite model'
    )
    parser.add_argument(
        '--data_root', 
        type=str, 
        required=True,
        help='Path to data root directory'
    )
    parser.add_argument(
        '--max_samples', 
        type=int, 
        default=None,
        help='Maximum number of samples to validate (None = all)'
    )
    parser.add_argument(
        '--output_dir', 
        type=str, 
        default='ml/outputs',
        help='Output directory for results'
    )
    
    args = parser.parse_args()
    
    # Create test dataframe
    test_df = create_test_dataframe(args.data_root)
    if test_df is None:
        print("❌ Không thể tạo test dataset")
        return
    
    # Validate models
    results = validate_models(
        keras_model_path=args.keras_model,
        tflite_model_path=args.tflite_model,
        test_df=test_df,
        max_samples=args.max_samples
    )
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_path = output_dir / 'p2_validation_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Validation results saved to: {results_path}")
    
    # Summary
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)
    
    if 'error' not in results['keras_model'] and 'error' not in results['tflite_model']:
        print(f"✅ Validation thành công!")
        print(f"   Keras accuracy: {results['keras_model']['accuracy']:.4f}")
        print(f"   TFLite accuracy: {results['tflite_model']['accuracy']:.4f}")
        print(f"   Accuracy retention: {results['comparison']['accuracy_retention']*100:.2f}%")
        print(f"   Speed improvement: {results['comparison']['speed_ratio']:.2f}x")
        
        if results['comparison']['accuracy_retention'] > 0.95:
            print("🎉 TFLite model đạt yêu cầu accuracy retention > 95%")
        else:
            print("⚠️  TFLite model có accuracy retention < 95%")
    else:
        print("❌ Validation thất bại")
        if 'error' in results['keras_model']:
            print(f"   Keras error: {results['keras_model']['error']}")
        if 'error' in results['tflite_model']:
            print(f"   TFLite error: {results['tflite_model']['error']}")


if __name__ == '__main__':
    main()
