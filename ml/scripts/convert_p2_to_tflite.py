"""
Convert Pipeline 2 (EfficientNetB0) model to TFLite format
Refactored từ dat301m-training-pipeline-comparation.py cho inference
"""

import argparse
import os
import sys
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.efficientnet_p2 import (
    load_trained_p2_model, 
    create_p2_model_from_weights,
    build_efficientnet_p2_multi_output,
    ConfigP2
)
from data.preprocessing_p2 import (
    create_representative_dataset_generator,
    Config as DataConfig
)


def create_sample_dataframe(data_root, num_samples=100):
    """
    Tạo sample DataFrame từ test data cho representative dataset
    
    Args:
        data_root: Đường dẫn đến thư mục data
        num_samples: Số lượng samples
    
    Returns:
        DataFrame với columns ['filepath', 'label']
    """
    import glob
    
    test_data_path = os.path.join(data_root, "test")
    
    if not os.path.exists(test_data_path):
        print(f"Cảnh báo: Không tìm thấy {test_data_path}")
        print("Tạo dummy data cho representative dataset...")
        
        # Tạo dummy data nếu không có test data
        dummy_data = {
            'filepath': [f'dummy_path_{i}.jpg' for i in range(num_samples)],
            'label': ['NORMAL' if i % 2 == 0 else 'PNEUMONIA' for i in range(num_samples)]
        }
        return pd.DataFrame(dummy_data)
    
    # Load real test data
    test_normal_files = glob.glob(os.path.join(test_data_path, 'NORMAL', '*.jpeg'))
    test_pneumonia_files = glob.glob(os.path.join(test_data_path, 'PNEUMONIA', '*.jpeg'))
    
    # Lấy sample từ mỗi class
    normal_sample = test_normal_files[:min(num_samples//2, len(test_normal_files))]
    pneumonia_sample = test_pneumonia_files[:min(num_samples//2, len(test_pneumonia_files))]
    
    filepaths = normal_sample + pneumonia_sample
    labels = ['NORMAL'] * len(normal_sample) + ['PNEUMONIA'] * len(pneumonia_sample)
    
    df = pd.DataFrame({'filepath': filepaths, 'label': labels})
    print(f"Tạo representative dataset với {len(df)} samples")
    
    return df


def convert_p2_to_tflite(
    model_path: str,
    output_path: str,
    data_root: str = None,
    quantize: bool = True,
    multi_output: bool = False
):
    """
    Convert Pipeline 2 model to TFLite
    
    Args:
        model_path: Đường dẫn đến P2 model (.keras)
        output_path: Đường dẫn output (.tflite)
        data_root: Đường dẫn đến data root (cho representative dataset)
        quantize: Có áp dụng quantization không
        multi_output: Có tạo model với multiple outputs không
    """
    print("=" * 80)
    print("CONVERT PIPELINE 2 TO TFLITE")
    print("=" * 80)
    print(f"Model path: {model_path}")
    print(f"Output path: {output_path}")
    print(f"Quantization: {quantize}")
    print(f"Multi-output: {multi_output}")
    print("=" * 80)
    
    # 1. Load model
    if multi_output:
        print("\n1. Tạo multi-output model từ trained weights...")
        model = create_p2_model_from_weights(model_path, multi_output=True)
    else:
        print("\n1. Load trained model...")
        model = load_trained_p2_model(model_path)
    
    print(f"✅ Model loaded: {model.name}")
    print(f"Input shape: {model.input_shape}")
    if isinstance(model.output, list):
        for i, output in enumerate(model.output):
            print(f"Output {i+1} shape: {output.shape}")
    else:
        print(f"Output shape: {model.output_shape}")
    
    # 2. Tạo TFLite converter
    print("\n2. Tạo TFLite converter...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # 3. Cấu hình optimization
    if quantize:
        print("3. Cấu hình quantization...")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        # Tạo representative dataset nếu có data
        if data_root and os.path.exists(data_root):
            print("   Tạo representative dataset...")
            sample_df = create_sample_dataframe(data_root, num_samples=100)
            
            # Chỉ tạo representative dataset nếu có real data
            if not sample_df['filepath'].str.contains('dummy_path').any():
                representative_data_gen = create_representative_dataset_generator(
                    sample_df, num_samples=50
                )
                converter.representative_dataset = representative_data_gen
                print("   ✅ Representative dataset được tạo")
            else:
                print("   ⚠️  Không có real data, bỏ qua representative dataset")
        else:
            print("   ⚠️  Không có data root, bỏ qua representative dataset")
    else:
        print("3. Bỏ qua quantization")
    
    # 4. Convert
    print("\n4. Converting to TFLite...")
    try:
        tflite_model = converter.convert()
        print("✅ Conversion thành công!")
    except Exception as e:
        print(f"❌ Lỗi conversion: {e}")
        raise
    
    # 5. Lưu model
    print("\n5. Lưu TFLite model...")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    
    model_size_mb = len(tflite_model) / (1024 * 1024)
    print(f"✅ TFLite model đã lưu tại: {output_path}")
    print(f"📊 Kích thước model: {model_size_mb:.2f} MB")
    
    # 6. Tạo metadata
    metadata = {
        "model_name": "P2_EffNetB0_Baseline",
        "input_shape": list(model.input_shape),
        "output_shapes": [list(output.shape) for output in model.output] if isinstance(model.output, list) else [list(model.output_shape)],
        "preprocessing": "EfficientNet",
        "quantized": quantize,
        "multi_output": multi_output,
        "model_size_mb": round(model_size_mb, 2)
    }
    
    metadata_path = output_path.with_suffix('.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✅ Metadata đã lưu tại: {metadata_path}")
    
    return str(output_path), metadata


def test_tflite_model(tflite_path: str):
    """Test TFLite model inference"""
    print("\n" + "=" * 80)
    print("TESTING TFLITE MODEL")
    print("=" * 80)
    
    # Load TFLite model
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    
    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print(f"Input details:")
    for i, detail in enumerate(input_details):
        print(f"  Input {i}: {detail['name']} - Shape: {detail['shape']} - Type: {detail['dtype']}")
    
    print(f"Output details:")
    for i, detail in enumerate(output_details):
        print(f"  Output {i}: {detail['name']} - Shape: {detail['shape']} - Type: {detail['dtype']}")
    
    # Test với random input
    print("\nTesting với random input...")
    input_shape = input_details[0]['shape']
    test_input = np.random.random(input_shape).astype(np.float32)
    
    # EfficientNet preprocessing simulation: [0,1] -> [-1,1]
    test_input = test_input * 2.0 - 1.0
    
    interpreter.set_tensor(input_details[0]['index'], test_input)
    interpreter.invoke()
    
    # Get outputs
    outputs = []
    for detail in output_details:
        output = interpreter.get_tensor(detail['index'])
        outputs.append(output)
        print(f"Output shape: {output.shape}, Sample values: {output.flatten()[:5]}")
    
    print("✅ Test inference thành công!")
    return outputs


def main():
    parser = argparse.ArgumentParser(description='Convert Pipeline 2 to TFLite')
    parser.add_argument(
        '--model_path', 
        type=str, 
        default='ml/models/saved_models/P2_EffNetB0_Baseline_final.keras',
        help='Path to P2 model (.keras)'
    )
    parser.add_argument(
        '--output_path', 
        type=str, 
        default='ml/models/tflite/pneumonia_efficientnet_p2.tflite',
        help='Output path for TFLite model'
    )
    parser.add_argument(
        '--data_root', 
        type=str, 
        default=None,
        help='Path to data root (for representative dataset)'
    )
    parser.add_argument(
        '--quantize', 
        action='store_true', 
        default=True,
        help='Apply quantization'
    )
    parser.add_argument(
        '--multi_output', 
        action='store_true',
        help='Create model with multiple outputs (classification + feature maps)'
    )
    parser.add_argument(
        '--test', 
        action='store_true',
        help='Test the converted model'
    )
    
    args = parser.parse_args()
    
    # Convert model
    tflite_path, metadata = convert_p2_to_tflite(
        model_path=args.model_path,
        output_path=args.output_path,
        data_root=args.data_root,
        quantize=args.quantize,
        multi_output=args.multi_output
    )
    
    # Test model if requested
    if args.test:
        test_tflite_model(tflite_path)
    
    print("\n" + "=" * 80)
    print("CONVERSION COMPLETED")
    print("=" * 80)
    print(f"TFLite model: {tflite_path}")
    print(f"Metadata: {Path(tflite_path).with_suffix('.json')}")
    print(f"Model size: {metadata['model_size_mb']} MB")


if __name__ == '__main__':
    main()
