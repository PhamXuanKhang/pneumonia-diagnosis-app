"""
Utility script để tạo dummy P2 model cho testing
Sử dụng khi chưa có trained model thực tế
"""

import os
import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import tensorflow as tf
from models.efficientnet_p2 import build_efficientnet_p2_architecture, ConfigP2


def create_dummy_p2_model(output_path, compile_model=True):
    """
    Tạo dummy P2 model với random weights
    
    Args:
        output_path: Đường dẫn lưu model
        compile_model: Có compile model không
    
    Returns:
        model_path: Đường dẫn đến model đã lưu
    """
    print("=" * 80)
    print("CREATING DUMMY PIPELINE 2 MODEL")
    print("=" * 80)
    print(f"Output path: {output_path}")
    print("=" * 80)
    
    # 1. Build model architecture
    print("\n1. Building model architecture...")
    model, base_model = build_efficientnet_p2_architecture()
    
    print(f"✅ Model created: {model.name}")
    print(f"   Input shape: {model.input_shape}")
    print(f"   Output shape: {model.output_shape}")
    print(f"   Total parameters: {model.count_params():,}")
    
    # 2. Compile model if requested
    if compile_model:
        print("\n2. Compiling model...")
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        print("✅ Model compiled")
    
    # 3. Create dummy training history (for metadata)
    dummy_history = {
        'accuracy': [0.7, 0.8, 0.85, 0.9, 0.92],
        'val_accuracy': [0.65, 0.75, 0.8, 0.85, 0.87],
        'loss': [0.6, 0.4, 0.3, 0.2, 0.15],
        'val_loss': [0.7, 0.5, 0.4, 0.3, 0.25]
    }
    
    # 4. Save model
    print("\n3. Saving model...")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    model.save(output_path)
    print(f"✅ Dummy model saved: {output_path}")
    
    # 5. Save metadata
    import json
    metadata = {
        'model_name': ConfigP2.MODEL_NAME,
        'model_type': 'dummy',
        'architecture': 'EfficientNetB0',
        'input_shape': list(model.input_shape),
        'output_shape': list(model.output_shape),
        'total_parameters': int(model.count_params()),
        'trainable_parameters': int(sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])),
        'training_history': dummy_history,
        'note': 'This is a dummy model with random weights for testing purposes'
    }
    
    metadata_path = output_path.with_suffix('.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✅ Metadata saved: {metadata_path}")
    
    # 6. Model summary
    print("\n4. Model Summary:")
    print("-" * 40)
    model.summary()
    
    print("\n" + "=" * 80)
    print("DUMMY MODEL CREATION COMPLETED")
    print("=" * 80)
    print(f"Model file: {output_path}")
    print(f"Metadata file: {metadata_path}")
    print(f"Model size: {output_path.stat().st_size / (1024*1024):.2f} MB")
    print("\n⚠️  WARNING: This is a dummy model with random weights!")
    print("   Use only for testing conversion and inference pipelines.")
    print("=" * 80)
    
    return str(output_path)


def test_dummy_model(model_path):
    """
    Test dummy model với random input
    
    Args:
        model_path: Đường dẫn đến model
    """
    print("\n" + "=" * 80)
    print("TESTING DUMMY MODEL")
    print("=" * 80)
    
    # Load model
    model = tf.keras.models.load_model(model_path)
    print(f"✅ Model loaded: {model.name}")
    
    # Create random input
    import numpy as np
    input_shape = (1, 224, 224, 3)
    test_input = np.random.random(input_shape).astype(np.float32)
    
    # EfficientNet preprocessing simulation
    test_input = test_input * 2.0 - 1.0  # [0,1] -> [-1,1]
    
    print(f"Test input shape: {test_input.shape}")
    print(f"Test input range: [{test_input.min():.3f}, {test_input.max():.3f}]")
    
    # Predict
    prediction = model.predict(test_input, verbose=0)
    print(f"Prediction shape: {prediction.shape}")
    print(f"Prediction value: {prediction[0][0]:.6f}")
    print(f"Predicted class: {'PNEUMONIA' if prediction[0][0] > 0.5 else 'NORMAL'}")
    
    print("✅ Dummy model test successful!")


def main():
    parser = argparse.ArgumentParser(description='Create dummy P2 model for testing')
    parser.add_argument(
        '--output_path', 
        type=str, 
        default='ml/models/saved_models/P2_EffNetB0_Baseline_final.keras',
        help='Output path for dummy model'
    )
    parser.add_argument(
        '--no_compile', 
        action='store_true',
        help='Do not compile the model'
    )
    parser.add_argument(
        '--test', 
        action='store_true',
        help='Test the created model'
    )
    
    args = parser.parse_args()
    
    # Create dummy model
    model_path = create_dummy_p2_model(
        output_path=args.output_path,
        compile_model=not args.no_compile
    )
    
    # Test model if requested
    if args.test:
        test_dummy_model(model_path)
    
    print(f"\n🎉 Dummy model creation completed!")
    print(f"📁 Model saved at: {model_path}")


if __name__ == '__main__':
    main()
