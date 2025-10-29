"""Convert TensorFlow model to TFLite"""

import argparse
import tensorflow as tf
from pathlib import Path


def convert_to_tflite(model_path: str, output_path: str, 
                     quantize: bool = False):
    """
    Convert SavedModel to TFLite format
    
    Args:
        model_path: Path to SavedModel directory
        output_path: Output path for TFLite model
        quantize: Apply quantization for smaller model size
    """
    print(f"Loading model from {model_path}")
    
    # Convert the model
    converter = tf.lite.TFLiteConverter.from_saved_model(model_path)
    
    if quantize:
        print("Applying quantization...")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
    
    tflite_model = converter.convert()
    
    # Save the model
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    
    print(f"TFLite model saved to {output_path}")
    print(f"Model size: {len(tflite_model) / 1024 / 1024:.2f} MB")


def test_tflite_model(tflite_path: str, input_shape: tuple):
    """Test TFLite model inference"""
    import numpy as np
    
    print(f"\nTesting TFLite model: {tflite_path}")
    
    # Load TFLite model
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    
    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print(f"Input shape: {input_details[0]['shape']}")
    print(f"Output shape: {output_details[0]['shape']}")
    
    # Test with random input
    test_input = np.random.random(input_shape).astype(np.float32)
    interpreter.set_tensor(input_details[0]['index'], test_input)
    
    interpreter.invoke()
    
    output = interpreter.get_tensor(output_details[0]['index'])
    print(f"Test inference successful! Output shape: {output.shape}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert TensorFlow model to TFLite')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to SavedModel directory')
    parser.add_argument('--output_path', type=str, required=True,
                       help='Output path for TFLite model')
    parser.add_argument('--quantize', action='store_true',
                       help='Apply quantization')
    parser.add_argument('--test', action='store_true',
                       help='Test the converted model')
    parser.add_argument('--input_shape', type=str, default='1,224,224,3',
                       help='Input shape for testing (comma-separated)')
    
    args = parser.parse_args()
    
    convert_to_tflite(args.model_path, args.output_path, args.quantize)
    
    if args.test:
        input_shape = tuple(map(int, args.input_shape.split(',')))
        test_tflite_model(args.output_path, input_shape)
