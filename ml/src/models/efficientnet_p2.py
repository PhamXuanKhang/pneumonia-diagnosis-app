"""
Model architecture cho Pipeline 2 (EfficientNetB0 Baseline)
Trích xuất từ dat301m-training-pipeline-comparation.py Cell 6
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, Model
from tensorflow.keras.layers import (
    Input, GlobalAveragePooling2D, Dense, Dropout
)
from tensorflow.keras.applications import EfficientNetB0
import os


class ConfigP2:
    """Configuration cho Pipeline 2"""
    INPUT_SHAPE = (224, 224, 3)
    MODEL_NAME = "P2_EffNetB0_Baseline"
    FEATURE_LAYER_NAME = 'top_conv'  # EfficientNetB0 feature layer (7, 7, 1280)


def build_common_head(input_tensor):
    """Đầu phân loại chung cho Pipeline 2"""
    # Global Average Pooling
    x = GlobalAveragePooling2D(name='gap')(input_tensor)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    output = Dense(1, activation='sigmoid', name='output')(x)
    return output


def build_efficientnet_p2_architecture(input_shape=ConfigP2.INPUT_SHAPE):
    """
    Xây dựng kiến trúc Pipeline 2 (EfficientNetB0 Baseline)
    
    Args:
        input_shape: Shape của input (224, 224, 3)
    
    Returns:
        model: Keras model
        base_model: EfficientNetB0 base model
    """
    # 1. Tạo EfficientNetB0 base model
    base_model = EfficientNetB0(
        weights='imagenet', 
        include_top=False, 
        input_shape=input_shape
    )
    base_model.trainable = False
    
    # 2. Xây dựng model
    image_input = Input(shape=input_shape, name='image_input')
    
    # Trích xuất feature map từ top_conv layer
    feature_extractor = Model(
        inputs=base_model.input, 
        outputs=base_model.get_layer(ConfigP2.FEATURE_LAYER_NAME).output,
        name=f"efficientnetb0_feature_extractor"
    )
    feature_map = feature_extractor(image_input)
    
    # 3. Thêm classification head
    outputs = build_common_head(feature_map)
    model = Model(inputs=image_input, outputs=outputs, name=ConfigP2.MODEL_NAME)
    
    return model, base_model


def build_efficientnet_p2_multi_output(input_shape=ConfigP2.INPUT_SHAPE):
    """
    Xây dựng Pipeline 2 với multiple outputs cho visualization
    
    Args:
        input_shape: Shape của input (224, 224, 3)
    
    Returns:
        model: Keras model với 2 outputs [classification, feature_maps]
        base_model: EfficientNetB0 base model
    """
    # 1. Tạo EfficientNetB0 base model
    base_model = EfficientNetB0(
        weights='imagenet', 
        include_top=False, 
        input_shape=input_shape
    )
    base_model.trainable = False
    
    # 2. Xây dựng model
    image_input = Input(shape=input_shape, name='image_input')
    
    # Trích xuất feature map từ top_conv layer
    feature_extractor = Model(
        inputs=base_model.input, 
        outputs=base_model.get_layer(ConfigP2.FEATURE_LAYER_NAME).output,
        name=f"efficientnetb0_feature_extractor"
    )
    feature_map = feature_extractor(image_input)
    
    # 3. Classification output
    classification_output = build_common_head(feature_map)
    
    # 4. Feature map output (cho visualization)
    feature_output = layers.Lambda(
        lambda x: x, 
        name='feature_maps'
    )(feature_map)
    
    # 5. Model với multiple outputs
    model = Model(
        inputs=image_input, 
        outputs=[classification_output, feature_output],
        name=f"{ConfigP2.MODEL_NAME}_MultiOutput"
    )
    
    return model, base_model


def load_trained_p2_model(model_path):
    """
    Load trained Pipeline 2 model
    
    Args:
        model_path: Đường dẫn đến file .keras
    
    Returns:
        model: Loaded Keras model
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model không tồn tại tại: {model_path}")
    
    try:
        model = keras.models.load_model(model_path)
        print(f"✅ Đã load model từ: {model_path}")
        return model
    except Exception as e:
        raise RuntimeError(f"Lỗi khi load model: {e}")


def create_p2_model_from_weights(weights_path, multi_output=False):
    """
    Tạo model P2 từ architecture và load weights
    
    Args:
        weights_path: Đường dẫn đến file weights (.keras)
        multi_output: Có tạo model với multiple outputs không
    
    Returns:
        model: Keras model với weights đã load
    """
    # 1. Tạo architecture
    if multi_output:
        model, base_model = build_efficientnet_p2_multi_output()
    else:
        model, base_model = build_efficientnet_p2_architecture()
    
    # 2. Load trained model để lấy weights
    trained_model = load_trained_p2_model(weights_path)
    
    # 3. Transfer weights từ trained model
    if multi_output:
        # Với multi-output, chỉ copy weights cho classification branch
        for layer in model.layers:
            if layer.name in [l.name for l in trained_model.layers]:
                try:
                    trained_layer = trained_model.get_layer(layer.name)
                    layer.set_weights(trained_layer.get_weights())
                except:
                    continue
    else:
        # Với single output, copy toàn bộ weights
        model.set_weights(trained_model.get_weights())
    
    print(f"✅ Đã tạo model P2 và load weights từ: {weights_path}")
    return model


def get_model_summary(model):
    """
    In summary của model
    
    Args:
        model: Keras model
    """
    print("=" * 80)
    print(f"MODEL SUMMARY: {model.name}")
    print("=" * 80)
    model.summary()
    
    print("\n" + "=" * 80)
    print("INPUT/OUTPUT DETAILS")
    print("=" * 80)
    print(f"Input shape: {model.input_shape}")
    if isinstance(model.output, list):
        for i, output in enumerate(model.output):
            print(f"Output {i+1} shape: {output.shape}")
    else:
        print(f"Output shape: {model.output_shape}")


if __name__ == "__main__":
    # Test model building
    print("Testing Pipeline 2 model architecture...")
    
    # Test single output model
    model, base_model = build_efficientnet_p2_architecture()
    get_model_summary(model)
    
    print("\n" + "="*50)
    
    # Test multi-output model
    multi_model, _ = build_efficientnet_p2_multi_output()
    get_model_summary(multi_model)
