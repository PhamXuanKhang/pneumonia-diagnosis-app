"""Transfer learning models"""

import tensorflow as tf
from .base_model import BaseModel


class TransferLearningModel(BaseModel):
    """Transfer learning using pre-trained models"""
    
    AVAILABLE_MODELS = {
        'mobilenetv2': tf.keras.applications.MobileNetV2,
        'resnet50': tf.keras.applications.ResNet50,
        'efficientnetb0': tf.keras.applications.EfficientNetB0,
        'inceptionv3': tf.keras.applications.InceptionV3,
    }
    
    def __init__(self, input_shape: tuple, num_classes: int, 
                 base_model_name: str = 'mobilenetv2',
                 trainable_base: bool = False):
        super().__init__(input_shape, num_classes)
        self.base_model_name = base_model_name.lower()
        self.trainable_base = trainable_base
        
        if self.base_model_name not in self.AVAILABLE_MODELS:
            raise ValueError(f"Model {base_model_name} not available. "
                           f"Choose from {list(self.AVAILABLE_MODELS.keys())}")
    
    def build(self) -> tf.keras.Model:
        """Build transfer learning model"""
        base_model_class = self.AVAILABLE_MODELS[self.base_model_name]
        
        # Load pre-trained model
        base_model = base_model_class(
            input_shape=self.input_shape,
            include_top=False,
            weights='imagenet'
        )
        base_model.trainable = self.trainable_base
        
        # Build model
        inputs = tf.keras.Input(shape=self.input_shape)
        x = base_model(inputs, training=False)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dense(512, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        x = tf.keras.layers.Dense(256, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        
        outputs = tf.keras.layers.Dense(
            self.num_classes,
            activation='softmax' if self.num_classes > 2 else 'sigmoid'
        )(x)
        
        model = tf.keras.Model(inputs=inputs, outputs=outputs, 
                              name=f'{self.base_model_name}_transfer')
        return model
    
    def unfreeze_base_model(self, num_layers: int = None):
        """Unfreeze base model layers for fine-tuning"""
        if self.model is None:
            raise ValueError("Model must be built before unfreezing")
        
        base_model = self.model.layers[1]
        base_model.trainable = True
        
        if num_layers:
            # Freeze all layers except the last num_layers
            for layer in base_model.layers[:-num_layers]:
                layer.trainable = False
