"""Custom CNN model for image classification"""

import tensorflow as tf
from .base_model import BaseModel


class CNNModel(BaseModel):
    """Custom CNN architecture"""
    
    def __init__(self, input_shape: tuple, num_classes: int, 
                 filters: list = [32, 64, 128, 256]):
        super().__init__(input_shape, num_classes)
        self.filters = filters
        
    def build(self) -> tf.keras.Model:
        """Build CNN model"""
        inputs = tf.keras.Input(shape=self.input_shape)
        x = inputs
        
        # Convolutional blocks
        for i, num_filters in enumerate(self.filters):
            x = tf.keras.layers.Conv2D(
                num_filters, (3, 3), padding='same',
                name=f'conv_{i+1}_1'
            )(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Activation('relu')(x)
            
            x = tf.keras.layers.Conv2D(
                num_filters, (3, 3), padding='same',
                name=f'conv_{i+1}_2'
            )(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Activation('relu')(x)
            
            x = tf.keras.layers.MaxPooling2D((2, 2))(x)
            x = tf.keras.layers.Dropout(0.25)(x)
        
        # Dense layers
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dense(512, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        x = tf.keras.layers.Dense(256, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        
        # Output layer
        outputs = tf.keras.layers.Dense(
            self.num_classes, 
            activation='softmax' if self.num_classes > 2 else 'sigmoid'
        )(x)
        
        model = tf.keras.Model(inputs=inputs, outputs=outputs, name='cnn_model')
        return model
