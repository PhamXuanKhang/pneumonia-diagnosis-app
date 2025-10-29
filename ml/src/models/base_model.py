"""Base model class"""

from abc import ABC, abstractmethod
import tensorflow as tf


class BaseModel(ABC):
    """Abstract base class for all models"""
    
    def __init__(self, input_shape: tuple, num_classes: int):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        
    @abstractmethod
    def build(self) -> tf.keras.Model:
        """Build the model architecture"""
        pass
    
    def compile(self, optimizer='adam', loss='sparse_categorical_crossentropy',
                metrics=['accuracy']):
        """Compile the model"""
        if self.model is None:
            self.model = self.build()
        
        self.model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics
        )
        
    def summary(self):
        """Print model summary"""
        if self.model is None:
            self.model = self.build()
        return self.model.summary()
