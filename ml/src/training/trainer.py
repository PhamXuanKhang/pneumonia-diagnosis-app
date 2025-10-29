"""Model training utilities"""

import tensorflow as tf
from pathlib import Path
from typing import Optional, List


class Trainer:
    """Handle model training"""
    
    def __init__(self, model: tf.keras.Model, model_dir: str):
        self.model = model
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.history = None
        
    def train(self, 
              train_dataset: tf.data.Dataset,
              val_dataset: Optional[tf.data.Dataset] = None,
              epochs: int = 50,
              callbacks: Optional[List[tf.keras.callbacks.Callback]] = None):
        """
        Train the model
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            epochs: Number of epochs
            callbacks: List of callbacks
            
        Returns:
            Training history
        """
        self.history = self.model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs,
            callbacks=callbacks or []
        )
        
        return self.history
    
    def save_model(self, model_name: str = 'model'):
        """Save model in SavedModel format"""
        save_path = self.model_dir / 'saved_models' / model_name
        self.model.save(save_path)
        print(f"Model saved to {save_path}")
        
    def save_weights(self, weights_name: str = 'weights'):
        """Save model weights"""
        save_path = self.model_dir / 'checkpoints' / f'{weights_name}.h5'
        save_path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save_weights(save_path)
        print(f"Weights saved to {save_path}")
        
    def load_weights(self, weights_path: str):
        """Load model weights"""
        self.model.load_weights(weights_path)
        print(f"Weights loaded from {weights_path}")
