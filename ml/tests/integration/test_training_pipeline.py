"""Integration tests for training pipeline"""

import pytest
import tensorflow as tf
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent / 'src'))

from src.models.cnn_model import CNNModel
from src.training.trainer import Trainer


class TestTrainingPipeline:
    """Test complete training pipeline"""
    
    def test_model_training(self, tmp_path):
        """Test model training with dummy data"""
        # Create dummy dataset
        x_train = np.random.random((100, 64, 64, 3)).astype(np.float32)
        y_train = np.random.randint(0, 2, 100)
        
        train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        train_ds = train_ds.batch(16)
        
        # Build model
        model = CNNModel(
            input_shape=(64, 64, 3),
            num_classes=2,
            filters=[16, 32]
        )
        model.compile()
        
        # Train
        trainer = Trainer(model.model, str(tmp_path))
        history = trainer.train(
            train_dataset=train_ds,
            epochs=2,
            callbacks=[]
        )
        
        assert history is not None
        assert 'loss' in history.history
        assert 'accuracy' in history.history
