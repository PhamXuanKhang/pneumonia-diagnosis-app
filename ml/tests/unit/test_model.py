"""Unit tests for models"""

import pytest
import tensorflow as tf
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent / 'src'))

from src.models.cnn_model import CNNModel
from src.models.transfer_learning import TransferLearningModel


class TestCNNModel:
    """Test CNNModel class"""
    
    def test_build_model(self):
        """Test model building"""
        model = CNNModel(
            input_shape=(224, 224, 3),
            num_classes=2
        )
        
        built_model = model.build()
        
        assert isinstance(built_model, tf.keras.Model)
        assert built_model.input_shape == (None, 224, 224, 3)
        assert built_model.output_shape == (None, 2)


class TestTransferLearningModel:
    """Test TransferLearningModel class"""
    
    def test_build_mobilenetv2(self):
        """Test building MobileNetV2 model"""
        model = TransferLearningModel(
            input_shape=(224, 224, 3),
            num_classes=2,
            base_model_name='mobilenetv2'
        )
        
        built_model = model.build()
        
        assert isinstance(built_model, tf.keras.Model)
        assert built_model.input_shape == (None, 224, 224, 3)
    
    def test_invalid_base_model(self):
        """Test invalid base model name"""
        with pytest.raises(ValueError):
            model = TransferLearningModel(
                input_shape=(224, 224, 3),
                num_classes=2,
                base_model_name='invalid_model'
            )
