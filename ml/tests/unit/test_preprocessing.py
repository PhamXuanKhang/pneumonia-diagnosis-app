"""Unit tests for preprocessing"""

import pytest
import tensorflow as tf
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent / 'src'))

from src.data.preprocessing import ImagePreprocessor


class TestImagePreprocessor:
    """Test ImagePreprocessor class"""
    
    def test_normalize(self):
        """Test normalization"""
        image = tf.constant(np.random.randint(0, 255, (224, 224, 3)), dtype=tf.uint8)
        label = tf.constant(0)
        
        normalized_image, normalized_label = ImagePreprocessor.normalize(image, label)
        
        assert normalized_image.dtype == tf.float32
        assert tf.reduce_max(normalized_image) <= 1.0
        assert tf.reduce_min(normalized_image) >= 0.0
        assert normalized_label == label
    
    def test_resize_and_rescale(self):
        """Test resize and rescale layer"""
        layer = ImagePreprocessor.resize_and_rescale((224, 224))
        
        assert isinstance(layer, tf.keras.Sequential)
        assert len(layer.layers) == 2
