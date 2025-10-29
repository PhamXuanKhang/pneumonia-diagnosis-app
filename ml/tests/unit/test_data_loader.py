"""Unit tests for data loader"""

import pytest
import tensorflow as tf
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent / 'src'))

from src.data.data_loader import DataLoader


class TestDataLoader:
    """Test DataLoader class"""
    
    def test_init(self):
        """Test DataLoader initialization"""
        loader = DataLoader(
            data_dir="data/processed/train",
            img_size=(224, 224),
            batch_size=32
        )
        
        assert loader.img_size == (224, 224)
        assert loader.batch_size == 32
        assert loader.data_dir.name == "train"
    
    def test_img_size_tuple(self):
        """Test image size is tuple"""
        loader = DataLoader("data/processed/train")
        assert isinstance(loader.img_size, tuple)
        assert len(loader.img_size) == 2
