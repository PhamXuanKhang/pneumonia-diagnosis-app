"""Data loading utilities"""

import os
import tensorflow as tf
from pathlib import Path
from typing import Tuple, Optional


class DataLoader:
    """Load and prepare image datasets for training"""
    
    def __init__(self, data_dir: str, img_size: Tuple[int, int] = (224, 224), 
                 batch_size: int = 32):
        self.data_dir = Path(data_dir)
        self.img_size = img_size
        self.batch_size = batch_size
        
    def load_dataset(self, subset: str = 'train', 
                     validation_split: Optional[float] = None) -> tf.data.Dataset:
        """
        Load dataset from directory
        
        Args:
            subset: 'train', 'validation', or 'test'
            validation_split: Fraction of data to use for validation
            
        Returns:
            tf.data.Dataset
        """
        if validation_split:
            dataset = tf.keras.preprocessing.image_dataset_from_directory(
                self.data_dir,
                validation_split=validation_split,
                subset=subset,
                seed=42,
                image_size=self.img_size,
                batch_size=self.batch_size
            )
        else:
            dataset = tf.keras.preprocessing.image_dataset_from_directory(
                self.data_dir,
                image_size=self.img_size,
                batch_size=self.batch_size
            )
        
        return dataset
    
    def get_class_names(self) -> list:
        """Get class names from directory structure"""
        return sorted([d.name for d in self.data_dir.iterdir() if d.is_dir()])
