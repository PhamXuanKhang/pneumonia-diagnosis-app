"""Data augmentation utilities"""

import tensorflow as tf


class DataAugmentor:
    """Apply data augmentation to images"""
    
    @staticmethod
    def get_augmentation_layer():
        """Create data augmentation layer"""
        return tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.1),
            tf.keras.layers.RandomZoom(0.1),
            tf.keras.layers.RandomContrast(0.1),
        ])
    
    @staticmethod
    def augment_dataset(dataset: tf.data.Dataset, 
                       augmentation_layer: tf.keras.Sequential) -> tf.data.Dataset:
        """Apply augmentation to dataset"""
        return dataset.map(
            lambda x, y: (augmentation_layer(x, training=True), y),
            num_parallel_calls=tf.data.AUTOTUNE
        )
