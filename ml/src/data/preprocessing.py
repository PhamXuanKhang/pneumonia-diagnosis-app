"""Image preprocessing utilities"""

import tensorflow as tf


class ImagePreprocessor:
    """Preprocess images for model input"""
    
    @staticmethod
    def normalize(image: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """Normalize pixel values to [0, 1]"""
        image = tf.cast(image, tf.float32) / 255.0
        return image, label
    
    @staticmethod
    def standardize(image: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """Standardize using ImageNet mean and std"""
        mean = tf.constant([0.485, 0.456, 0.406])
        std = tf.constant([0.229, 0.224, 0.225])
        image = tf.cast(image, tf.float32) / 255.0
        image = (image - mean) / std
        return image, label
    
    @staticmethod
    def resize_and_rescale(target_size: Tuple[int, int]):
        """Create a preprocessing layer for resizing and rescaling"""
        return tf.keras.Sequential([
            tf.keras.layers.Resizing(target_size[0], target_size[1]),
            tf.keras.layers.Rescaling(1./255)
        ])
