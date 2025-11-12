"""
Preprocessing module cho Pipeline 2 (EfficientNetB0 Baseline)
Trích xuất từ dat301m-training-pipeline-comparation.py Cell 5
"""

import numpy as np
import cv2
import tensorflow as tf
import tensorflow.keras.applications.efficientnet as effnet_preprocess
from tensorflow.keras.utils import Sequence
import albumentations as A
import os
import glob


class Config:
    """Configuration cho Pipeline 2"""
    # Paths
    DATA_ROOT = "/kaggle/input/chest-xray-pneumonia/chest_xray"
    
    # Data Params
    IMG_SIZE = (224, 224)
    INPUT_SHAPE = (*IMG_SIZE, 3)
    
    # Training Params
    BATCH_SIZE = 32
    
    # Class Mapping
    CLASS_NAMES = ["NORMAL", "PNEUMONIA"]
    CLASS_INDICES = {"NORMAL": 0, "PNEUMONIA": 1}


def get_train_transforms(img_size):
    """Augmentation cho training data"""
    return A.Compose([
        A.Resize(img_size[0], img_size[1], interpolation=cv2.INTER_LINEAR),
        A.Rotate(limit=15, p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=0, p=0.5),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
    ], p=1.0)


def get_val_test_transforms(img_size):
    """Augmentation cho validation/test data"""
    return A.Compose([
        A.Resize(img_size[0], img_size[1], interpolation=cv2.INTER_LINEAR),
    ], p=1.0)


class PneumoniaDataGeneratorP2(Sequence):
    """
    Data Generator cho Pipeline 2 (EfficientNetB0 Baseline)
    Chỉ hỗ trợ mode='baseline' cho P2
    """
    
    def __init__(self, df, batch_size, target_size, 
                 preprocessing_function=None,
                 augmentation=None, 
                 shuffle=True):
        
        self.df = df.copy()
        self.batch_size = batch_size
        self.target_size = target_size
        self.preprocessing_function = preprocessing_function
        self.augmentation = augmentation
        self.shuffle = shuffle
        
        self.labels = self.df['label'].map(Config.CLASS_INDICES).values
        self.filepaths = self.df['filepath'].values
        
        self.indices = np.arange(len(self.filepaths))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.filepaths) / self.batch_size))

    def __getitem__(self, index):
        start_idx = index * self.batch_size
        end_idx = min((index + 1) * self.batch_size, len(self.filepaths))
        current_batch_size = end_idx - start_idx
        
        batch_indices = self.indices[start_idx:end_idx]

        # Khởi tạo batch cho baseline mode
        X_img = np.empty((current_batch_size, *self.target_size, 3), dtype=np.float32)
        y = np.empty((current_batch_size), dtype=int)

        for i, idx in enumerate(batch_indices):
            img_path = self.filepaths[idx]
            
            # 1. Tải ảnh
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # 2. Áp dụng augmentation
            if self.augmentation:
                augmented = self.augmentation(image=img)
                X_img[i,] = augmented['image']
            else:
                # Resize về target size nếu không có augmentation
                img = cv2.resize(img, self.target_size, interpolation=cv2.INTER_LINEAR)
                X_img[i,] = img

            y[i] = self.labels[idx]

        # 3. Áp dụng EfficientNet preprocessing
        if self.preprocessing_function:
            X_img = self.preprocessing_function(X_img)
        else:
            X_img = X_img / 255.0  # Mặc định

        return X_img, y

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
            
    def get_true_labels(self):
        return self.labels


def create_representative_dataset_generator(data_df, num_samples=100):
    """
    Tạo representative dataset cho TFLite quantization
    
    Args:
        data_df: DataFrame chứa đường dẫn ảnh
        num_samples: Số lượng samples cho quantization
    
    Returns:
        Generator function cho TFLite converter
    """
    # Tạo augmentation cho representative data (chỉ resize)
    transforms = get_val_test_transforms(Config.IMG_SIZE)
    
    # Tạo generator
    generator = PneumoniaDataGeneratorP2(
        df=data_df.sample(n=min(num_samples, len(data_df))),
        batch_size=1,
        target_size=Config.IMG_SIZE,
        preprocessing_function=effnet_preprocess.preprocess_input,
        augmentation=transforms,
        shuffle=False
    )
    
    def representative_data_gen():
        """Generator function cho TFLite converter"""
        for i in range(len(generator)):
            # Lấy một batch (batch_size=1)
            batch_x, _ = generator[i]
            # Yield dưới dạng list of numpy arrays
            yield [batch_x.astype(np.float32)]
    
    return representative_data_gen


def preprocess_single_image(image_path, target_size=(224, 224)):
    """
    Preprocess một ảnh đơn lẻ theo chuẩn EfficientNet
    
    Args:
        image_path: Đường dẫn đến ảnh
        target_size: Kích thước target (width, height)
    
    Returns:
        Preprocessed image tensor shape (1, 224, 224, 3)
    """
    # Đọc ảnh
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize
    img = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)
    
    # Expand dims để có batch dimension
    img = np.expand_dims(img, axis=0).astype(np.float32)
    
    # EfficientNet preprocessing
    img = effnet_preprocess.preprocess_input(img)
    
    return img


def preprocess_image_array(img_array, target_size=(224, 224)):
    """
    Preprocess image array theo chuẩn EfficientNet
    
    Args:
        img_array: numpy array shape (H, W, 3) với values [0, 255]
        target_size: Kích thước target (width, height)
    
    Returns:
        Preprocessed image tensor shape (1, 224, 224, 3)
    """
    # Resize nếu cần
    if img_array.shape[:2] != target_size:
        img_array = cv2.resize(img_array, target_size, interpolation=cv2.INTER_LINEAR)
    
    # Expand dims để có batch dimension
    img_array = np.expand_dims(img_array, axis=0).astype(np.float32)
    
    # EfficientNet preprocessing: [0, 255] -> [-1, 1]
    img_array = effnet_preprocess.preprocess_input(img_array)
    
    return img_array
