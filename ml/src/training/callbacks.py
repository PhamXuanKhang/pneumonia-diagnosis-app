"""Training callbacks"""

import tensorflow as tf
from pathlib import Path


def get_callbacks(checkpoint_dir: str, log_dir: str, 
                  early_stopping_patience: int = 10):
    """
    Create training callbacks
    
    Args:
        checkpoint_dir: Directory to save checkpoints
        log_dir: Directory for TensorBoard logs
        early_stopping_patience: Patience for early stopping
        
    Returns:
        List of callbacks
    """
    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    callbacks = [
        # Model checkpoint
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path / 'model_epoch_{epoch:02d}_val_loss_{val_loss:.4f}.h5'),
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False,
            mode='min',
            verbose=1
        ),
        
        # Early stopping
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=early_stopping_patience,
            restore_best_weights=True,
            verbose=1
        ),
        
        # Reduce learning rate on plateau
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        
        # TensorBoard
        tf.keras.callbacks.TensorBoard(
            log_dir=str(log_path),
            histogram_freq=1,
            write_graph=True,
            write_images=True
        ),
        
        # CSV Logger
        tf.keras.callbacks.CSVLogger(
            str(log_path / 'training_log.csv'),
            separator=',',
            append=False
        )
    ]
    
    return callbacks
