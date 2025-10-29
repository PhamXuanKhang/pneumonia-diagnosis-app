"""Training script"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

import tensorflow as tf
from src.data.data_loader import DataLoader
from src.data.preprocessing import ImagePreprocessor
from src.data.data_augmentation import DataAugmentor
from src.models.transfer_learning import TransferLearningModel
from src.training.trainer import Trainer
from src.training.callbacks import get_callbacks
from src.utils.config import load_config
from src.utils.logger import setup_logger


def main(args):
    """Main training function"""
    
    # Setup logger
    logger = setup_logger('training', 'logs/training.log')
    logger.info("Starting training...")
    
    # Load config
    config = load_config(args.config)
    
    # Data loading
    logger.info("Loading data...")
    data_loader = DataLoader(
        data_dir=config['data']['train_dir'],
        img_size=tuple(config['data']['img_size']),
        batch_size=config['training']['batch_size']
    )
    
    train_ds = data_loader.load_dataset('train', validation_split=0.2)
    val_ds = data_loader.load_dataset('validation', validation_split=0.2)
    
    # Preprocessing
    train_ds = train_ds.map(ImagePreprocessor.normalize)
    val_ds = val_ds.map(ImagePreprocessor.normalize)
    
    # Data augmentation
    if config['data']['augmentation']:
        augmentor = DataAugmentor()
        aug_layer = augmentor.get_augmentation_layer()
        train_ds = augmentor.augment_dataset(train_ds, aug_layer)
    
    # Optimize dataset
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)
    
    # Build model
    logger.info(f"Building model: {config['model']['name']}")
    model = TransferLearningModel(
        input_shape=tuple(config['data']['img_size']) + (3,),
        num_classes=config['model']['num_classes'],
        base_model_name=config['model']['base_model'],
        trainable_base=config['model']['trainable_base']
    )
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(config['training']['learning_rate']),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    model.summary()
    
    # Training
    logger.info("Starting training...")
    trainer = Trainer(model.model, config['paths']['model_dir'])
    
    callbacks = get_callbacks(
        checkpoint_dir=config['paths']['checkpoint_dir'],
        log_dir=config['paths']['log_dir'],
        early_stopping_patience=config['training']['early_stopping_patience']
    )
    
    history = trainer.train(
        train_dataset=train_ds,
        val_dataset=val_ds,
        epochs=config['training']['epochs'],
        callbacks=callbacks
    )
    
    # Save model
    logger.info("Saving model...")
    trainer.save_model(config['model']['name'])
    
    logger.info("Training completed!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train pneumonia classification model')
    parser.add_argument('--config', type=str, default='configs/training_config.yaml',
                       help='Path to config file')
    
    args = parser.parse_args()
    main(args)
