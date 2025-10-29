"""Visualization utilities"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path


class Visualizer:
    """Visualize training results and predictions"""
    
    @staticmethod
    def plot_training_history(history, save_path: str = None):
        """Plot training and validation metrics"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Accuracy
        axes[0].plot(history.history['accuracy'], label='Train Accuracy')
        if 'val_accuracy' in history.history:
            axes[0].plot(history.history['val_accuracy'], label='Val Accuracy')
        axes[0].set_title('Model Accuracy')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].legend()
        axes[0].grid(True)
        
        # Loss
        axes[1].plot(history.history['loss'], label='Train Loss')
        if 'val_loss' in history.history:
            axes[1].plot(history.history['val_loss'], label='Val Loss')
        axes[1].set_title('Model Loss')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training history plot saved to {save_path}")
        
        plt.show()
    
    @staticmethod
    def plot_confusion_matrix(cm: np.ndarray, class_names: list, 
                             save_path: str = None):
        """Plot confusion matrix"""
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to {save_path}")
        
        plt.show()
    
    @staticmethod
    def plot_sample_predictions(images, true_labels, pred_labels, 
                               class_names: list, num_samples: int = 9,
                               save_path: str = None):
        """Plot sample predictions"""
        num_samples = min(num_samples, len(images))
        rows = int(np.sqrt(num_samples))
        cols = int(np.ceil(num_samples / rows))
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, 15))
        axes = axes.flatten()
        
        for i in range(num_samples):
            axes[i].imshow(images[i])
            true_class = class_names[true_labels[i]]
            pred_class = class_names[pred_labels[i]]
            color = 'green' if true_labels[i] == pred_labels[i] else 'red'
            axes[i].set_title(f'True: {true_class}\nPred: {pred_class}', 
                            color=color)
            axes[i].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Sample predictions saved to {save_path}")
        
        plt.show()
