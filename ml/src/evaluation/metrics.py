"""Model evaluation metrics"""

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    accuracy_score, precision_recall_fscore_support,
    roc_auc_score, roc_curve
)
import json


class ModelEvaluator:
    """Evaluate model performance"""
    
    def __init__(self, model: tf.keras.Model, class_names: list):
        self.model = model
        self.class_names = class_names
        
    def evaluate(self, test_dataset: tf.data.Dataset):
        """
        Evaluate model on test dataset
        
        Returns:
            Dictionary of metrics
        """
        # Get predictions
        y_true = []
        y_pred = []
        y_pred_proba = []
        
        for images, labels in test_dataset:
            predictions = self.model.predict(images, verbose=0)
            y_pred_proba.extend(predictions)
            y_pred.extend(np.argmax(predictions, axis=1))
            y_true.extend(labels.numpy())
        
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        y_pred_proba = np.array(y_pred_proba)
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted'
        )
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Classification report
        report = classification_report(
            y_true, y_pred, 
            target_names=self.class_names,
            output_dict=True
        )
        
        metrics = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'confusion_matrix': cm.tolist(),
            'classification_report': report
        }
        
        # ROC AUC for binary classification
        if len(self.class_names) == 2:
            auc = roc_auc_score(y_true, y_pred_proba[:, 1])
            metrics['auc'] = float(auc)
        
        return metrics
    
    def save_metrics(self, metrics: dict, output_path: str):
        """Save metrics to JSON file"""
        with open(output_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        print(f"Metrics saved to {output_path}")
