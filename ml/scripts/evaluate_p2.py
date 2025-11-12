"""
Evaluation script cho Pipeline 2 (EfficientNetB0 Baseline)
Refactored từ dat301m-training-pipeline-comparation.py Cell 8
"""

import argparse
import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix, roc_curve
)

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.efficientnet_p2 import load_trained_p2_model, ConfigP2
from data.preprocessing_p2 import (
    PneumoniaDataGeneratorP2, 
    get_val_test_transforms,
    Config as DataConfig
)


def create_test_dataframe(data_root):
    """
    Tạo test DataFrame từ test data
    
    Args:
        data_root: Đường dẫn đến thư mục data
    
    Returns:
        DataFrame với columns ['filepath', 'label']
    """
    import glob
    
    test_data_path = os.path.join(data_root, "test")
    
    if not os.path.exists(test_data_path):
        raise FileNotFoundError(f"Không tìm thấy test data tại: {test_data_path}")
    
    # Load test data
    test_normal_files = glob.glob(os.path.join(test_data_path, 'NORMAL', '*.jpeg'))
    test_pneumonia_files = glob.glob(os.path.join(test_data_path, 'PNEUMONIA', '*.jpeg'))
    
    test_filepaths = test_normal_files + test_pneumonia_files
    test_labels = ['NORMAL'] * len(test_normal_files) + ['PNEUMONIA'] * len(test_pneumonia_files)
    
    test_df = pd.DataFrame({'filepath': test_filepaths, 'label': test_labels})
    
    print(f"Test dataset loaded:")
    print(f"  - Total samples: {len(test_df)}")
    print(f"  - NORMAL: {len(test_normal_files)}")
    print(f"  - PNEUMONIA: {len(test_pneumonia_files)}")
    
    return test_df


def evaluate_p2_model(model_path, test_df, output_dir):
    """
    Evaluate Pipeline 2 model trên test dataset
    
    Args:
        model_path: Đường dẫn đến P2 model
        test_df: Test DataFrame
        output_dir: Thư mục output
    
    Returns:
        metrics: Dictionary chứa evaluation metrics
        eval_data: Dictionary chứa prediction data
    """
    print("=" * 80)
    print(f"EVALUATING PIPELINE 2: {ConfigP2.MODEL_NAME}")
    print("=" * 80)
    print(f"Model path: {model_path}")
    print(f"Test samples: {len(test_df)}")
    print("=" * 80)
    
    # 1. Load model
    print("\n1. Loading model...")
    try:
        model = load_trained_p2_model(model_path)
        print(f"✅ Model loaded: {model.name}")
    except Exception as e:
        print(f"❌ Lỗi loading model: {e}")
        return None, None
    
    # 2. Prepare test data
    print("\n2. Preparing test data...")
    transforms = get_val_test_transforms(DataConfig.IMG_SIZE)
    
    test_generator = PneumoniaDataGeneratorP2(
        df=test_df,
        batch_size=32,
        target_size=DataConfig.IMG_SIZE,
        preprocessing_function=tf.keras.applications.efficientnet.preprocess_input,
        augmentation=transforms,
        shuffle=False
    )
    
    print(f"✅ Test generator created: {len(test_generator)} batches")
    
    # 3. Get true labels
    y_true = test_generator.get_true_labels()
    print(f"True labels shape: {y_true.shape}")
    
    # 4. Predict
    print("\n3. Running predictions...")
    y_prob = model.predict(test_generator, verbose=1)
    
    # Handle predictions
    y_prob = y_prob[:len(y_true)]  # Ensure same length
    y_pred = (y_prob > 0.5).astype(int).flatten()
    y_prob_flat = y_prob.flatten()
    
    print(f"Predictions shape: {y_prob.shape}")
    print(f"Sample predictions: {y_prob_flat[:10]}")
    
    # 5. Calculate metrics
    print("\n4. Calculating metrics...")
    metrics = {
        'model_name': ConfigP2.MODEL_NAME,
        'test_samples': len(y_true),
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, zero_division=0)),
        'f1_score': float(f1_score(y_true, y_pred, zero_division=0)),
        'auc_roc': float(roc_auc_score(y_true, y_prob_flat))
    }
    
    # 6. Classification report
    print("\n5. Classification Report:")
    print("-" * 40)
    report = classification_report(
        y_true, y_pred, 
        target_names=DataConfig.CLASS_NAMES, 
        zero_division=0
    )
    print(report)
    
    # 7. Confusion Matrix
    print("\n6. Confusion Matrix:")
    print("-" * 40)
    cm = confusion_matrix(y_true, y_pred)
    print(f"                Predicted")
    print(f"              Normal  Pneumonia")
    print(f"Actual Normal    {cm[0,0]:4d}      {cm[0,1]:4d}")
    print(f"    Pneumonia    {cm[1,0]:4d}      {cm[1,1]:4d}")
    
    # 8. Save confusion matrix plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=DataConfig.CLASS_NAMES, 
                yticklabels=DataConfig.CLASS_NAMES)
    plt.title(f'Confusion Matrix - {ConfigP2.MODEL_NAME}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    
    cm_path = output_dir / 'confusion_matrix.png'
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Confusion matrix saved: {cm_path}")
    
    # 9. ROC Curve
    print("\n7. ROC Curve:")
    print("-" * 40)
    fpr, tpr, _ = roc_curve(y_true, y_prob_flat)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'{ConfigP2.MODEL_NAME} (AUC = {metrics["auc_roc"]:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Chance (AUC = 0.50)')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(True)
    
    roc_path = output_dir / 'roc_curve.png'
    plt.savefig(roc_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ ROC curve saved: {roc_path}")
    
    # 10. Evaluation data for further analysis
    eval_data = {
        'y_true': y_true.tolist(),
        'y_pred': y_pred.tolist(),
        'y_prob': y_prob_flat.tolist(),
        'confusion_matrix': cm.tolist(),
        'roc_curve': {
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist()
        }
    }
    
    # 11. Print summary
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    print(f"Model: {metrics['model_name']}")
    print(f"Test Samples: {metrics['test_samples']}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1-Score: {metrics['f1_score']:.4f}")
    print(f"AUC-ROC: {metrics['auc_roc']:.4f}")
    print("=" * 80)
    
    return metrics, eval_data


def main():
    parser = argparse.ArgumentParser(description='Evaluate Pipeline 2 model')
    parser.add_argument(
        '--model_path', 
        type=str, 
        default='ml/models/saved_models/P2_EffNetB0_Baseline_final.keras',
        help='Path to P2 model (.keras)'
    )
    parser.add_argument(
        '--data_root', 
        type=str, 
        required=True,
        help='Path to data root directory'
    )
    parser.add_argument(
        '--output_dir', 
        type=str, 
        default='ml/outputs/evaluation',
        help='Output directory for results'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create test dataframe
    try:
        test_df = create_test_dataframe(args.data_root)
    except FileNotFoundError as e:
        print(f"❌ {e}")
        return
    
    # Evaluate model
    metrics, eval_data = evaluate_p2_model(
        model_path=args.model_path,
        test_df=test_df,
        output_dir=output_dir
    )
    
    if metrics is None:
        print("❌ Evaluation failed")
        return
    
    # Save results
    metrics_path = output_dir / 'metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"✅ Metrics saved: {metrics_path}")
    
    eval_data_path = output_dir / 'evaluation_data.json'
    with open(eval_data_path, 'w') as f:
        json.dump(eval_data, f, indent=2)
    print(f"✅ Evaluation data saved: {eval_data_path}")
    
    # Create summary report
    summary_path = output_dir / 'evaluation_summary.txt'
    with open(summary_path, 'w') as f:
        f.write("PIPELINE 2 EVALUATION SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Model: {metrics['model_name']}\n")
        f.write(f"Test Samples: {metrics['test_samples']}\n\n")
        f.write("METRICS:\n")
        f.write(f"  Accuracy:  {metrics['accuracy']:.4f}\n")
        f.write(f"  Precision: {metrics['precision']:.4f}\n")
        f.write(f"  Recall:    {metrics['recall']:.4f}\n")
        f.write(f"  F1-Score:  {metrics['f1_score']:.4f}\n")
        f.write(f"  AUC-ROC:   {metrics['auc_roc']:.4f}\n\n")
        f.write("FILES GENERATED:\n")
        f.write(f"  - Metrics: {metrics_path.name}\n")
        f.write(f"  - Evaluation Data: {eval_data_path.name}\n")
        f.write(f"  - Confusion Matrix: confusion_matrix.png\n")
        f.write(f"  - ROC Curve: roc_curve.png\n")
    
    print(f"✅ Summary report saved: {summary_path}")
    
    print(f"\n🎉 Evaluation completed successfully!")
    print(f"📁 Results saved in: {output_dir}")


if __name__ == '__main__':
    # Import tensorflow here to avoid loading issues
    import tensorflow as tf
    main()
