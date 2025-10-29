"""Evaluation script"""

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / 'src'))

import tensorflow as tf
from src.data.data_loader import DataLoader
from src.data.preprocessing import ImagePreprocessor
from src.evaluation.metrics import ModelEvaluator
from src.evaluation.visualizer import Visualizer
from src.utils.config import load_config
from src.utils.logger import setup_logger


def main(args):
    """Main evaluation function"""
    
    # Setup logger
    logger = setup_logger('evaluation', 'logs/evaluation.log')
    logger.info("Starting evaluation...")
    
    # Load config
    config = load_config(args.config)
    
    # Load model
    logger.info(f"Loading model from {args.model_path}")
    model = tf.keras.models.load_model(args.model_path)
    
    # Load test data
    logger.info("Loading test data...")
    data_loader = DataLoader(
        data_dir=config['data']['test_dir'],
        img_size=tuple(config['data']['img_size']),
        batch_size=config['training']['batch_size']
    )
    
    test_ds = data_loader.load_dataset()
    test_ds = test_ds.map(ImagePreprocessor.normalize)
    
    class_names = data_loader.get_class_names()
    
    # Evaluate
    logger.info("Evaluating model...")
    evaluator = ModelEvaluator(model, class_names)
    metrics = evaluator.evaluate(test_ds)
    
    # Print metrics
    logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"Precision: {metrics['precision']:.4f}")
    logger.info(f"Recall: {metrics['recall']:.4f}")
    logger.info(f"F1 Score: {metrics['f1_score']:.4f}")
    
    # Save metrics
    output_dir = Path(config['paths']['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    evaluator.save_metrics(metrics, output_dir / 'metrics.json')
    
    # Visualize confusion matrix
    logger.info("Generating visualizations...")
    visualizer = Visualizer()
    visualizer.plot_confusion_matrix(
        metrics['confusion_matrix'],
        class_names,
        save_path=output_dir / 'confusion_matrix.png'
    )
    
    logger.info("Evaluation completed!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate pneumonia classification model')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to saved model')
    parser.add_argument('--config', type=str, default='configs/training_config.yaml',
                       help='Path to config file')
    
    args = parser.parse_args()
    main(args)
