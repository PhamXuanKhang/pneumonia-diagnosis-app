"""Data preprocessing pipeline"""

import argparse
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split
import random


def organize_data(source_dir: str, output_dir: str, 
                 train_ratio: float = 0.7,
                 val_ratio: float = 0.15,
                 test_ratio: float = 0.15):
    """
    Organize data into train/val/test splits
    
    Args:
        source_dir: Source directory with class subdirectories
        output_dir: Output directory for organized data
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
    """
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    
    if not source_path.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")
    
    # Get class directories
    class_dirs = [d for d in source_path.iterdir() if d.is_dir()]
    
    print(f"Found {len(class_dirs)} classes")
    
    for class_dir in class_dirs:
        class_name = class_dir.name
        print(f"\nProcessing class: {class_name}")
        
        # Get all images
        images = list(class_dir.glob('*.jpg')) + \
                list(class_dir.glob('*.jpeg')) + \
                list(class_dir.glob('*.png'))
        
        print(f"Found {len(images)} images")
        
        # Shuffle
        random.shuffle(images)
        
        # Split data
        train_size = int(len(images) * train_ratio)
        val_size = int(len(images) * val_ratio)
        
        train_images = images[:train_size]
        val_images = images[train_size:train_size + val_size]
        test_images = images[train_size + val_size:]
        
        print(f"Train: {len(train_images)}, Val: {len(val_images)}, Test: {len(test_images)}")
        
        # Copy files
        for split_name, split_images in [
            ('train', train_images),
            ('val', val_images),
            ('test', test_images)
        ]:
            split_dir = output_path / split_name / class_name
            split_dir.mkdir(parents=True, exist_ok=True)
            
            for img in split_images:
                shutil.copy2(img, split_dir / img.name)
    
    print(f"\nData organized in {output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Organize data into train/val/test splits')
    parser.add_argument('--source_dir', type=str, required=True,
                       help='Source directory with class subdirectories')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for organized data')
    parser.add_argument('--train_ratio', type=float, default=0.7,
                       help='Training set ratio')
    parser.add_argument('--val_ratio', type=float, default=0.15,
                       help='Validation set ratio')
    parser.add_argument('--test_ratio', type=float, default=0.15,
                       help='Test set ratio')
    
    args = parser.parse_args()
    
    organize_data(
        args.source_dir,
        args.output_dir,
        args.train_ratio,
        args.val_ratio,
        args.test_ratio
    )
