# ML Module - Pneumonia Diagnosis

Module Machine Learning cho chẩn đoán viêm phổi từ ảnh X-quang.

## Cấu trúc

```
ml/
├── data/                    # Dữ liệu
│   ├── raw/                # Dữ liệu gốc
│   ├── processed/          # Dữ liệu đã xử lý
│   └── external/           # Dữ liệu external
├── models/                 # Models
│   ├── checkpoints/        # Training checkpoints
│   ├── saved_models/       # SavedModel format
│   └── tflite/            # TFLite models
├── notebooks/             # Jupyter notebooks
├── src/                   # Source code
│   ├── data/             # Data processing
│   ├── models/           # Model architectures
│   ├── training/         # Training logic
│   ├── evaluation/       # Evaluation
│   └── utils/            # Utilities
├── scripts/              # Executable scripts
├── tests/                # Tests
└── configs/              # Configurations
```

## Cài đặt

```bash
# Tạo virtual environment
python -m venv venv
source venv/bin/activate

# Cài đặt dependencies
pip install -r requirements.txt
pip install -e .
```

## Sử dụng

### 1. Chuẩn bị dữ liệu

```bash
python scripts/data_pipeline.py \
  --source_dir data/raw \
  --output_dir data/processed \
  --train_ratio 0.7 \
  --val_ratio 0.15 \
  --test_ratio 0.15
```

### 2. Training

```bash
python scripts/train.py --config configs/training_config.yaml
```

### 3. Evaluation

```bash
python scripts/evaluate.py \
  --model_path models/saved_models/pneumonia_mobilenetv2 \
  --config configs/training_config.yaml
```

### 4. Convert to TFLite

```bash
python scripts/convert_to_tflite.py \
  --model_path models/saved_models/pneumonia_mobilenetv2 \
  --output_path models/tflite/pneumonia_mobilenetv2.tflite \
  --quantize \
  --test
```

## Configuration

### Model Config (`configs/model_config.yaml`)

```yaml
model:
  name: "pneumonia_classifier"
  type: "transfer_learning"
  base_model: "mobilenetv2"
  num_classes: 2
```

### Training Config (`configs/training_config.yaml`)

```yaml
training:
  batch_size: 32
  epochs: 50
  learning_rate: 0.001
  early_stopping_patience: 10
```

## Models

### Available Base Models
- MobileNetV2 (recommended cho mobile)
- ResNet50
- EfficientNetB0
- InceptionV3

### Custom CNN
Có thể sử dụng custom CNN architecture trong `src/models/cnn_model.py`

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test
pytest tests/unit/test_model.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## Notebooks

- `01_data_exploration.ipynb`: Khám phá dữ liệu
- `02_model_experimentation.ipynb`: Thử nghiệm models
- `03_model_evaluation.ipynb`: Đánh giá chi tiết

## API Reference

### DataLoader

```python
from src.data.data_loader import DataLoader

loader = DataLoader(
    data_dir="data/processed/train",
    img_size=(224, 224),
    batch_size=32
)
dataset = loader.load_dataset()
```

### Model Building

```python
from src.models.transfer_learning import TransferLearningModel

model = TransferLearningModel(
    input_shape=(224, 224, 3),
    num_classes=2,
    base_model_name='mobilenetv2'
)
model.compile()
```

### Training

```python
from src.training.trainer import Trainer
from src.training.callbacks import get_callbacks

trainer = Trainer(model.model, "models")
callbacks = get_callbacks("models/checkpoints", "logs")

history = trainer.train(
    train_dataset=train_ds,
    val_dataset=val_ds,
    epochs=50,
    callbacks=callbacks
)
```

## Performance

### Metrics
- Accuracy
- Precision
- Recall
- F1 Score
- AUC-ROC (binary classification)
- Confusion Matrix

### Visualization
- Training history plots
- Confusion matrix heatmap
- Sample predictions

## Troubleshooting

### Out of Memory
```python
# Giảm batch size
batch_size = 16

# Sử dụng mixed precision
policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)
```

### Overfitting
- Tăng dropout rate
- Thêm data augmentation
- Giảm model complexity
- Early stopping

### Underfitting
- Tăng model complexity
- Train lâu hơn
- Giảm regularization
- Kiểm tra data quality
