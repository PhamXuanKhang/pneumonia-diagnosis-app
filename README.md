# Pneumonia Diagnosis App

Ứng dụng chẩn đoán viêm phổi từ ảnh X-quang sử dụng Machine Learning và Flutter.

## Tổng quan

Dự án bao gồm:
- **ML Pipeline**: Training và evaluation model sử dụng TensorFlow
- **Mobile App**: Ứng dụng Flutter cho Android/iOS
- **MLOps**: DVC cho quản lý data/model, GitHub Actions cho CI/CD

## Cấu trúc dự án

```
pneumonia-diagnosis-app/
├── ml/                          # Machine Learning codebase
│   ├── data/                    # Dữ liệu (tracked by DVC)
│   ├── models/                  # Models (tracked by DVC)
│   ├── notebooks/               # Jupyter notebooks
│   ├── src/                     # Source code
│   ├── scripts/                 # Training/evaluation scripts
│   ├── tests/                   # Unit tests
│   └── configs/                 # Configuration files
├── mobile/                      # Flutter application
├── mobile_templates/            # Flutter code templates
├── .github/workflows/           # CI/CD workflows
├── docs/                        # Documentation
└── dvc.yaml                     # DVC pipeline
```

## Yêu cầu hệ thống

### Machine Learning
- Python 3.8+
- TensorFlow 2.15+
- CUDA (optional, cho GPU training)

### Mobile App
- Flutter 3.16+
- Android SDK (cho Android)
- Xcode (cho iOS)

## Cài đặt

### 1. Setup ML Environment

```bash
# Clone repository
git clone <repository-url>
cd pneumonia-diagnosis-app

# Tạo virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Cài đặt dependencies
cd ml
pip install -r requirements.txt
pip install -e .
```

### 2. Setup DVC

```bash
# Khởi tạo DVC
dvc init

# Cấu hình remote storage (Google Drive)
dvc remote add -d storage gdrive://<folder-id>

# Pull data
dvc pull
```

### 3. Setup Flutter

Xem hướng dẫn chi tiết trong [FLUTTER_SETUP.md](FLUTTER_SETUP.md)

```bash
# Tạo Flutter project
flutter create mobile

# Copy templates
cp -r mobile_templates/lib/* mobile/lib/
cp mobile_templates/pubspec.yaml mobile/

# Cài đặt dependencies
cd mobile
flutter pub get
```

## Sử dụng

### Training Model

```bash
# Chuẩn bị dữ liệu
python ml/scripts/data_pipeline.py \
  --source_dir ml/data/raw \
  --output_dir ml/data/processed

# Train model
python ml/scripts/train.py \
  --config ml/configs/training_config.yaml

# Evaluate model
python ml/scripts/evaluate.py \
  --model_path ml/models/saved_models/pneumonia_mobilenetv2 \
  --config ml/configs/training_config.yaml

# Convert to TFLite
python ml/scripts/convert_to_tflite.py \
  --model_path ml/models/saved_models/pneumonia_mobilenetv2 \
  --output_path ml/models/tflite/pneumonia_mobilenetv2.tflite \
  --quantize
```

### DVC Pipeline

```bash
# Chạy toàn bộ pipeline
dvc repro

# Chạy stage cụ thể
dvc repro train

# Xem metrics
dvc metrics show

# So sánh experiments
dvc metrics diff
```

### Chạy Mobile App

```bash
cd mobile

# Chạy trên emulator/device
flutter run

# Build APK
flutter build apk --release

# Build iOS
flutter build ios --release
```

## Testing

### ML Tests

```bash
cd ml

# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test
pytest tests/unit/test_model.py
```

### Flutter Tests

```bash
cd mobile

# Run unit tests
flutter test

# Run with coverage
flutter test --coverage
```

## CI/CD

GitHub Actions workflows:
- **ML Pipeline**: Chạy tests và linting cho ML code
- **Model Training**: Train model tự động (manual trigger)
- **Flutter Build**: Build và test Flutter app

## Cấu trúc ML Code

### Data Module
- `data_loader.py`: Load datasets
- `preprocessing.py`: Image preprocessing
- `data_augmentation.py`: Data augmentation

### Models Module
- `base_model.py`: Base model class
- `cnn_model.py`: Custom CNN architecture
- `transfer_learning.py`: Transfer learning models

### Training Module
- `trainer.py`: Training logic
- `callbacks.py`: Training callbacks

### Evaluation Module
- `metrics.py`: Evaluation metrics
- `visualizer.py`: Visualization utilities

## Cấu trúc Flutter Code

### Core
- `constants/`: App constants
- `config/`: Configuration
- `utils/`: Utility functions

### Data Layer
- `models/`: Data models
- `repositories/`: Repository implementations
- `datasources/`: Data sources

### Domain Layer
- `entities/`: Business entities
- `repositories/`: Repository interfaces
- `usecases/`: Business logic

### Presentation Layer
- `screens/`: UI screens
- `widgets/`: Reusable widgets
- `providers/`: State management

### ML Integration
- `tflite_model.dart`: TFLite model wrapper
- `image_preprocessing.dart`: Image preprocessing

## Model Information

- **Architecture**: MobileNetV2 (Transfer Learning)
- **Input Size**: 224x224x3
- **Classes**: NORMAL, PNEUMONIA
- **Framework**: TensorFlow/TFLite

## Dataset

Dataset cần có cấu trúc:
```
data/raw/
├── NORMAL/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── PNEUMONIA/
    ├── image1.jpg
    ├── image2.jpg
    └── ...
```

## Contributing

1. Fork repository
2. Tạo feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Tạo Pull Request

## License

MIT License - xem file [LICENSE](LICENSE)

## Tác giả

Your Name - your.email@example.com

## Tài liệu tham khảo

- [TensorFlow Documentation](https://www.tensorflow.org/api_docs)
- [Flutter Documentation](https://flutter.dev/docs)
- [DVC Documentation](https://dvc.org/doc)
- [TFLite Flutter Plugin](https://pub.dev/packages/tflite_flutter)
