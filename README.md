# Pneumonia Diagnosis App

Ứng dụng chẩn đoán viêm phổi từ ảnh X-quang ngực sử dụng Deep Learning và Flutter, được phát triển cho môn DAT301m.

[![Flutter](https://img.shields.io/badge/Flutter-3.0+-blue.svg)](https://flutter.dev/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15+-orange.svg)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📋 Tổng quan

Dự án này xây dựng một hệ thống hoàn chỉnh để chẩn đoán viêm phổi từ ảnh X-quang ngực, bao gồm:

- **ML Pipeline**: Training model sử dụng EfficientNetB0 với TensorFlow
- **Mobile App**: Ứng dụng Flutter đa nền tảng (Android/iOS) với TFLite inference
- **MLOps**: Quản lý data/model với DVC, CI/CD với GitHub Actions

## 🎯 Tính năng chính

### Machine Learning
- ✅ **EfficientNetB0 Architecture**: Transfer learning từ ImageNet weights
- ✅ **Binary Classification**: Phân loại NORMAL vs PNEUMONIA
- ✅ **TFLite Conversion**: Chuyển đổi model sang TensorFlow Lite với quantization
- ✅ **Multi-output Support**: Model hỗ trợ visualization với feature maps
- ✅ **Comprehensive Evaluation**: Metrics đầy đủ (Accuracy, Precision, Recall, F1, AUC-ROC)

### Mobile Application
- ✅ **Cross-platform**: Hỗ trợ Android và iOS
- ✅ **Camera Integration**: Chụp ảnh X-quang trực tiếp từ camera
- ✅ **Gallery Support**: Chọn ảnh từ thư viện
- ✅ **Real-time Inference**: Chẩn đoán nhanh với TFLite
- ✅ **Visualization**: Hiển thị kết quả với confidence score và feature maps
- ✅ **Material Design 3**: Giao diện hiện đại với dark mode support

### MLOps & DevOps
- ✅ **DVC Pipeline**: Version control cho data và models
- ✅ **GitHub Actions**: Automated testing và CI/CD
- ✅ **Jupyter Notebooks**: Exploratory data analysis và experiments
- ✅ **Comprehensive Testing**: Unit tests và integration tests

## 🏗️ Kiến trúc hệ thống

```
pneumonia-diagnosis-app/
├── ml/                          # Machine Learning Pipeline
│   ├── data/                    # Datasets (DVC tracked)
│   ├── models/                  # Trained models
│   │   ├── saved_models/        # Keras models (.keras)
│   │   └── tflite/             # TFLite models (.tflite)
│   ├── src/                     # ML source code
│   │   ├── data/               # Data processing & augmentation
│   │   ├── models/             # Model architectures (EfficientNetB0)
│   │   ├── training/           # Training logic & callbacks
│   │   ├── evaluation/         # Metrics & visualization
│   │   └── utils/              # Utilities
│   ├── scripts/                # Executable scripts
│   │   ├── create_dummy_p2_model.py    # Dummy model creation
│   │   ├── convert_p2_to_tflite.py     # TFLite conversion
│   │   ├── validate_tflite_p2.py       # TFLite validation
│   │   ├── evaluate_p2.py              # Model evaluation
│   │   └── data_pipeline.py            # Data preprocessing
│   ├── notebooks/              # Jupyter notebooks
│   ├── tests/                  # Unit & integration tests
│   └── configs/                # YAML configurations
│
├── mobile2/                     # Flutter Application
│   ├── lib/
│   │   ├── main.dart           # App entry point
│   │   ├── models/             # Data models
│   │   ├── services/           # Business logic
│   │   │   ├── inference_service_tflite.dart  # TFLite inference
│   │   │   ├── image_service.dart             # Image handling
│   │   │   ├── preprocessing_service.dart     # Image preprocessing
│   │   │   └── visualization_service.dart     # Result visualization
│   │   └── screens/            # UI screens
│   ├── assets/
│   │   └── models/             # TFLite models
│   └── pubspec.yaml            # Flutter dependencies
│
├── .github/workflows/           # CI/CD Pipelines
│   ├── ml-pipeline.yml         # ML testing
│   ├── model-training.yml      # Automated training
│   └── flutter-build.yml       # Flutter build & test
│
├── docs/                        # Documentation
├── dvc.yaml                     # DVC pipeline definition
└── params.yaml                  # DVC parameters
```

## 🚀 Quick Start

### Yêu cầu hệ thống

**Machine Learning:**
- Python 3.8+
- TensorFlow 2.15+
- CUDA (optional, cho GPU training)

**Mobile App:**
- Flutter 3.0+
- Android SDK / Xcode
- Android Studio / VS Code

### 1. Setup ML Environment

```bash
# Clone repository
git clone <repository-url>
cd pneumonia-diagnosis-app

# Tạo virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # macOS/Linux

# Install dependencies
cd ml
pip install -r requirements.txt
```

### 2. Tạo hoặc Download Model

**Option A: Sử dụng dummy model (cho testing)**
```bash
# Tạo dummy model
python ml/scripts/create_dummy_p2_model.py --test

# Convert sang TFLite
python ml/scripts/convert_p2_to_tflite.py --test
```

**Option B: Train model từ dataset**
```bash
# Download dataset (Kaggle Chest X-Ray Pneumonia)
# https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

# Chuẩn bị dữ liệu
python ml/scripts/data_pipeline.py \
  --source_dir ml/data/raw \
  --output_dir ml/data/processed

# Train model
python ml/scripts/train.py --config ml/configs/training_config.yaml

# Evaluate
python ml/scripts/evaluate_p2.py \
  --model_path ml/models/saved_models/P2_EffNetB0_Baseline_final.keras \
  --data_root /path/to/chest_xray

# Convert to TFLite
python ml/scripts/convert_p2_to_tflite.py \
  --model_path ml/models/saved_models/P2_EffNetB0_Baseline_final.keras \
  --output_path ml/models/tflite/pneumonia_efficientnet_p2.tflite \
  --quantize
```

### 3. Setup Flutter App

```bash
# Navigate to mobile directory
cd mobile2

# Install dependencies
flutter pub get

# Copy TFLite model to assets (nếu chưa có)
copy ..\ml\models\tflite\pneumonia_efficientnet_p2.tflite assets\models\pneumonia_classifier.tflite
```

### 4. Run App

```bash
# Check connected devices
flutter devices

# Run app
flutter run

# Build APK (Android)
flutter build apk --release

# Build iOS
flutter build ios --release
```

## 📊 Model Information

### Pipeline 2 (EfficientNetB0 Baseline)

- **Architecture**: EfficientNetB0 (pretrained on ImageNet)
- **Feature Extraction**: `top_conv` layer (7×7×1280)
- **Classification Head**: 
  - GlobalAveragePooling2D
  - Dense(128, ReLU)
  - Dropout(0.5)
  - Dense(1, Sigmoid)
- **Input Shape**: 224×224×3
- **Preprocessing**: EfficientNet normalization ([-1, 1])
- **Output**: Binary classification (NORMAL=0, PNEUMONIA=1)
- **Model Size**: ~16.7 MB (TFLite)

### Performance Metrics

Model được đánh giá trên các metrics:
- Accuracy
- Precision
- Recall
- F1 Score
- AUC-ROC
- Confusion Matrix

## 🛠️ Development

### ML Development

```bash
# Run tests
cd ml
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Analyze code
pylint src/

# Format code
black src/
```

### Flutter Development

```bash
cd mobile2

# Run tests
flutter test

# Analyze code
flutter analyze

# Format code
flutter format lib/

# Clean build
flutter clean
flutter pub get
```

### Jupyter Notebooks

```bash
cd ml
jupyter notebook

# Notebooks có sẵn:
# - notebooks/data.ipynb: Data exploration
# - notebooks/data_analysis.ipynb: Comprehensive analysis
# - notebooks/model_experiments/: Model experiments
```

## 📱 Mobile App Features

### Services

1. **InferenceService**: TFLite model inference
   - Model initialization
   - Prediction với Float32 input
   - Performance benchmarking
   - Model statistics

2. **ImageService**: Image handling
   - Camera capture
   - Gallery selection
   - Image validation

3. **PreprocessingService**: Image preprocessing
   - Resize to 224×224
   - EfficientNet normalization
   - Color space conversion

4. **VisualizationService**: Result visualization
   - Confidence score display
   - Feature map visualization
   - Diagnosis interpretation

### UI Components

- **DiagnosisScreen**: Main screen với camera/gallery integration
- **Material Design 3**: Modern UI với adaptive theming
- **Dark Mode**: Full dark mode support
- **Provider**: State management

## 🔄 CI/CD Pipeline

### GitHub Actions Workflows

1. **ML Pipeline** (`.github/workflows/ml-pipeline.yml`)
   - Python linting và testing
   - Code quality checks
   - Automated on push/PR

2. **Model Training** (`.github/workflows/model-training.yml`)
   - Automated model training
   - Manual trigger
   - Model evaluation và artifacts

3. **Flutter Build** (`.github/workflows/flutter-build.yml`)
   - Flutter testing
   - APK/IPA build
   - Code analysis

## 📖 Documentation

- [BUILD_COMMANDS.md](BUILD_COMMANDS.md) - Toàn bộ commands để build và deploy
- [QUICKSTART.md](QUICKSTART.md) - Hướng dẫn bắt đầu nhanh
- [FLUTTER_SETUP.md](FLUTTER_SETUP.md) - Setup Flutter chi tiết
- [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) - Cấu trúc dự án chi tiết
- [NOTES.md](NOTES.md) - Ghi chú quan trọng
- [ml/scripts/README_P2.md](ml/scripts/README_P2.md) - Pipeline 2 scripts documentation

## 🧪 Testing

### ML Tests

```bash
cd ml

# Unit tests
pytest tests/unit/ -v

# Integration tests
pytest tests/integration/ -v

# Specific test
pytest tests/unit/test_model.py -v
```

### Flutter Tests

```bash
cd mobile2

# Unit tests
flutter test

# Integration tests
flutter test integration_test/

# With coverage
flutter test --coverage
```

## 📦 Dependencies

### Python (ML)
- tensorflow >= 2.15.0
- numpy >= 1.24.0
- opencv-python >= 4.8.0
- scikit-learn >= 1.3.0
- matplotlib >= 3.7.0
- jupyter >= 1.0.0

### Flutter (Mobile)
- tflite_flutter: ^0.10.4
- image_picker: ^1.0.4
- camera: ^0.10.5
- image: ^4.1.3
- provider: ^6.1.1
- permission_handler: ^11.1.0

## 🤝 Contributing

1. Fork repository
2. Tạo feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Tạo Pull Request

## 📄 License

MIT License - xem file [LICENSE](LICENSE) để biết thêm chi tiết.

## 👥 Authors

Phạm Xuân Khang - DAT301m Project

## 🙏 Acknowledgments

- Dataset: [Kaggle Chest X-Ray Pneumonia Dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- EfficientNet: [Google Research](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet)
- TFLite Flutter: [tflite_flutter package](https://pub.dev/packages/tflite_flutter)

## 📞 Support

Nếu gặp vấn đề:
1. Xem [Documentation](#-documentation)
2. Check [GitHub Issues](../../issues)
3. Đọc [Troubleshooting](#-troubleshooting) section

## 🐛 Troubleshooting

### Common Issues

**1. TFLite model không load được**
```bash
# Verify model exists
dir mobile2\assets\models\

# Check pubspec.yaml có khai báo assets
# Rebuild app
flutter clean
flutter pub get
flutter run
```

**2. Python dependencies conflict**
```bash
pip install --upgrade pip
pip install -r requirements.txt --force-reinstall
```

**3. Flutter build fails**
```bash
flutter doctor
flutter clean
flutter pub get
flutter build apk --release
```

**4. Camera permission denied**
- Check AndroidManifest.xml có khai báo permissions
- Grant permissions manually trong Settings

## 🔮 Future Enhancements

- [ ] Multi-class classification (Normal, Bacterial, Viral)
- [ ] Grad-CAM visualization
- [ ] Model quantization optimization
- [ ] Cloud deployment (Firebase ML)
- [ ] Batch processing
- [ ] Export diagnosis reports (PDF)
- [ ] Multi-language support
- [ ] Offline model updates

---

**Made with ❤️ for DAT301m**
