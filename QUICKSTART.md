# Quick Start Guide

Hướng dẫn nhanh để bắt đầu với dự án.

## Bước 1: Clone Repository

```bash
git clone <repository-url>
cd pneumonia-diagnosis-app
```

## Bước 2: Setup ML Environment

```bash
# Tạo virtual environment
python -m venv venv

# Activate
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
cd ml
pip install -r requirements.txt
pip install -e .
```

## Bước 3: Setup DVC (Optional)

```bash
# Initialize DVC
dvc init

# Add remote storage (Google Drive)
dvc remote add -d storage gdrive://<folder-id>

# Pull data và models
dvc pull
```

## Bước 4: Chuẩn bị dữ liệu

### Option A: Sử dụng dataset có sẵn

```bash
# Download dataset (ví dụ: Kaggle Pneumonia Dataset)
# https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

# Extract vào ml/data/raw/
# Cấu trúc:
# ml/data/raw/
# ├── NORMAL/
# └── PNEUMONIA/
```

### Option B: Tổ chức dữ liệu của bạn

```bash
python ml/scripts/data_pipeline.py \
  --source_dir ml/data/raw \
  --output_dir ml/data/processed
```

## Bước 5: Train Model

```bash
# Train với config mặc định
python ml/scripts/train.py --config ml/configs/training_config.yaml

# Model sẽ được lưu vào ml/models/saved_models/
```

## Bước 6: Evaluate Model

```bash
python ml/scripts/evaluate.py \
  --model_path ml/models/saved_models/pneumonia_mobilenetv2 \
  --config ml/configs/training_config.yaml

# Kết quả trong ml/outputs/
```

## Bước 7: Convert to TFLite

```bash
python ml/scripts/convert_to_tflite.py \
  --model_path ml/models/saved_models/pneumonia_mobilenetv2 \
  --output_path ml/models/tflite/pneumonia_mobilenetv2.tflite \
  --quantize

# TFLite model: ml/models/tflite/pneumonia_mobilenetv2.tflite
```

## Bước 8: Setup Flutter App

```bash
# Install Flutter (nếu chưa có)
# https://flutter.dev/docs/get-started/install

# Tạo Flutter project
flutter create mobile

# Copy templates
cp -r mobile_templates/lib/* mobile/lib/
cp mobile_templates/pubspec.yaml mobile/

# Install dependencies
cd mobile
flutter pub get
```

## Bước 9: Copy Model vào Flutter

```bash
# Tạo assets folder
mkdir -p mobile/assets/models

# Copy TFLite model
cp ml/models/tflite/pneumonia_mobilenetv2.tflite mobile/assets/models/

# Tạo labels file
echo "NORMAL" > mobile/assets/models/labels.txt
echo "PNEUMONIA" >> mobile/assets/models/labels.txt
```

## Bước 10: Run Flutter App

```bash
cd mobile

# Kiểm tra devices
flutter devices

# Run app
flutter run

# Hoặc build APK
flutter build apk --release
```

## Testing

### Test ML Code

```bash
cd ml
pytest tests/ -v
```

### Test Flutter App

```bash
cd mobile
flutter test
```

## Troubleshooting

### Lỗi Python Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt --force-reinstall
```

### Lỗi Flutter

```bash
flutter clean
flutter pub get
flutter doctor
```

### Lỗi TFLite Model

- Kiểm tra model path trong `pubspec.yaml`
- Verify model file tồn tại
- Check model format (phải là .tflite)

## Next Steps

1. Đọc [README.md](README.md) để hiểu rõ hơn về dự án
2. Xem [FLUTTER_SETUP.md](FLUTTER_SETUP.md) cho chi tiết Flutter
3. Đọc [docs/architecture.md](docs/architecture.md) về kiến trúc
4. Xem [docs/deployment_guide.md](docs/deployment_guide.md) để deploy

## Useful Commands

```bash
# ML
python ml/scripts/train.py --help
python ml/scripts/evaluate.py --help
python ml/scripts/convert_to_tflite.py --help

# DVC
dvc status
dvc repro
dvc metrics show
dvc push
dvc pull

# Flutter
flutter doctor
flutter devices
flutter run
flutter build apk
flutter test
flutter analyze

# Git
git status
git add .
git commit -m "message"
git push
```

## Support

Nếu gặp vấn đề:
1. Check [README.md](README.md)
2. Check [docs/](docs/)
3. Open GitHub issue
4. Contact: your.email@example.com
