# Ghi chú quan trọng

## Về Lint Errors hiện tại

Các lint errors bạn thấy là **BÌNH THƯỜNG** và sẽ tự động biến mất sau khi:

### 1. Tạo Flutter Project

```bash
flutter create mobile
```

### 2. Copy Templates và Install Dependencies

```bash
# Copy templates
cp -r mobile_templates/lib/* mobile/lib/
cp mobile_templates/pubspec.yaml mobile/

# Install dependencies
cd mobile
flutter pub get
```

Sau khi chạy `flutter pub get`, các packages như `image`, `tflite_flutter` sẽ được tải về và lint errors sẽ biến mất.

## Cấu trúc hiện tại

```
pneumonia-diagnosis-app/
├── mobile/                    # Placeholder (chạy flutter create mobile)
├── mobile_templates/          # Templates để copy vào mobile/
└── ml/                        # ML code (hoàn chỉnh)
```

## Quy trình làm việc

### Bước 1: Setup ML
```bash
cd ml
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### Bước 2: Tạo Flutter Project
```bash
cd ..
flutter create mobile
```

### Bước 3: Copy Templates
```bash
# Windows PowerShell
Copy-Item -Recurse mobile_templates\lib\* mobile\lib\
Copy-Item mobile_templates\pubspec.yaml mobile\

# Hoặc thủ công copy paste
```

### Bước 4: Install Flutter Dependencies
```bash
cd mobile
flutter pub get
```

### Bước 5: Verify
```bash
flutter analyze
flutter test
```

## Tại sao tách mobile_templates?

1. **Không thể tạo Flutter project tự động** - Cần chạy `flutter create`
2. **Templates sẵn sàng** - Copy vào sau khi tạo project
3. **Tránh conflict** - Không ghi đè files Flutter tự generate

## Files quan trọng cần copy

Từ `mobile_templates/` vào `mobile/`:

### Source Code
- `lib/ml/tflite_model.dart` - TFLite integration
- `lib/ml/image_preprocessing.dart` - Image processing
- `lib/core/constants/app_constants.dart` - Constants
- `lib/data/models/diagnosis_result.dart` - Data models

### Configuration
- `pubspec.yaml` - Dependencies và assets

### Assets (sau khi train model)
- `assets/models/pneumonia_mobilenetv2.tflite`
- `assets/models/labels.txt`

## Checklist Setup

### ML Setup
- [ ] Tạo virtual environment
- [ ] Install Python dependencies
- [ ] Download dataset
- [ ] Chạy data pipeline
- [ ] Train model
- [ ] Convert to TFLite

### Flutter Setup
- [ ] Cài đặt Flutter SDK
- [ ] Chạy `flutter doctor`
- [ ] Chạy `flutter create mobile`
- [ ] Copy templates từ mobile_templates/
- [ ] Update pubspec.yaml
- [ ] Chạy `flutter pub get`
- [ ] Copy TFLite model vào assets/
- [ ] Test trên emulator/device

### Git Setup
- [ ] Initialize git (`git init`)
- [ ] Add remote
- [ ] Commit initial structure
- [ ] Setup DVC (optional)

## Lưu ý quan trọng

1. **Không commit model files lớn vào Git** - Dùng DVC
2. **Không commit API keys** - Dùng .env
3. **Test trước khi commit** - Chạy tests
4. **Follow naming conventions** - Xem code style

## Troubleshooting

### Flutter create fails
```bash
flutter doctor
flutter upgrade
```

### Pub get fails
```bash
flutter clean
flutter pub cache repair
flutter pub get
```

### Model loading fails
- Check model path trong pubspec.yaml
- Verify model file exists
- Check model format (.tflite)

### Import errors
- Chạy `flutter pub get`
- Restart IDE
- Chạy `flutter clean`

## Next Steps

1. Đọc [QUICKSTART.md](QUICKSTART.md)
2. Setup ML environment
3. Tạo Flutter project
4. Copy templates
5. Train model
6. Test app

## Support

Nếu gặp vấn đề:
1. Check documentation trong `docs/`
2. Xem [README.md](README.md)
3. Check GitHub issues
4. Contact maintainer
