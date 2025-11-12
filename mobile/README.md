# Pneumonia Diagnosis Mobile App

Flutter app cho chẩn đoán viêm phổi từ ảnh X-quang sử dụng TensorFlow Lite.

## Tính năng

- **Chụp/Chọn ảnh X-quang** từ camera hoặc thư viện
- **On-device inference** với TensorFlow Lite
- **Preprocessing chính xác** theo chuẩn EfficientNet
- **Visualization heatmap** cho vùng tập trung của model
- **UI thân thiện** với Material Design 3
- **Placeholder MedGemma** cho phân tích AI y tế

## Kiến trúc

```
lib/
├── main.dart                    # Entry point
├── models/                      # Data models
│   ├── prediction_result.dart   # Kết quả dự đoán
│   └── tflite_model.dart       # Model metadata
├── services/                    # Business logic
│   ├── inference_service.dart   # TFLite inference
│   ├── image_service.dart      # Camera/Gallery
│   ├── preprocessing_service.dart # EfficientNet preprocessing
│   └── visualization_service.dart # Heatmap generation
└── screens/
    └── diagnosis_screen.dart    # Main UI
```

## Dependencies

- `tflite_flutter`: TensorFlow Lite inference
- `image_picker`: Camera/Gallery access
- `image`: Image processing
- `provider`: State management
- `flutter_spinkit`: Loading animations

## Setup & Build

### Prerequisites

1. **Flutter SDK** >= 3.0.0
2. **Android Studio** hoặc VS Code
3. **Android device/emulator** (API 24+)

### Installation

```bash
# 1. Navigate to mobile directory
cd mobile

# 2. Install dependencies
flutter pub get

# 3. Check Flutter setup
flutter doctor

# 4. Connect Android device or start emulator
flutter devices

# 5. Run app
flutter run
```

### Build APK

```bash
# Debug APK
flutter build apk --debug

# Release APK
flutter build apk --release

# APK location: build/app/outputs/flutter-apk/
```

## Model Integration

### 1. Tạo TFLite Model

```bash
# Từ thư mục gốc
cd ml

# Tạo dummy model (cho testing)
python scripts/create_dummy_p2_model.py --test

# Convert sang TFLite
python scripts/convert_p2_to_tflite.py --test
```

### 2. Copy Model vào Assets

```bash
# Copy TFLite model
cp ml/models/tflite/pneumonia_efficientnet_p2.tflite mobile/assets/models/

# Copy metadata (optional)
cp ml/models/tflite/pneumonia_efficientnet_p2.json mobile/assets/models/
```

### 3. Update pubspec.yaml

```yaml
flutter:
  assets:
    - assets/models/pneumonia_efficientnet_p2.tflite
    - assets/models/pneumonia_efficientnet_p2.json
```

## Permissions

App yêu cầu các permissions sau:

- **Camera**: Chụp ảnh X-quang
- **Storage/Photos**: Truy cập thư viện ảnh
- **Internet**: (Optional) Cho future features

### Android Permissions

Thêm vào `android/app/src/main/AndroidManifest.xml`:

```xml
<uses-permission android:name="android.permission.CAMERA" />
<uses-permission android:name="android.permission.READ_EXTERNAL_STORAGE" />
<uses-permission android:name="android.permission.WRITE_EXTERNAL_STORAGE" />
```

## Testing

### Unit Tests

```bash
flutter test
```

### Integration Tests

```bash
flutter test integration_test/
```

### Manual Testing

1. **Model Loading**: Kiểm tra model khởi tạo thành công
2. **Image Selection**: Test camera và gallery
3. **Preprocessing**: Verify EfficientNet preprocessing
4. **Inference**: Test prediction accuracy
5. **Visualization**: Check heatmap generation

## Performance

### Target Metrics

- **Inference Time**: < 2s trên mid-range device
- **Memory Usage**: < 100MB peak
- **Model Size**: < 20MB
- **App Size**: < 50MB

### Optimization

- **Model Quantization**: INT8 quantization
- **Image Compression**: Resize to 224x224
- **Memory Management**: Dispose resources properly
- **UI Optimization**: Lazy loading, caching

## Troubleshooting

### Common Issues

1. **TFLite Model Loading Failed**
   - Kiểm tra model file trong assets
   - Verify model format và compatibility

2. **Camera Permission Denied**
   - Check AndroidManifest.xml permissions
   - Request permissions at runtime

3. **Out of Memory**
   - Reduce image size
   - Dispose unused resources

4. **Slow Inference**
   - Enable GPU delegate
   - Use quantized model

### Debug Commands

```bash
# Check device logs
flutter logs

# Profile performance
flutter run --profile

# Analyze app size
flutter build apk --analyze-size
```

## Deployment

### Release Build

```bash
# Build release APK
flutter build apk --release --target-platform android-arm64

# Build App Bundle (for Play Store)
flutter build appbundle --release
```

### Code Signing

1. Tạo keystore file
2. Configure `android/key.properties`
3. Update `android/app/build.gradle`

## Future Enhancements

- [ ] **Multi-language support**
- [ ] **History/Export functionality**
- [ ] **Real MedGemma integration**
- [ ] **Batch processing**
- [ ] **Cloud sync**
- [ ] **Doctor consultation features**

## License

MIT License - See LICENSE file for details.
