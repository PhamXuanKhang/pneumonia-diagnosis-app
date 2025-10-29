# Hướng dẫn khởi tạo Flutter Project

## Bước 1: Cài đặt Flutter

### Windows
```powershell
# Download Flutter SDK từ https://flutter.dev/docs/get-started/install/windows
# Giải nén và thêm vào PATH

# Kiểm tra cài đặt
flutter doctor
```

### macOS/Linux
```bash
# Download Flutter SDK
git clone https://github.com/flutter/flutter.git -b stable
export PATH="$PATH:`pwd`/flutter/bin"

# Kiểm tra cài đặt
flutter doctor
```

## Bước 2: Khởi tạo Flutter Project

```bash
# Di chuyển vào thư mục dự án
cd pneumonia-diagnosis-app

# Tạo Flutter project
flutter create mobile

# Di chuyển vào thư mục mobile
cd mobile
```

## Bước 3: Cấu hình Dependencies

Thêm các dependencies vào `mobile/pubspec.yaml`:

```yaml
dependencies:
  flutter:
    sdk: flutter
  
  # State management
  provider: ^6.1.1
  riverpod: ^2.4.9
  flutter_riverpod: ^2.4.9
  
  # TFLite
  tflite_flutter: ^0.10.4
  tflite_flutter_helper: ^0.3.1
  
  # Image processing
  image_picker: ^1.0.5
  image: ^4.1.3
  
  # Camera
  camera: ^0.10.5+7
  
  # Permissions
  permission_handler: ^11.1.0
  
  # UI
  cupertino_icons: ^1.0.6
  google_fonts: ^6.1.0
  
  # HTTP
  http: ^1.1.2
  dio: ^5.4.0
  
  # Local storage
  shared_preferences: ^2.2.2
  path_provider: ^2.1.1
  
  # Utils
  intl: ^0.18.1
  logger: ^2.0.2+1

dev_dependencies:
  flutter_test:
    sdk: flutter
  flutter_lints: ^3.0.1
  mockito: ^5.4.4
  build_runner: ^2.4.7
```

Sau đó chạy:
```bash
flutter pub get
```

## Bước 4: Cấu hình Android

### `mobile/android/app/build.gradle`

```gradle
android {
    compileSdkVersion 34
    
    defaultConfig {
        applicationId "com.example.pneumonia_diagnosis"
        minSdkVersion 21
        targetSdkVersion 34
        versionCode 1
        versionName "1.0"
    }
    
    buildTypes {
        release {
            signingConfig signingConfigs.debug
        }
    }
}
```

### `mobile/android/app/src/main/AndroidManifest.xml`

Thêm permissions:
```xml
<uses-permission android:name="android.permission.CAMERA" />
<uses-permission android:name="android.permission.READ_EXTERNAL_STORAGE" />
<uses-permission android:name="android.permission.WRITE_EXTERNAL_STORAGE" />
<uses-permission android:name="android.permission.INTERNET" />
```

## Bước 5: Cấu hình iOS

### `mobile/ios/Runner/Info.plist`

Thêm permissions:
```xml
<key>NSCameraUsageDescription</key>
<string>App cần truy cập camera để chụp ảnh X-quang</string>
<key>NSPhotoLibraryUsageDescription</key>
<string>App cần truy cập thư viện ảnh để chọn ảnh X-quang</string>
```

## Bước 6: Copy TFLite Model

```bash
# Copy model từ ML folder
mkdir -p mobile/assets/models
cp ml/models/tflite/pneumonia_mobilenetv2.tflite mobile/assets/models/

# Cập nhật pubspec.yaml
# Thêm vào phần assets:
#   assets:
#     - assets/models/
#     - assets/images/
```

## Bước 7: Cấu trúc Clean Architecture

Tạo cấu trúc thư mục:

```
mobile/lib/
├── core/
│   ├── config/
│   ├── constants/
│   ├── theme/
│   └── utils/
├── data/
│   ├── models/
│   ├── repositories/
│   └── datasources/
├── domain/
│   ├── entities/
│   ├── repositories/
│   └── usecases/
├── presentation/
│   ├── screens/
│   ├── widgets/
│   └── providers/
├── ml/
│   ├── tflite_model.dart
│   └── image_preprocessing.dart
└── main.dart
```

## Bước 8: Chạy App

```bash
# Kiểm tra devices
flutter devices

# Chạy trên Android emulator/device
flutter run

# Chạy trên iOS simulator/device
flutter run

# Build APK
flutter build apk --release

# Build iOS
flutter build ios --release
```

## Bước 9: Testing

```bash
# Run unit tests
flutter test

# Run integration tests
flutter test integration_test

# Run with coverage
flutter test --coverage
```

## Lưu ý quan trọng

1. **TFLite Model**: Đảm bảo model đã được convert sang TFLite format
2. **Permissions**: Cấu hình đầy đủ permissions cho camera và storage
3. **Image Size**: Resize ảnh về đúng kích thước input của model (224x224)
4. **Preprocessing**: Normalize ảnh giống như trong training
5. **Error Handling**: Xử lý các trường hợp lỗi khi load model hoặc inference

## Troubleshooting

### Lỗi TFLite
```bash
# Thêm vào android/app/build.gradle
android {
    aaptOptions {
        noCompress 'tflite'
    }
}
```

### Lỗi Camera Permission
- Kiểm tra AndroidManifest.xml và Info.plist
- Request permission runtime trong code

### Lỗi Build
```bash
# Clean và rebuild
flutter clean
flutter pub get
flutter run
```

## Tài liệu tham khảo

- [Flutter Documentation](https://flutter.dev/docs)
- [TFLite Flutter Plugin](https://pub.dev/packages/tflite_flutter)
- [Clean Architecture Flutter](https://resocoder.com/flutter-clean-architecture-tdd/)
