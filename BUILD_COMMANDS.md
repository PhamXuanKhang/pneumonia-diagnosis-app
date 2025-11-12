# TOÀN BỘ CÁC CÂU LỆNH TERMINAL - PNEUMONIA DIAGNOSIS APP

## 🚀 SETUP & BUILD COMMANDS

### 1. SETUP PYTHON ML PIPELINE

```bash
# Navigate to project root
cd d:\tailieuki7\DAT301m\pneumonia-diagnosis-app

# Create Python virtual environment
python -m venv .venv

# Activate virtual environment (Windows)
.venv\Scripts\activate

# Install Python dependencies
pip install -r ml\requirements.txt

# Test ML pipeline với dummy model
python ml\scripts\create_dummy_p2_model.py --test

# Convert dummy model to TFLite
python ml\scripts\convert_p2_to_tflite.py --test
```

### 2. SETUP FLUTTER APP

```bash
# Navigate to mobile directory
cd mobile

# Check Flutter installation
flutter doctor

# Install Flutter dependencies
flutter pub get

# Copy TFLite model to assets (sau khi tạo từ ML pipeline)
copy ..\ml\models\tflite\pneumonia_efficientnet_p2.tflite assets\models\pneumonia_classifier.tflite

# Verify assets
dir assets\models\
```

### 3. ANDROID SETUP

```bash
# Check connected devices
flutter devices

# Start Android emulator (nếu cần)
# Mở Android Studio > AVD Manager > Start emulator

# Hoặc dùng command line
emulator -avd Pixel_4_API_30
```

### 4. RUN APP (DEVELOPMENT)

```bash
# Run in debug mode
flutter run

# Run with specific device
flutter run -d <device_id>

# Run with hot reload
flutter run --hot

# Run with verbose logging
flutter run --verbose
```

### 5. BUILD APK

```bash
# Build debug APK
flutter build apk --debug

# Build release APK
flutter build apk --release

# Build for specific architecture
flutter build apk --release --target-platform android-arm64

# Build App Bundle (for Google Play Store)
flutter build appbundle --release

# Check build output
dir build\app\outputs\flutter-apk\
```

### 6. INSTALL APK ON DEVICE

```bash
# Install debug APK
flutter install

# Install specific APK file
adb install build\app\outputs\flutter-apk\app-debug.apk

# Install release APK
adb install build\app\outputs\flutter-apk\app-release.apk

# Uninstall app
adb uninstall com.example.pneumonia_diagnosis
```

## 🔧 DEVELOPMENT COMMANDS

### Testing

```bash
# Run unit tests
flutter test

# Run tests with coverage
flutter test --coverage

# Run integration tests
flutter test integration_test\

# Analyze code
flutter analyze
```

### Performance & Debugging

```bash
# Profile app performance
flutter run --profile

# Debug with observatory
flutter run --debug --observatory-port=8888

# Check app size
flutter build apk --analyze-size

# View device logs
flutter logs

# Clear build cache
flutter clean
```

### Code Generation & Formatting

```bash
# Format code
flutter format lib\

# Generate code (if using code generation)
flutter packages pub run build_runner build

# Fix imports
flutter format --fix lib\
```

## 📱 DEVICE COMMANDS

### ADB Commands

```bash
# List connected devices
adb devices

# Install APK
adb install -r app-release.apk

# Uninstall app
adb uninstall com.example.pneumonia_diagnosis

# View device logs
adb logcat

# Clear app data
adb shell pm clear com.example.pneumonia_diagnosis

# Take screenshot
adb shell screencap -p /sdcard/screenshot.png
adb pull /sdcard/screenshot.png
```

### Device Info

```bash
# Check device info
adb shell getprop ro.build.version.release
adb shell getprop ro.product.model

# Check available storage
adb shell df -h

# Check memory usage
adb shell dumpsys meminfo com.example.pneumonia_diagnosis
```

## 🛠️ TROUBLESHOOTING COMMANDS

### Common Issues

```bash
# Flutter doctor issues
flutter doctor --verbose

# Clean and rebuild
flutter clean
flutter pub get
flutter build apk --release

# Fix Gradle issues
cd android
.\gradlew clean
cd ..

# Update Flutter
flutter upgrade

# Check Flutter channel
flutter channel
```

### Permission Issues

```bash
# Grant camera permission
adb shell pm grant com.example.pneumonia_diagnosis android.permission.CAMERA

# Grant storage permission
adb shell pm grant com.example.pneumonia_diagnosis android.permission.READ_EXTERNAL_STORAGE
adb shell pm grant com.example.pneumonia_diagnosis android.permission.WRITE_EXTERNAL_STORAGE

# List app permissions
adb shell dumpsys package com.example.pneumonia_diagnosis | findstr permission
```

## 🚀 COMPLETE BUILD WORKFLOW

### Full Build Process (từ đầu)

```bash
# 1. Setup project
cd d:\tailieuki7\DAT301m\pneumonia-diagnosis-app

# 2. Setup Python environment
python -m venv .venv
.venv\Scripts\activate
pip install -r ml\requirements.txt

# 3. Create and convert model
python ml\scripts\create_dummy_p2_model.py --test
python ml\scripts\convert_p2_to_tflite.py --test

# 4. Setup Flutter
cd mobile
flutter pub get

# 5. Copy model to assets
copy ..\ml\models\tflite\pneumonia_efficientnet_p2.tflite assets\models\pneumonia_classifier.tflite

# 6. Build APK
flutter build apk --release

# 7. Install on device
adb install build\app\outputs\flutter-apk\app-release.apk

echo "✅ Build completed successfully!"
echo "📱 APK location: mobile\build\app\outputs\flutter-apk\app-release.apk"
```

### Quick Development Cycle

```bash
# Quick development workflow
cd mobile

# 1. Clean and get dependencies
flutter clean && flutter pub get

# 2. Run app
flutter run --hot

# 3. Make changes and hot reload (press 'r' in terminal)

# 4. Build when ready
flutter build apk --debug
```

## 📋 VERIFICATION COMMANDS

### Pre-Build Checklist

```bash
# Check Flutter setup
flutter doctor

# Verify dependencies
flutter pub deps

# Check for issues
flutter analyze

# Run tests
flutter test

# Check device connection
flutter devices

# Verify assets
dir mobile\assets\models\
```

### Post-Build Verification

```bash
# Check APK size
dir mobile\build\app\outputs\flutter-apk\

# Install and test
adb install mobile\build\app\outputs\flutter-apk\app-release.apk

# Check app launches
adb shell am start -n com.example.pneumonia_diagnosis/.MainActivity

# Monitor logs
adb logcat | findstr "flutter"
```

## 🎯 PRODUCTION DEPLOYMENT

### Release Build

```bash
# 1. Update version in pubspec.yaml
# version: 1.0.0+1

# 2. Build release
flutter build apk --release --target-platform android-arm64

# 3. Sign APK (if configured)
# Requires keystore setup in android/key.properties

# 4. Test release build
adb install build\app\outputs\flutter-apk\app-release.apk

# 5. Generate App Bundle for Play Store
flutter build appbundle --release
```

### Performance Testing

```bash
# Profile build
flutter build apk --profile
flutter run --profile

# Memory profiling
flutter run --profile --trace-startup

# Size analysis
flutter build apk --analyze-size --target-platform android-arm64
```

---

## ⚡ QUICK REFERENCE

### Most Used Commands

```bash
# Development
flutter run --hot
flutter pub get
flutter clean

# Building
flutter build apk --release
adb install build\app\outputs\flutter-apk\app-release.apk

# Debugging
flutter logs
adb logcat
flutter analyze
```

### File Locations

- **APK Output**: `mobile\build\app\outputs\flutter-apk\`
- **TFLite Model**: `mobile\assets\models\pneumonia_classifier.tflite`
- **Source Code**: `mobile\lib\`
- **Dependencies**: `mobile\pubspec.yaml`

---

**🎉 Tất cả commands đã sẵn sàng để build và deploy ứng dụng Pneumonia Diagnosis!**
