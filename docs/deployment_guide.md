# Deployment Guide

Hướng dẫn deploy ML model và Mobile app.

## ML Model Deployment

### 1. Training Model

```bash
# Activate virtual environment
source venv/bin/activate  # Windows: venv\Scripts\activate

# Train model
cd ml
python scripts/train.py --config configs/training_config.yaml

# Evaluate
python scripts/evaluate.py \
  --model_path models/saved_models/pneumonia_mobilenetv2 \
  --config configs/training_config.yaml
```

### 2. Convert to TFLite

```bash
# Convert với quantization
python scripts/convert_to_tflite.py \
  --model_path models/saved_models/pneumonia_mobilenetv2 \
  --output_path models/tflite/pneumonia_mobilenetv2.tflite \
  --quantize \
  --test
```

### 3. Version Control với DVC

```bash
# Add model to DVC
dvc add models/saved_models/pneumonia_mobilenetv2
dvc add models/tflite/pneumonia_mobilenetv2.tflite

# Commit DVC files
git add models/saved_models/pneumonia_mobilenetv2.dvc
git add models/tflite/pneumonia_mobilenetv2.tflite.dvc
git commit -m "Add trained model v1.0"

# Push to remote storage
dvc push

# Tag version
git tag -a v1.0 -m "Model version 1.0"
git push origin v1.0
```

## Mobile App Deployment

### 1. Chuẩn bị

```bash
cd mobile

# Copy TFLite model
mkdir -p assets/models
cp ../ml/models/tflite/pneumonia_mobilenetv2.tflite assets/models/

# Tạo labels file
echo "NORMAL\nPNEUMONIA" > assets/models/labels.txt

# Update pubspec.yaml để include assets
# assets:
#   - assets/models/
```

### 2. Build Android

#### Debug Build
```bash
flutter build apk --debug
```

#### Release Build
```bash
# Tạo keystore (lần đầu)
keytool -genkey -v -keystore ~/upload-keystore.jks \
  -keyalg RSA -keysize 2048 -validity 10000 \
  -alias upload

# Tạo key.properties
# android/key.properties:
# storePassword=<password>
# keyPassword=<password>
# keyAlias=upload
# storeFile=<path-to-keystore>

# Build release APK
flutter build apk --release

# Build App Bundle (cho Google Play)
flutter build appbundle --release
```

Output: `build/app/outputs/flutter-apk/app-release.apk`

### 3. Build iOS

```bash
# Build iOS (cần macOS)
flutter build ios --release

# Hoặc build archive
flutter build ipa
```

### 4. Testing Build

```bash
# Install APK trên device
adb install build/app/outputs/flutter-apk/app-release.apk

# Hoặc
flutter install
```

## Google Play Store Deployment

### 1. Chuẩn bị

- Tạo Google Play Developer account
- Tạo app listing
- Chuẩn bị screenshots, icons, descriptions

### 2. Upload

```bash
# Build App Bundle
flutter build appbundle --release

# Upload file: build/app/outputs/bundle/release/app-release.aab
```

### 3. Release Tracks

- **Internal Testing**: Test với team
- **Closed Testing**: Test với beta users
- **Open Testing**: Public beta
- **Production**: Release chính thức

## Apple App Store Deployment

### 1. Chuẩn bị

- Apple Developer account
- App Store Connect setup
- Certificates và Provisioning Profiles

### 2. Build và Upload

```bash
# Build IPA
flutter build ipa

# Upload với Xcode hoặc Transporter app
```

### 3. App Store Review

- Submit for review
- Đợi approval (1-3 ngày)
- Release

## CI/CD với GitHub Actions

### Automated Deployment

File `.github/workflows/deploy.yml`:

```yaml
name: Deploy

on:
  push:
    tags:
      - 'v*'

jobs:
  deploy-android:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: subosito/flutter-action@v2
      - name: Build APK
        run: |
          cd mobile
          flutter build apk --release
      - name: Upload to Play Store
        uses: r0adkll/upload-google-play@v1
        with:
          serviceAccountJsonPlainText: ${{ secrets.SERVICE_ACCOUNT_JSON }}
          packageName: com.example.pneumonia_diagnosis
          releaseFiles: mobile/build/app/outputs/bundle/release/app-release.aab
          track: production
```

## Model Updates

### Over-The-Air (OTA) Updates

**Option 1: App Update**
- Release new app version với model mới
- Users update app từ store

**Option 2: Dynamic Model Loading**
```dart
// Download model từ server
Future<void> updateModel() async {
  final response = await http.get('https://api.example.com/models/latest');
  final modelBytes = response.bodyBytes;
  
  // Save to local storage
  final file = File('${appDir}/models/model.tflite');
  await file.writeAsBytes(modelBytes);
  
  // Reload model
  await tfliteModel.loadModel();
}
```

## Monitoring

### App Monitoring

**Firebase Crashlytics**
```dart
// Setup in main.dart
FlutterError.onError = FirebaseCrashlytics.instance.recordFlutterError;
```

**Firebase Analytics**
```dart
// Track events
FirebaseAnalytics.instance.logEvent(
  name: 'prediction_made',
  parameters: {'result': result.label},
);
```

### Model Monitoring

- Track prediction distribution
- Monitor confidence scores
- Collect feedback
- A/B testing

## Rollback

### Model Rollback

```bash
# Checkout previous version
git checkout v1.0

# Pull old model
dvc checkout models/tflite/pneumonia_mobilenetv2.tflite.dvc
dvc pull

# Rebuild app
cd mobile
flutter build apk --release
```

### App Rollback

- Google Play: Rollback to previous version
- App Store: Submit previous version

## Security

### API Keys
```bash
# Không commit API keys
# Sử dụng environment variables hoặc secrets

# .env
API_KEY=your_api_key_here

# Load trong app
import 'package:flutter_dotenv/flutter_dotenv.dart';
final apiKey = dotenv.env['API_KEY'];
```

### Code Obfuscation
```bash
flutter build apk --release --obfuscate --split-debug-info=build/debug-info
```

### Model Protection
- Encrypt model file
- Use ProGuard (Android)
- Use code obfuscation

## Troubleshooting

### Build Errors

**Android**
```bash
# Clean build
cd android
./gradlew clean
cd ..
flutter clean
flutter pub get
flutter build apk
```

**iOS**
```bash
cd ios
pod deintegrate
pod install
cd ..
flutter clean
flutter build ios
```

### Model Loading Errors

- Kiểm tra model path trong assets
- Verify model format (TFLite)
- Check input/output shapes
- Test trên emulator trước

### Performance Issues

- Profile app với Flutter DevTools
- Optimize image loading
- Use compute() cho heavy operations
- Implement caching

## Checklist

### Pre-Deployment
- [ ] Model trained và evaluated
- [ ] Model converted to TFLite
- [ ] Model tested trên mobile
- [ ] App tested trên devices
- [ ] Screenshots prepared
- [ ] Store listings ready
- [ ] Privacy policy created
- [ ] Terms of service created

### Post-Deployment
- [ ] Monitor crash reports
- [ ] Track analytics
- [ ] Collect user feedback
- [ ] Plan updates
- [ ] Monitor performance
