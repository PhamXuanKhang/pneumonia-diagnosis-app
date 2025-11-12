# KẾ HOẠCH PHÁT TRIỂN FLUTTER APP - PNEUMONIA DIAGNOSIS

## PHÂN TÍCH HIỆN TRẠNG

### 1. Pipeline 2 (EfficientNetB0 Baseline) Analysis
Từ file `dat301m-training-pipeline-comparation.py`, Pipeline 2 có các đặc điểm:

**Model Architecture:**
- Base model: `EfficientNetB0` (ImageNet pretrained)
- Feature extraction layer: `top_conv` (7x7x1280)
- Input shape: (224, 224, 3)
- Output: Binary classification (NORMAL/PNEUMONIA)

**Preprocessing (`effnet_preprocess`):**
```python
# Từ line 559-579: EfficientNet preprocessing
preprocessing_function=effnet_preprocess.preprocess_input
```
- EfficientNet preprocessing: chuẩn hóa pixel từ [0,255] về [-1,1]
- Resize về (224, 224)
- RGB format

**Training Details:**
- Batch size: 32
- 2-stage training: head (7 epochs) + finetune (15 epochs)
- Early stopping với patience=3
- Adam optimizer với lr=1e-5 cho fine-tuning

### 2. Cấu Trúc Project Hiện Tại
```
pneumonia-diagnosis-app/
├── ml/                          # ML pipeline (Python)
├── mobile/                      # Flutter app (trống)
├── dat301m-training-pipeline-comparation.py  # Source code analysis
└── các file cấu hình khác...
```

## KẾ HOẠCH TRIỂN KHAI

---

#### 1.0. Yêu cầu Refactor (Agent Task)

**Nguồn:** `@dat301m-training-pipeline-comparation.py`

**Nhiệm vụ:** Agent, hãy đọc file notebook nguồn. Dựa vào đó, hãy **kiểm tra và thay thế/tạo mới** nội dung cho các file code trong thư mục `ml/` (như `ml/scripts/`, `ml/src/, ...`) để chúng thực hiện đúng chức năng như mô tả trong kế hoạch này.

**Quy tắc:**
1. **Một Pipeline:** Chỉ trích xuất logic liên quan đến **Pipeline 2 (EfficientNetB0 Baseline)**. Bỏ qua toàn bộ code của P1, P3, P4.
2. **Một Model:** Model đã được huấn luyện và nằm tại `ml/models/saved_models/P2_EffNetB0_Baseline_final.keras`. Mọi script (validation, conversion) phải load model từ đường dẫn này. Dùng code kiến trúc đã dựng sẵn và khi cần weights thì load từ model lên. **Tuyệt đối không train lại**.
3. **Mục tiêu (Inference):** Toàn bộ code phải được dọn dẹp để phục vụ mục tiêu *inference*, không phải *training*.
4. **Tách biệt logic:**
   * **Preprocessing:** Logic tiền xử lý của P2 (từ `PneumoniaDataGenerator` và `effnet_preprocess` trong Cell 5) cần được module hóa (ví dụ: `ml/src/data/preprocessing.py`) để `ml/scripts/convert_to_tflite.py` (cho `representative_dataset`) và `validate_tflite.py` có thể tái sử dụng.
   * **Data:** Logic của `PneumoniaDataGenerator` (Cell 5) chỉ giữ lại để tạo `representative_dataset` cho việc quantization (lượng tử hóa). Tận dụng tối đa các cấu trúc file hiện có.

---

### Phase 1: Model Conversion & Preparation

#### 1.1. Tạo Script Chuyển Đổi TFLite
**Vị trí:** `ml/scripts/convert_p2_to_tflite.py`

**Chức năng:**
- Load trained Pipeline 2 model (EfficientNetB0)
- Convert sang TensorFlow Lite với optimization
- Tạo representative dataset cho quantization
- Export model metadata (input/output specs)

```python
# Core logic sẽ include:
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen
tflite_model = converter.convert()
```

#### 1.2. Validation Script
**Vị trí:** `ml/scripts/validate_tflite.py`

**Chức năng:**
- So sánh accuracy giữa Keras model và TFLite model
- Kiểm tra inference time trên mobile
- Tạo test cases cho Flutter app

### Phase 2: Flutter Application Development

#### 2.1. Cấu Trúc Flutter App
**Vị trí:** `mobile/`

```
mobile/
├── lib/
│   ├── main.dart                    # Entry point
│   ├── models/
│   │   ├── tflite_model.dart       # TFLite wrapper
│   │   └── prediction_result.dart   # Result model
│   ├── services/
│   │   ├── image_service.dart      # Image picker/camera
│   │   ├── preprocessing_service.dart # On-device preprocessing
│   │   └── inference_service.dart   # TFLite inference
│   ├── widgets/
│   │   ├── image_picker_widget.dart
│   │   ├── result_display_widget.dart
│   │   └── visualization_widget.dart
│   └── screens/
│       └── diagnosis_screen.dart    # Main screen
├── assets/
│   └── models/
│       ├── pneumonia_efficientnet.tflite
│       └── model_metadata.json
└── pubspec.yaml
```

#### 2.2. Dependencies
**pubspec.yaml:**
```yaml
dependencies:
  flutter: ^3.16.0
  tflite_flutter: ^0.10.4        # TFLite inference
  image_picker: ^1.0.4           # Camera/gallery
  image: ^4.1.3                  # Image processing
  path_provider: ^2.1.1          # File system
  camera: ^0.10.5+5              # Camera access
```

### Phase 3: On-Device Preprocessing Implementation

#### 3.1. EfficientNet Preprocessing in Dart
**File:** `lib/services/preprocessing_service.dart`

**Yêu cầu cốt lõi:** Tái hiện chính xác preprocessing của EfficientNet

```dart
class PreprocessingService {
  // EfficientNet preprocessing: [0,255] -> [-1,1]
  static Float32List preprocessEfficientNet(img.Image image) {
    // 1. Resize to 224x224
    final resized = img.copyResize(image, width: 224, height: 224);
    
    // 2. Convert to Float32 and normalize
    final input = Float32List(1 * 224 * 224 * 3);
    int pixelIndex = 0;
    
    for (int y = 0; y < 224; y++) {
      for (int x = 0; x < 224; x++) {
        final pixel = resized.getPixel(x, y);
        // EfficientNet preprocessing: (pixel / 255.0) * 2.0 - 1.0
        input[pixelIndex++] = (img.getRed(pixel) / 255.0) * 2.0 - 1.0;   // R
        input[pixelIndex++] = (img.getGreen(pixel) / 255.0) * 2.0 - 1.0; // G  
        input[pixelIndex++] = (img.getBlue(pixel) / 255.0) * 2.0 - 1.0;  // B
      }
    }
    return input;
  }
}
```

#### 3.2. Validation Test
**File:** `test/preprocessing_test.dart`
- So sánh output preprocessing Dart vs Python
- Unit tests cho edge cases

### Phase 4: On-Device Visualization Solution

#### 4.1. Vấn Đề với Grad-CAM on-device
**Thách thức:**
- Grad-CAM cần tính gradients → không support trong TFLite
- Computational complexity cao cho mobile device
- Memory constraints

#### 4.2. Giải Pháp: Class Activation Mapping (CAM) Alternative

**Approach 1: Feature Map Visualization**
- Sử dụng Global Average Pooling layer của EfficientNet
- Extract feature maps từ `top_conv` layer (7x7x1280)
- Weighted sum dựa trên final classification weights

```dart
class VisualizationService {
  // Extract intermediate feature maps
  static Future<List<double>> extractFeatureMaps(
    Interpreter interpreter, 
    Float32List input
  ) async {
    // Run inference với custom output tensors
    // Output: [classification, feature_maps]
    // Resize feature maps từ 7x7 về 224x224 cho visualization
  }
}
```

**Approach 2: TFLite Model với Multiple Outputs**
- Modify conversion script để export model với 2 outputs:
  - Output 1: Classification (1 neuron)  
  - Output 2: Feature maps từ top_conv layer (7x7x1280)
- On-device sẽ process cả 2 outputs

#### 4.3. Implementation Strategy
**File:** `ml/scripts/convert_p2_multi_output.py`

```python
# Tạo model với multiple outputs
feature_extractor = Model(
    inputs=model.input,
    outputs=[
        model.get_layer('top_conv').output,  # Feature maps
        model.output                         # Classification
    ]
)
```

**File:** `lib/services/visualization_service.dart`

```dart
class VisualizationService {
  static Future<ui.Image> generateHeatmap(
    Float32List featureMaps,
    List<int> featureShape, // [7, 7, 1280]
    double classificationScore
  ) async {
    // 1. Weighted average của feature maps
    // 2. Resize 7x7 → 224x224
    // 3. Apply colormap (jet/hot)
    // 4. Overlay lên original image
  }
}
```

### Phase 5: Testing & Validation

#### 5.1. Accuracy Validation
- So sánh predictions giữa:
  - Python Keras model
  - TFLite model (Python)
  - TFLite model (Flutter)

#### 5.2. Performance Testing
- Inference latency trên các devices
- Memory usage monitoring
- Preprocessing timing

### Phase 6: UI/UX Implementation

#### 6.1. Main Screen Design
```dart
class DiagnosisScreen extends StatefulWidget {
  // Material Design 3
  // Dark/Light theme support
  // Accessibility features
}
```

**Features:**
- Image capture/selection
- Real-time preprocessing preview
- Prediction results với confidence score
- Heatmap visualization overlay
- History/export functionality

### Phase 7: Model Update Pipeline

#### 7.1. Model Versioning
- Semantic versioning cho TFLite models
- Compatibility checking
- Rollback mechanism

#### 7.2. CI/CD Integration
- GitHub Actions để auto-convert models
- Flutter integration tests
- Automated APK building

## TECHNICAL SPECIFICATIONS

### Hardware Requirements
- **Minimum:** Android 7.0 (API 24), 3GB RAM
- **Recommended:** Android 10.0+, 4GB+ RAM
- **Camera:** Auto-focus capability

### Performance Targets
- **Inference Time:** < 2 seconds trên mid-range device
- **Memory Usage:** < 100MB peak
- **Model Size:** < 20MB (after quantization)
- **Accuracy:** 95%+ retention từ original model

### Security Considerations
- Tất cả processing on-device
- Không upload ảnh lên server
- Local data encryption nếu cần cache

## RISK MITIGATION

### Technical Risks
1. **TFLite conversion accuracy loss**
   - Mitigation: Quantization-aware training
   - Fallback: FP16 instead of INT8

2. **Preprocessing differences Dart vs Python**
   - Mitigation: Extensive unit testing
   - Validation: Pixel-level comparison

3. **Visualization complexity**
   - Mitigation: Simplified CAM approach
   - Fallback: Attention regions instead của full heatmap

### Performance Risks
1. **Mobile inference too slow**
   - Mitigation: Model pruning, quantization
   - Hardware acceleration: GPU delegate

2. **Memory constraints**
   - Mitigation: Streaming inference
   - Batch size = 1

## DELIVERABLES TIMELINE

**Week 1-2:** Model conversion & validation
- TFLite conversion scripts
- Accuracy validation
- Performance benchmarking

**Week 3-4:** Flutter app skeleton
- Basic UI/UX
- Image capture functionality
- TFLite integration

**Week 5-6:** Preprocessing implementation
- Dart preprocessing service
- Validation tests
- Performance optimization

**Week 7-8:** Visualization feature
- Feature map extraction
- Heatmap generation
- UI integration

**Week 9-10:** Testing & polish
- End-to-end testing
- Performance optimization
- Documentation

## SUCCESS CRITERIA

1. **Functional:** App có thể classify X-ray images với accuracy tương tự Python model
2. **Performance:** Inference time < 2s trên target devices
3. **User Experience:** Intuitive interface với meaningful visualizations
4. **Technical:** 100% on-device processing, không cần internet
5. **Maintainable:** Clear code structure, comprehensive tests, CI/CD pipeline

---

**Tóm tắt:** Kế hoạch tập trung vào việc chuyển đổi Pipeline 2 (EfficientNetB0) sang TFLite, implement preprocessing chính xác trong Dart, và tạo giải pháp visualization khả thi cho mobile platform thông qua feature map extraction thay vì Grad-CAM truyền thống.
