# Architecture Documentation

## Tổng quan hệ thống

Hệ thống gồm 2 phần chính:
1. **ML Pipeline**: Training, evaluation, và deployment models
2. **Mobile App**: Ứng dụng Flutter cho end users

## ML Architecture

### Data Flow

```
Raw Data → Preprocessing → Augmentation → Training → Model → TFLite → Mobile App
```

### Components

#### 1. Data Pipeline
- **Input**: Raw X-ray images
- **Processing**: 
  - Resize to 224x224
  - Normalize pixel values
  - Split train/val/test
  - Data augmentation
- **Output**: Processed datasets

#### 2. Model Architecture

**Transfer Learning (Recommended)**
```
Input (224x224x3)
    ↓
MobileNetV2 Base (frozen/unfrozen)
    ↓
GlobalAveragePooling2D
    ↓
Dense(512, relu) + Dropout(0.5)
    ↓
Dense(256, relu) + Dropout(0.3)
    ↓
Dense(2, softmax)
```

**Custom CNN**
```
Input (224x224x3)
    ↓
[Conv2D(32) → BatchNorm → ReLU → Conv2D(32) → BatchNorm → ReLU → MaxPool → Dropout] × 4
    ↓
GlobalAveragePooling2D
    ↓
Dense(512, relu) + Dropout(0.5)
    ↓
Dense(256, relu) + Dropout(0.5)
    ↓
Dense(2, softmax)
```

#### 3. Training Pipeline

```python
# Pseudocode
data = load_and_preprocess()
model = build_model()
model.compile(optimizer='adam', loss='categorical_crossentropy')

callbacks = [
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau,
    TensorBoard
]

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=50,
    callbacks=callbacks
)
```

#### 4. Evaluation Pipeline

- Confusion Matrix
- Classification Report
- ROC Curve (binary)
- Sample Predictions Visualization

#### 5. Model Conversion

```
TensorFlow SavedModel → TFLite Converter → Quantized TFLite Model
```

## Mobile Architecture

### Clean Architecture Layers

```
Presentation Layer (UI)
    ↓
Domain Layer (Business Logic)
    ↓
Data Layer (Data Sources)
```

### Components

#### 1. Presentation Layer
- **Screens**: Home, Camera, Gallery, Result, History
- **Widgets**: Reusable UI components
- **State Management**: Riverpod/Provider

#### 2. Domain Layer
- **Entities**: DiagnosisResult, ImageData
- **Use Cases**: PredictImage, SaveHistory, LoadHistory
- **Repository Interfaces**: Abstract definitions

#### 3. Data Layer
- **Models**: Data transfer objects
- **Repositories**: Implementation của interfaces
- **Data Sources**: 
  - Local: SharedPreferences, SQLite
  - ML: TFLite model inference

#### 4. ML Integration

```dart
// Inference Flow
Image File → Preprocessing → TFLite Model → Postprocessing → Result
```

**Preprocessing**:
1. Decode image
2. Resize to 224x224
3. Convert to Float32
4. Normalize [0, 1]
5. Reshape to [1, 224, 224, 3]

**Inference**:
```dart
interpreter.run(inputBuffer, outputBuffer)
```

**Postprocessing**:
1. Get probabilities
2. Find max probability
3. Map to class label
4. Format result

## MLOps Architecture

### DVC Pipeline

```yaml
stages:
  data_preparation → train → evaluate → convert_tflite
```

### CI/CD Pipeline

```
Git Push → GitHub Actions → Tests → Build → Deploy
```

**Workflows**:
1. **ML Pipeline**: Lint, test ML code
2. **Model Training**: Train model (manual trigger)
3. **Flutter Build**: Build APK/IPA

### Version Control

- **Code**: Git
- **Data**: DVC (Google Drive)
- **Models**: DVC (Google Drive)
- **Experiments**: DVC + Git tags

## Data Architecture

### Storage Structure

```
Google Drive (DVC Remote)
├── data/
│   ├── raw.dvc
│   └── processed.dvc
└── models/
    ├── saved_models.dvc
    └── tflite.dvc
```

### Local Structure

```
ml/
├── data/
│   ├── raw/           # Original data
│   ├── processed/     # Processed data
│   └── external/      # External data
└── models/
    ├── checkpoints/   # Training checkpoints
    ├── saved_models/  # TF SavedModel
    └── tflite/        # TFLite models
```

## Security Considerations

1. **Data Privacy**
   - No PHI (Protected Health Information) in code
   - Local processing on device
   - Optional cloud backup with encryption

2. **Model Security**
   - Model files in assets (not exposed)
   - No model API endpoints (offline inference)

3. **App Security**
   - Secure storage for history
   - Permission handling
   - Input validation

## Performance Optimization

### ML Performance
- **Model**: MobileNetV2 (lightweight)
- **Quantization**: Float16 quantization
- **Input Size**: 224x224 (balance accuracy/speed)

### Mobile Performance
- **Lazy Loading**: Load model on demand
- **Caching**: Cache inference results
- **Background Processing**: Run inference in isolate
- **Memory Management**: Dispose resources properly

## Scalability

### ML Scalability
- **Training**: GPU support, distributed training
- **Data**: DVC for large datasets
- **Experiments**: DVC experiments tracking

### Mobile Scalability
- **Users**: Offline-first architecture
- **Features**: Modular architecture
- **Platforms**: Android + iOS support

## Monitoring

### ML Monitoring
- **Training**: TensorBoard
- **Metrics**: DVC metrics tracking
- **Experiments**: DVC experiments

### Mobile Monitoring
- **Crashes**: Firebase Crashlytics (optional)
- **Analytics**: Firebase Analytics (optional)
- **Performance**: Flutter DevTools

## Future Enhancements

1. **ML**
   - Multi-class classification
   - Ensemble models
   - Active learning
   - Model compression

2. **Mobile**
   - Real-time camera inference
   - Cloud sync
   - Multi-language support
   - Accessibility features

3. **MLOps**
   - Automated retraining
   - A/B testing
   - Model monitoring
   - Continuous deployment
