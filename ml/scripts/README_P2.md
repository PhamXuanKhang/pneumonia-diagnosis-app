# Pipeline 2 (EfficientNetB0) Scripts

Các scripts được refactor từ `dat301m-training-pipeline-comparation.py` để phục vụ mục tiêu inference cho Pipeline 2.

## Scripts Overview

### 1. `create_dummy_p2_model.py`
Tạo dummy model P2 với random weights cho testing.

```bash
# Tạo dummy model
python ml/scripts/create_dummy_p2_model.py --output_path ml/models/saved_models/P2_EffNetB0_Baseline_final.keras --test

# Tạo model không compile
python ml/scripts/create_dummy_p2_model.py --no_compile
```

### 2. `convert_p2_to_tflite.py`
Convert Pipeline 2 model sang TensorFlow Lite format.

```bash
# Convert với quantization (recommended)
python ml/scripts/convert_p2_to_tflite.py \
    --model_path ml/models/saved_models/P2_EffNetB0_Baseline_final.keras \
    --output_path ml/models/tflite/pneumonia_efficientnet_p2.tflite \
    --data_root /path/to/chest_xray \
    --quantize \
    --test

# Convert multi-output model (cho visualization)
python ml/scripts/convert_p2_to_tflite.py \
    --multi_output \
    --output_path ml/models/tflite/pneumonia_efficientnet_p2_multi.tflite
```

**Outputs:**
- `.tflite` file: TensorFlow Lite model
- `.json` file: Model metadata

### 3. `validate_tflite_p2.py`
Validate TFLite model accuracy so với Keras model.

```bash
# Validate với full test set
python ml/scripts/validate_tflite_p2.py \
    --keras_model ml/models/saved_models/P2_EffNetB0_Baseline_final.keras \
    --tflite_model ml/models/tflite/pneumonia_efficientnet_p2.tflite \
    --data_root /path/to/chest_xray

# Validate với limited samples
python ml/scripts/validate_tflite_p2.py \
    --data_root /path/to/chest_xray \
    --max_samples 100
```

**Outputs:**
- `p2_validation_results.json`: Validation metrics và comparison

### 4. `evaluate_p2.py`
Evaluate Pipeline 2 model trên test dataset.

```bash
# Evaluate model
python ml/scripts/evaluate_p2.py \
    --model_path ml/models/saved_models/P2_EffNetB0_Baseline_final.keras \
    --data_root /path/to/chest_xray \
    --output_dir ml/outputs/evaluation
```

**Outputs:**
- `metrics.json`: Evaluation metrics
- `evaluation_data.json`: Prediction data
- `confusion_matrix.png`: Confusion matrix plot
- `roc_curve.png`: ROC curve plot
- `evaluation_summary.txt`: Text summary

## Module Dependencies

### `ml/src/data/preprocessing_p2.py`
- `PneumoniaDataGeneratorP2`: Data generator cho P2
- `create_representative_dataset_generator`: Tạo representative dataset cho quantization
- `preprocess_single_image`: Preprocess một ảnh đơn lẻ
- EfficientNet preprocessing functions

### `ml/src/models/efficientnet_p2.py`
- `build_efficientnet_p2_architecture`: Xây dựng P2 architecture
- `build_efficientnet_p2_multi_output`: Xây dựng P2 với multiple outputs
- `load_trained_p2_model`: Load trained model
- `create_p2_model_from_weights`: Tạo model từ architecture + weights

## Usage Workflow

### 1. Testing với Dummy Model
```bash
# Tạo dummy model cho testing
python ml/scripts/create_dummy_p2_model.py --test

# Convert dummy model
python ml/scripts/convert_p2_to_tflite.py --test

# Validate dummy model (sẽ fail vì không có real data)
python ml/scripts/validate_tflite_p2.py --data_root /path/to/dummy/data
```

### 2. Production với Real Model
```bash
# Giả sử đã có trained model tại ml/models/saved_models/P2_EffNetB0_Baseline_final.keras

# 1. Evaluate Keras model
python ml/scripts/evaluate_p2.py --data_root /path/to/chest_xray

# 2. Convert to TFLite
python ml/scripts/convert_p2_to_tflite.py \
    --data_root /path/to/chest_xray \
    --quantize --test

# 3. Validate TFLite model
python ml/scripts/validate_tflite_p2.py --data_root /path/to/chest_xray
```

## Key Features

### Pipeline 2 Specific
- **EfficientNetB0 architecture**: Pretrained ImageNet weights
- **EfficientNet preprocessing**: [-1, 1] normalization
- **Feature extraction**: từ `top_conv` layer (7x7x1280)
- **Binary classification**: NORMAL vs PNEUMONIA

### TFLite Optimization
- **Quantization**: INT8 quantization với representative dataset
- **Multi-output support**: Classification + feature maps cho visualization
- **Metadata export**: Model specs cho Flutter integration

### Validation & Testing
- **Accuracy retention**: So sánh Keras vs TFLite accuracy
- **Performance metrics**: Inference time, model size
- **Comprehensive evaluation**: Accuracy, Precision, Recall, F1, AUC-ROC

## File Structure
```
ml/
├── scripts/
│   ├── create_dummy_p2_model.py     # Dummy model creation
│   ├── convert_p2_to_tflite.py      # TFLite conversion
│   ├── validate_tflite_p2.py        # TFLite validation
│   ├── evaluate_p2.py               # Model evaluation
│   └── README_P2.md                 # This file
├── src/
│   ├── data/
│   │   └── preprocessing_p2.py      # P2 preprocessing module
│   └── models/
│       └── efficientnet_p2.py       # P2 model architecture
├── models/
│   ├── saved_models/
│   │   └── P2_EffNetB0_Baseline_final.keras  # Trained model
│   └── tflite/
│       ├── pneumonia_efficientnet_p2.tflite # TFLite model
│       └── pneumonia_efficientnet_p2.json   # Metadata
└── outputs/
    └── evaluation/                  # Evaluation results
```

## Notes

1. **Model Path**: Tất cả scripts mặc định sử dụng `ml/models/saved_models/P2_EffNetB0_Baseline_final.keras`
2. **Data Format**: Expects Kaggle chest X-ray dataset structure
3. **Preprocessing**: Sử dụng EfficientNet preprocessing ([-1, 1] normalization)
4. **Quantization**: Requires representative dataset từ real data
5. **Testing**: Dummy model có thể được sử dụng cho testing pipeline mà không cần real data
