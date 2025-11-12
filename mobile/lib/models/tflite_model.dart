/// Model wrapper cho TFLite model
class TFLiteModelInfo {
  final String modelPath;
  final String modelName;
  final List<int> inputShape;
  final List<List<int>> outputShapes;
  final String preprocessing;
  final bool isQuantized;
  final bool isMultiOutput;
  final double modelSizeMB;

  TFLiteModelInfo({
    required this.modelPath,
    required this.modelName,
    required this.inputShape,
    required this.outputShapes,
    required this.preprocessing,
    required this.isQuantized,
    required this.isMultiOutput,
    required this.modelSizeMB,
  });

  /// Tạo từ metadata JSON
  factory TFLiteModelInfo.fromJson(Map<String, dynamic> json) {
    return TFLiteModelInfo(
      modelPath: json['model_path'] ?? '',
      modelName: json['model_name'] ?? 'Unknown',
      inputShape: List<int>.from(json['input_shape'] ?? [1, 224, 224, 3]),
      outputShapes: (json['output_shapes'] as List?)
          ?.map((shape) => List<int>.from(shape))
          .toList() ?? [[1, 1]],
      preprocessing: json['preprocessing'] ?? 'EfficientNet',
      isQuantized: json['quantized'] ?? false,
      isMultiOutput: json['multi_output'] ?? false,
      modelSizeMB: (json['model_size_mb'] ?? 0.0).toDouble(),
    );
  }

  /// Convert to JSON
  Map<String, dynamic> toJson() {
    return {
      'model_path': modelPath,
      'model_name': modelName,
      'input_shape': inputShape,
      'output_shapes': outputShapes,
      'preprocessing': preprocessing,
      'quantized': isQuantized,
      'multi_output': isMultiOutput,
      'model_size_mb': modelSizeMB,
    };
  }

  /// Lấy input width
  int get inputWidth => inputShape.length >= 3 ? inputShape[2] : 224;

  /// Lấy input height
  int get inputHeight => inputShape.length >= 2 ? inputShape[1] : 224;

  /// Lấy input channels
  int get inputChannels => inputShape.length >= 4 ? inputShape[3] : 3;

  /// Có hỗ trợ visualization không
  bool get supportsVisualization => isMultiOutput && outputShapes.length >= 2;

  @override
  String toString() {
    return 'TFLiteModelInfo(name: $modelName, input: $inputShape, quantized: $isQuantized)';
  }
}
