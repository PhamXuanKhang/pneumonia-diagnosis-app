/// Model cho kết quả dự đoán
class PredictionResult {
  final double normalProbability;
  final double pneumoniaProbability;
  final String predictedClass;
  final double confidence;
  final DateTime timestamp;
  final String? imagePath;
  final List<double>? featureMaps;

  PredictionResult({
    required this.normalProbability,
    required this.pneumoniaProbability,
    required this.predictedClass,
    required this.confidence,
    required this.timestamp,
    this.imagePath,
    this.featureMaps,
  });

  /// Tạo từ raw prediction output
  factory PredictionResult.fromPrediction(
    double rawPrediction, {
    String? imagePath,
    List<double>? featureMaps,
  }) {
    final pneumoniaProbability = rawPrediction;
    final normalProbability = 1.0 - rawPrediction;
    final predictedClass = rawPrediction > 0.5 ? 'PNEUMONIA' : 'NORMAL';
    final confidence = rawPrediction > 0.5 ? rawPrediction : normalProbability;

    return PredictionResult(
      normalProbability: normalProbability,
      pneumoniaProbability: pneumoniaProbability,
      predictedClass: predictedClass,
      confidence: confidence,
      timestamp: DateTime.now(),
      imagePath: imagePath,
      featureMaps: featureMaps,
    );
  }

  /// Convert to JSON
  Map<String, dynamic> toJson() {
    return {
      'normalProbability': normalProbability,
      'pneumoniaProbability': pneumoniaProbability,
      'predictedClass': predictedClass,
      'confidence': confidence,
      'timestamp': timestamp.toIso8601String(),
      'imagePath': imagePath,
      'featureMaps': featureMaps,
    };
  }

  /// Create from JSON
  factory PredictionResult.fromJson(Map<String, dynamic> json) {
    return PredictionResult(
      normalProbability: json['normalProbability']?.toDouble() ?? 0.0,
      pneumoniaProbability: json['pneumoniaProbability']?.toDouble() ?? 0.0,
      predictedClass: json['predictedClass'] ?? 'UNKNOWN',
      confidence: json['confidence']?.toDouble() ?? 0.0,
      timestamp: DateTime.parse(json['timestamp']),
      imagePath: json['imagePath'],
      featureMaps: json['featureMaps']?.cast<double>(),
    );
  }

  /// Có phải là kết quả high confidence không
  bool get isHighConfidence => confidence > 0.8;

  /// Có phải là kết quả uncertain không
  bool get isUncertain => confidence < 0.6;

  /// Màu sắc cho UI dựa trên kết quả
  String get resultColor {
    if (predictedClass == 'PNEUMONIA') {
      return isHighConfidence ? '#F44336' : '#FF9800'; // Red or Orange
    } else {
      return isHighConfidence ? '#4CAF50' : '#FF9800'; // Green or Orange
    }
  }

  @override
  String toString() {
    return 'PredictionResult(class: $predictedClass, confidence: ${(confidence * 100).toStringAsFixed(1)}%)';
  }
}
