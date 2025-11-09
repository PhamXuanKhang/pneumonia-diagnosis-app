class DiagnosisResult {
  final String label;
  final double confidence;
  final Map<String, double> probabilities;
  final DateTime timestamp;
  final String? imagePath;
  
  DiagnosisResult({
    required this.label,
    required this.confidence,
    required this.probabilities,
    required this.timestamp,
    this.imagePath,
  });
  
  // Convert to JSON
  Map<String, dynamic> toJson() {
    return {
      'label': label,
      'confidence': confidence,
      'probabilities': probabilities,
      'timestamp': timestamp.toIso8601String(),
      'imagePath': imagePath,
    };
  }
  
  // Create from JSON
  factory DiagnosisResult.fromJson(Map<String, dynamic> json) {
    return DiagnosisResult(
      label: json['label'],
      confidence: json['confidence'],
      probabilities: Map<String, double>.from(json['probabilities']),
      timestamp: DateTime.parse(json['timestamp']),
      imagePath: json['imagePath'],
    );
  }
  
  // Get confidence level
  String get confidenceLevel {
    if (confidence >= 0.9) return 'Rất cao';
    if (confidence >= 0.7) return 'Cao';
    if (confidence >= 0.5) return 'Trung bình';
    return 'Thấp';
  }
  
  // Get confidence color
  String get confidenceColor {
    if (confidence >= 0.9) return 'green';
    if (confidence >= 0.7) return 'blue';
    if (confidence >= 0.5) return 'orange';
    return 'red';
  }
  
  // Is pneumonia detected
  bool get isPneumonia => label.toUpperCase() == 'PNEUMONIA';
  
  // Get formatted confidence
  String get formattedConfidence => '${(confidence * 100).toStringAsFixed(1)}%';
}
