class AppConstants {
  // App info
  static const String appName = 'Pneumonia Diagnosis';
  static const String appVersion = '1.0.0';
  
  // Model info
  static const String modelName = 'pneumonia_mobilenetv2';
  static const int modelInputSize = 224;
  static const List<String> classNames = ['NORMAL', 'PNEUMONIA'];
  
  // Thresholds
  static const double confidenceThreshold = 0.7;
  static const double highConfidenceThreshold = 0.9;
  
  // Image settings
  static const int imageQuality = 90;
  static const int maxImageSize = 1024;
  
  // API endpoints (if using backend)
  static const String baseUrl = 'https://api.example.com';
  static const String predictEndpoint = '/predict';
  
  // Storage keys
  static const String keyHistory = 'diagnosis_history';
  static const String keySettings = 'app_settings';
  
  // Error messages
  static const String errorLoadModel = 'Không thể tải model';
  static const String errorLoadImage = 'Không thể tải ảnh';
  static const String errorPrediction = 'Lỗi khi phân tích ảnh';
  static const String errorPermission = 'Không có quyền truy cập';
}
