import 'dart:typed_data';
import 'dart:ui' as ui;
import 'package:image/image.dart' as img;

/// Service cho preprocessing ảnh theo chuẩn EfficientNet
class PreprocessingService {
  /// EfficientNet preprocessing: [0,255] -> [-1,1]
  static Float32List preprocessEfficientNet(img.Image image, {int targetSize = 224}) {
    // 1. Resize to target size
    final resized = img.copyResize(image, width: targetSize, height: targetSize);
    
    // 2. Convert to Float32 and normalize
    final input = Float32List(1 * targetSize * targetSize * 3);
    int pixelIndex = 0;
    
    for (int y = 0; y < targetSize; y++) {
      for (int x = 0; x < targetSize; x++) {
        final pixel = resized.getPixel(x, y);
        
        // EfficientNet preprocessing: (pixel / 255.0) * 2.0 - 1.0
        // Range: [0, 255] -> [0, 1] -> [-1, 1]
        input[pixelIndex++] = (img.getRed(pixel) / 255.0) * 2.0 - 1.0;   // R
        input[pixelIndex++] = (img.getGreen(pixel) / 255.0) * 2.0 - 1.0; // G  
        input[pixelIndex++] = (img.getBlue(pixel) / 255.0) * 2.0 - 1.0;  // B
      }
    }
    
    return input;
  }

  /// Preprocess từ UI Image
  static Future<Float32List> preprocessFromUIImage(ui.Image uiImage, {int targetSize = 224}) async {
    // Convert UI Image to bytes
    final byteData = await uiImage.toByteData(format: ui.ImageByteFormat.png);
    if (byteData == null) {
      throw Exception('Failed to convert UI Image to bytes');
    }
    
    // Decode to img.Image
    final image = img.decodeImage(byteData.buffer.asUint8List());
    if (image == null) {
      throw Exception('Failed to decode image');
    }
    
    return preprocessEfficientNet(image, targetSize: targetSize);
  }

  /// Preprocess từ file path
  static Future<Float32List> preprocessFromFile(String imagePath, {int targetSize = 224}) async {
    // Read image file
    final bytes = await _readImageFile(imagePath);
    
    // Decode to img.Image
    final image = img.decodeImage(bytes);
    if (image == null) {
      throw Exception('Failed to decode image from file: $imagePath');
    }
    
    return preprocessEfficientNet(image, targetSize: targetSize);
  }

  /// Preprocess từ Uint8List
  static Float32List preprocessFromBytes(Uint8List bytes, {int targetSize = 224}) {
    // Decode to img.Image
    final image = img.decodeImage(bytes);
    if (image == null) {
      throw Exception('Failed to decode image from bytes');
    }
    
    return preprocessEfficientNet(image, targetSize: targetSize);
  }

  /// Validate preprocessing output
  static bool validatePreprocessedData(Float32List data, {int expectedSize = 224}) {
    final expectedLength = expectedSize * expectedSize * 3;
    
    if (data.length != expectedLength) {
      print('Invalid data length: ${data.length}, expected: $expectedLength');
      return false;
    }
    
    // Check value range [-1, 1]
    for (final value in data) {
      if (value < -1.0 || value > 1.0) {
        print('Invalid value range: $value, expected: [-1, 1]');
        return false;
      }
    }
    
    return true;
  }

  /// Get preprocessing statistics
  static Map<String, double> getPreprocessingStats(Float32List data) {
    if (data.isEmpty) return {};
    
    double min = data[0];
    double max = data[0];
    double sum = 0.0;
    
    for (final value in data) {
      if (value < min) min = value;
      if (value > max) max = value;
      sum += value;
    }
    
    final mean = sum / data.length;
    
    // Calculate standard deviation
    double sumSquaredDiff = 0.0;
    for (final value in data) {
      final diff = value - mean;
      sumSquaredDiff += diff * diff;
    }
    final std = sumSquaredDiff / data.length;
    
    return {
      'min': min,
      'max': max,
      'mean': mean,
      'std': std,
      'length': data.length.toDouble(),
    };
  }

  /// Helper method to read image file
  static Future<Uint8List> _readImageFile(String imagePath) async {
    try {
      // This would be implemented based on the platform
      // For now, throw an exception as this needs platform-specific implementation
      throw UnimplementedError('File reading needs platform-specific implementation');
    } catch (e) {
      throw Exception('Failed to read image file: $e');
    }
  }

  /// Create test input for debugging
  static Float32List createTestInput({int targetSize = 224}) {
    final input = Float32List(1 * targetSize * targetSize * 3);
    
    // Fill with gradient pattern for testing
    int pixelIndex = 0;
    for (int y = 0; y < targetSize; y++) {
      for (int x = 0; x < targetSize; x++) {
        final normalizedX = (x / targetSize) * 2.0 - 1.0; // [-1, 1]
        final normalizedY = (y / targetSize) * 2.0 - 1.0; // [-1, 1]
        
        input[pixelIndex++] = normalizedX; // R
        input[pixelIndex++] = normalizedY; // G
        input[pixelIndex++] = 0.0;         // B
      }
    }
    
    return input;
  }
}
