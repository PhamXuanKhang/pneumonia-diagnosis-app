import 'dart:io';
import 'package:image/image.dart' as img;

class ImagePreprocessing {
  // Resize image
  static img.Image resizeImage(img.Image image, int width, int height) {
    return img.copyResize(image, width: width, height: height);
  }
  
  // Convert to grayscale
  static img.Image toGrayscale(img.Image image) {
    return img.grayscale(image);
  }
  
  // Normalize image
  static List<double> normalizeImage(img.Image image) {
    List<double> normalized = [];
    
    for (int y = 0; y < image.height; y++) {
      for (int x = 0; x < image.width; x++) {
        int pixel = image.getPixel(x, y);
        
        double r = img.getRed(pixel) / 255.0;
        double g = img.getGreen(pixel) / 255.0;
        double b = img.getBlue(pixel) / 255.0;
        
        normalized.addAll([r, g, b]);
      }
    }
    
    return normalized;
  }
  
  // Load image from file
  static Future<img.Image?> loadImage(File file) async {
    try {
      final bytes = await file.readAsBytes();
      return img.decodeImage(bytes);
    } catch (e) {
      print('Error loading image: $e');
      return null;
    }
  }
  
  // Save image to file
  static Future<void> saveImage(img.Image image, String path) async {
    try {
      final bytes = img.encodeJpg(image);
      await File(path).writeAsBytes(bytes);
    } catch (e) {
      print('Error saving image: $e');
    }
  }
  
  // Apply contrast enhancement
  static img.Image enhanceContrast(img.Image image, {double contrast = 1.2}) {
    return img.adjustColor(image, contrast: contrast);
  }
  
  // Apply brightness adjustment
  static img.Image adjustBrightness(img.Image image, {int brightness = 0}) {
    return img.adjustColor(image, brightness: brightness);
  }
}
