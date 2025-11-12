import 'dart:typed_data';
import 'dart:ui' as ui;
import 'dart:math' as math;
import 'package:flutter/material.dart';
import 'package:image/image.dart' as img;

/// Service cho visualization (heatmap generation)
class VisualizationService {
  /// Singleton instance
  static final VisualizationService _instance = VisualizationService._internal();
  factory VisualizationService() => _instance;
  VisualizationService._internal();

  /// Generate heatmap từ feature maps
  Future<ui.Image> generateHeatmap(
    List<double> featureMaps,
    List<int> featureShape, // [7, 7, 1280]
    double classificationScore, {
    int targetWidth = 224,
    int targetHeight = 224,
  }) async {
    try {
      // 1. Reshape feature maps to [H, W, C]
      final height = featureShape[0]; // 7
      final width = featureShape[1];  // 7
      final channels = featureShape[2]; // 1280

      if (featureMaps.length != height * width * channels) {
        throw Exception('Feature maps size mismatch: ${featureMaps.length} vs ${height * width * channels}');
      }

      // 2. Weighted average của feature maps
      // Sử dụng classification score để weight các channels
      final heatmapData = <double>[];
      
      for (int h = 0; h < height; h++) {
        for (int w = 0; w < width; w++) {
          double pixelValue = 0.0;
          
          // Average across all channels với weight
          for (int c = 0; c < channels; c++) {
            final index = h * width * channels + w * channels + c;
            final featureValue = featureMaps[index];
            
            // Weight feature value với classification score
            pixelValue += featureValue * classificationScore;
          }
          
          pixelValue /= channels; // Average
          heatmapData.add(pixelValue);
        }
      }

      // 3. Normalize heatmap values to [0, 1]
      final normalizedHeatmap = _normalizeHeatmap(heatmapData);

      // 4. Resize từ 7x7 lên target size
      final resizedHeatmap = _resizeHeatmap(
        normalizedHeatmap, 
        height, width, 
        targetHeight, targetWidth
      );

      // 5. Apply colormap (jet colormap)
      final coloredHeatmap = _applyJetColormap(resizedHeatmap, targetHeight, targetWidth);

      // 6. Convert to UI Image
      return _createUIImageFromPixels(coloredHeatmap, targetWidth, targetHeight);

    } catch (e) {
      print('Error generating heatmap: $e');
      // Return empty/default heatmap
      return _createDefaultHeatmap(targetWidth, targetHeight);
    }
  }

  /// Overlay heatmap lên original image
  Future<ui.Image> overlayHeatmap(
    ui.Image originalImage,
    ui.Image heatmapImage, {
    double alpha = 0.4,
  }) async {
    try {
      // Convert UI Images to img.Image
      final originalBytes = await originalImage.toByteData(format: ui.ImageByteFormat.png);
      final heatmapBytes = await heatmapImage.toByteData(format: ui.ImageByteFormat.png);

      if (originalBytes == null || heatmapBytes == null) {
        throw Exception('Failed to convert images to bytes');
      }

      final originalImg = img.decodeImage(originalBytes.buffer.asUint8List());
      final heatmapImg = img.decodeImage(heatmapBytes.buffer.asUint8List());

      if (originalImg == null || heatmapImg == null) {
        throw Exception('Failed to decode images');
      }

      // Resize heatmap to match original image size
      final resizedHeatmap = img.copyResize(
        heatmapImg,
        width: originalImg.width,
        height: originalImg.height,
      );

      // Blend images
      final blended = img.Image(width: originalImg.width, height: originalImg.height);
      
      for (int y = 0; y < originalImg.height; y++) {
        for (int x = 0; x < originalImg.width; x++) {
          final originalPixel = originalImg.getPixel(x, y);
          final heatmapPixel = resizedHeatmap.getPixel(x, y);

          // Blend colors
          final blendedR = ((1 - alpha) * img.getRed(originalPixel) + alpha * img.getRed(heatmapPixel)).round();
          final blendedG = ((1 - alpha) * img.getGreen(originalPixel) + alpha * img.getGreen(heatmapPixel)).round();
          final blendedB = ((1 - alpha) * img.getBlue(originalPixel) + alpha * img.getBlue(heatmapPixel)).round();

          blended.setPixel(x, y, img.ColorRgb8(blendedR, blendedG, blendedB));
        }
      }

      // Convert back to UI Image
      final blendedBytes = img.encodePng(blended);
      final codec = await ui.instantiateImageCodec(blendedBytes);
      final frame = await codec.getNextFrame();
      
      return frame.image;

    } catch (e) {
      print('Error overlaying heatmap: $e');
      return originalImage; // Return original if overlay fails
    }
  }

  /// Create attention regions visualization (simplified version)
  Future<ui.Image> createAttentionRegions(
    List<double> featureMaps,
    List<int> featureShape,
    double threshold, {
    int targetWidth = 224,
    int targetHeight = 224,
  }) async {
    try {
      // Similar to heatmap but with binary regions
      final height = featureShape[0];
      final width = featureShape[1];
      final channels = featureShape[2];

      final attentionData = <double>[];
      
      for (int h = 0; h < height; h++) {
        for (int w = 0; w < width; w++) {
          double pixelValue = 0.0;
          
          for (int c = 0; c < channels; c++) {
            final index = h * width * channels + w * channels + c;
            pixelValue += featureMaps[index].abs();
          }
          
          pixelValue /= channels;
          
          // Apply threshold
          attentionData.add(pixelValue > threshold ? 1.0 : 0.0);
        }
      }

      // Resize and create binary mask
      final resizedAttention = _resizeHeatmap(
        attentionData, 
        height, width, 
        targetHeight, targetWidth
      );

      // Create binary colored regions
      final coloredRegions = <int>[];
      for (final value in resizedAttention) {
        if (value > 0.5) {
          // Red for attention regions
          coloredRegions.addAll([255, 0, 0, 128]); // RGBA
        } else {
          // Transparent
          coloredRegions.addAll([0, 0, 0, 0]);
        }
      }

      return _createUIImageFromRGBA(coloredRegions, targetWidth, targetHeight);

    } catch (e) {
      print('Error creating attention regions: $e');
      return _createDefaultHeatmap(targetWidth, targetHeight);
    }
  }

  /// Normalize heatmap values to [0, 1]
  List<double> _normalizeHeatmap(List<double> data) {
    if (data.isEmpty) return data;

    final minVal = data.reduce(math.min);
    final maxVal = data.reduce(math.max);
    
    if (maxVal == minVal) {
      return List.filled(data.length, 0.5);
    }

    return data.map((value) => (value - minVal) / (maxVal - minVal)).toList();
  }

  /// Resize heatmap using bilinear interpolation
  List<double> _resizeHeatmap(
    List<double> data,
    int srcHeight, int srcWidth,
    int dstHeight, int dstWidth,
  ) {
    final resized = <double>[];
    
    final scaleX = srcWidth / dstWidth;
    final scaleY = srcHeight / dstHeight;

    for (int y = 0; y < dstHeight; y++) {
      for (int x = 0; x < dstWidth; x++) {
        final srcX = x * scaleX;
        final srcY = y * scaleY;
        
        final x1 = srcX.floor();
        final y1 = srcY.floor();
        final x2 = math.min(x1 + 1, srcWidth - 1);
        final y2 = math.min(y1 + 1, srcHeight - 1);
        
        final dx = srcX - x1;
        final dy = srcY - y1;
        
        // Bilinear interpolation
        final v1 = data[y1 * srcWidth + x1];
        final v2 = data[y1 * srcWidth + x2];
        final v3 = data[y2 * srcWidth + x1];
        final v4 = data[y2 * srcWidth + x2];
        
        final interpolated = v1 * (1 - dx) * (1 - dy) +
                           v2 * dx * (1 - dy) +
                           v3 * (1 - dx) * dy +
                           v4 * dx * dy;
        
        resized.add(interpolated);
      }
    }

    return resized;
  }

  /// Apply jet colormap to heatmap
  List<int> _applyJetColormap(List<double> data, int height, int width) {
    final colored = <int>[];

    for (final value in data) {
      final color = _jetColormap(value);
      colored.addAll([color[0], color[1], color[2], 255]); // RGBA
    }

    return colored;
  }

  /// Jet colormap implementation
  List<int> _jetColormap(double value) {
    // Clamp value to [0, 1]
    value = math.max(0.0, math.min(1.0, value));

    int r, g, b;

    if (value < 0.25) {
      r = 0;
      g = (value * 4 * 255).round();
      b = 255;
    } else if (value < 0.5) {
      r = 0;
      g = 255;
      b = ((1 - (value - 0.25) * 4) * 255).round();
    } else if (value < 0.75) {
      r = ((value - 0.5) * 4 * 255).round();
      g = 255;
      b = 0;
    } else {
      r = 255;
      g = ((1 - (value - 0.75) * 4) * 255).round();
      b = 0;
    }

    return [r, g, b];
  }

  /// Create UI Image from RGBA pixels
  Future<ui.Image> _createUIImageFromPixels(List<int> pixels, int width, int height) async {
    final completer = Completer<ui.Image>();
    
    ui.decodeImageFromPixels(
      Uint8List.fromList(pixels),
      width,
      height,
      ui.PixelFormat.rgba8888,
      completer.complete,
    );

    return completer.future;
  }

  /// Create UI Image from RGBA data
  Future<ui.Image> _createUIImageFromRGBA(List<int> rgba, int width, int height) async {
    return _createUIImageFromPixels(rgba, width, height);
  }

  /// Create default/empty heatmap
  Future<ui.Image> _createDefaultHeatmap(int width, int height) async {
    final pixels = <int>[];
    
    // Create gradient for testing
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        final intensity = ((x + y) / (width + height) * 255).round();
        pixels.addAll([intensity, 0, 255 - intensity, 128]); // RGBA
      }
    }

    return _createUIImageFromPixels(pixels, width, height);
  }

  /// Get visualization statistics
  Map<String, dynamic> getVisualizationStats(List<double> featureMaps) {
    if (featureMaps.isEmpty) return {};

    final min = featureMaps.reduce(math.min);
    final max = featureMaps.reduce(math.max);
    final sum = featureMaps.reduce((a, b) => a + b);
    final mean = sum / featureMaps.length;

    // Calculate standard deviation
    final variance = featureMaps
        .map((x) => math.pow(x - mean, 2))
        .reduce((a, b) => a + b) / featureMaps.length;
    final std = math.sqrt(variance);

    return {
      'min': min,
      'max': max,
      'mean': mean,
      'std': std,
      'length': featureMaps.length,
      'non_zero_count': featureMaps.where((x) => x != 0.0).length,
    };
  }
}
