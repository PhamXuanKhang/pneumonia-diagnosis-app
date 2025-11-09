import 'dart:io';
import 'dart:typed_data';
import 'package:flutter/services.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:image/image.dart' as img;

class TFLiteModel {
  Interpreter? _interpreter;
  List<String>? _labels;
  
  static const String modelPath = 'assets/models/pneumonia_mobilenetv2.tflite';
  static const int inputSize = 224;
  
  // Load model
  Future<void> loadModel() async {
    try {
      _interpreter = await Interpreter.fromAsset(modelPath);
      print('Model loaded successfully');
      print('Input shape: ${_interpreter?.getInputTensor(0).shape}');
      print('Output shape: ${_interpreter?.getOutputTensor(0).shape}');
    } catch (e) {
      print('Error loading model: $e');
      rethrow;
    }
  }
  
  // Load labels
  Future<void> loadLabels() async {
    try {
      final labelsData = await rootBundle.loadString('assets/models/labels.txt');
      _labels = labelsData.split('\n').where((label) => label.isNotEmpty).toList();
      print('Labels loaded: $_labels');
    } catch (e) {
      print('Error loading labels: $e');
      _labels = ['NORMAL', 'PNEUMONIA'];
    }
  }
  
  // Preprocess image
  Float32List preprocessImage(img.Image image) {
    // Resize image to model input size
    img.Image resizedImage = img.copyResize(
      image,
      width: inputSize,
      height: inputSize,
    );
    
    // Convert to Float32List and normalize
    Float32List inputBytes = Float32List(1 * inputSize * inputSize * 3);
    int pixelIndex = 0;
    
    for (int y = 0; y < inputSize; y++) {
      for (int x = 0; x < inputSize; x++) {
        int pixel = resizedImage.getPixel(x, y);
        
        // Normalize to [0, 1]
        inputBytes[pixelIndex++] = img.getRed(pixel) / 255.0;
        inputBytes[pixelIndex++] = img.getGreen(pixel) / 255.0;
        inputBytes[pixelIndex++] = img.getBlue(pixel) / 255.0;
      }
    }
    
    return inputBytes;
  }
  
  // Run inference
  Future<Map<String, dynamic>> predict(File imageFile) async {
    if (_interpreter == null) {
      await loadModel();
    }
    
    if (_labels == null) {
      await loadLabels();
    }
    
    // Read and decode image
    final imageBytes = await imageFile.readAsBytes();
    img.Image? image = img.decodeImage(imageBytes);
    
    if (image == null) {
      throw Exception('Failed to decode image');
    }
    
    // Preprocess
    Float32List input = preprocessImage(image);
    
    // Reshape input to [1, 224, 224, 3]
    var inputShape = [1, inputSize, inputSize, 3];
    var inputBuffer = input.buffer.asFloat32List().reshape(inputShape);
    
    // Prepare output buffer [1, 2]
    var outputBuffer = List.filled(1 * 2, 0.0).reshape([1, 2]);
    
    // Run inference
    _interpreter!.run(inputBuffer, outputBuffer);
    
    // Get results
    List<double> probabilities = outputBuffer[0].cast<double>();
    
    // Find max probability
    int maxIndex = 0;
    double maxProb = probabilities[0];
    for (int i = 1; i < probabilities.length; i++) {
      if (probabilities[i] > maxProb) {
        maxProb = probabilities[i];
        maxIndex = i;
      }
    }
    
    return {
      'label': _labels![maxIndex],
      'confidence': maxProb,
      'probabilities': {
        _labels![0]: probabilities[0],
        _labels![1]: probabilities[1],
      },
    };
  }
  
  // Dispose interpreter
  void dispose() {
    _interpreter?.close();
  }
}
