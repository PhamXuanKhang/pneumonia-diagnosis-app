import 'dart:typed_data';
import 'dart:io';
import 'package:flutter/services.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:path_provider/path_provider.dart';
import 'package:path/path.dart' as path;

import '../models/prediction_result.dart';
import '../models/tflite_model.dart';

/// Service cho TFLite inference
class InferenceService {
  Interpreter? _interpreter;
  TFLiteModelInfo? _modelInfo;
  bool _isInitialized = false;

  /// Singleton instance
  static final InferenceService _instance = InferenceService._internal();
  factory InferenceService() => _instance;
  InferenceService._internal();

  /// Initialize TFLite model
  Future<void> initialize({
    String modelAssetPath = 'assets/models/pneumonia_efficientnet_p2.tflite',
    String? metadataAssetPath = 'assets/models/pneumonia_efficientnet_p2.json',
  }) async {
    try {
      print('Initializing TFLite model...');
      
      // 1. Load model từ assets
      final modelPath = await _loadModelFromAssets(modelAssetPath);
      
      // 2. Load metadata nếu có
      TFLiteModelInfo? modelInfo;
      if (metadataAssetPath != null) {
        try {
          modelInfo = await _loadModelMetadata(metadataAssetPath);
        } catch (e) {
          print('Warning: Could not load model metadata: $e');
        }
      }
      
      // 3. Create interpreter
      _interpreter = await Interpreter.fromFile(File(modelPath));
      _modelInfo = modelInfo;
      _isInitialized = true;
      
      print('✅ TFLite model initialized successfully');
      _printModelInfo();
      
    } catch (e) {
      print('❌ Failed to initialize TFLite model: $e');
      throw Exception('TFLite initialization failed: $e');
    }
  }

  /// Check if model is initialized
  bool get isInitialized => _isInitialized && _interpreter != null;

  /// Get model info
  TFLiteModelInfo? get modelInfo => _modelInfo;

  /// Run inference
  Future<PredictionResult> predict(Float32List inputData, {String? imagePath}) async {
    if (!isInitialized) {
      throw Exception('Model not initialized. Call initialize() first.');
    }

    try {
      final startTime = DateTime.now();
      
      // 1. Prepare input
      final input = inputData.reshape([1, 224, 224, 3]);
      
      // 2. Prepare outputs
      List<List<double>> outputs;
      List<double>? featureMaps;
      
      if (_modelInfo?.isMultiOutput == true) {
        // Multi-output model: [classification, feature_maps]
        final classificationOutput = List.filled(1, 0.0).reshape([1, 1]);
        final featureMapOutput = List.filled(7 * 7 * 1280, 0.0).reshape([1, 7, 7, 1280]);
        
        outputs = [classificationOutput, featureMapOutput];
        
        // Run inference
        _interpreter!.runForMultipleInputs([input], {
          0: classificationOutput,
          1: featureMapOutput,
        });
        
        // Extract results
        final prediction = classificationOutput[0][0];
        featureMaps = featureMapOutput[0].expand((x) => x).expand((x) => x).toList();
        
        final endTime = DateTime.now();
        final inferenceTime = endTime.difference(startTime).inMilliseconds;
        
        print('Inference completed in ${inferenceTime}ms');
        print('Prediction: $prediction');
        print('Feature maps shape: ${featureMaps.length}');
        
        return PredictionResult.fromPrediction(
          prediction,
          imagePath: imagePath,
          featureMaps: featureMaps,
        );
        
      } else {
        // Single output model: [classification]
        final output = List.filled(1, 0.0).reshape([1, 1]);
        
        // Run inference
        _interpreter!.run(input, output);
        
        final prediction = output[0][0];
        final endTime = DateTime.now();
        final inferenceTime = endTime.difference(startTime).inMilliseconds;
        
        print('Inference completed in ${inferenceTime}ms');
        print('Prediction: $prediction');
        
        return PredictionResult.fromPrediction(
          prediction,
          imagePath: imagePath,
        );
      }
      
    } catch (e) {
      print('❌ Inference failed: $e');
      throw Exception('Inference failed: $e');
    }
  }

  /// Run batch inference (for testing)
  Future<List<PredictionResult>> predictBatch(List<Float32List> inputBatch) async {
    final results = <PredictionResult>[];
    
    for (int i = 0; i < inputBatch.length; i++) {
      try {
        final result = await predict(inputBatch[i]);
        results.add(result);
      } catch (e) {
        print('Batch inference failed for item $i: $e');
        // Add dummy result
        results.add(PredictionResult.fromPrediction(0.5));
      }
    }
    
    return results;
  }

  /// Get model performance info
  Map<String, dynamic> getModelPerformanceInfo() {
    if (!isInitialized) return {};
    
    return {
      'model_name': _modelInfo?.modelName ?? 'Unknown',
      'model_size_mb': _modelInfo?.modelSizeMB ?? 0.0,
      'input_shape': _modelInfo?.inputShape ?? [1, 224, 224, 3],
      'is_quantized': _modelInfo?.isQuantized ?? false,
      'supports_visualization': _modelInfo?.supportsVisualization ?? false,
    };
  }

  /// Dispose resources
  void dispose() {
    _interpreter?.close();
    _interpreter = null;
    _modelInfo = null;
    _isInitialized = false;
    print('TFLite resources disposed');
  }

  /// Load model from assets to local storage
  Future<String> _loadModelFromAssets(String assetPath) async {
    try {
      // Get app documents directory
      final appDir = await getApplicationDocumentsDirectory();
      final modelDir = Directory(path.join(appDir.path, 'models'));
      
      if (!await modelDir.exists()) {
        await modelDir.create(recursive: true);
      }
      
      // Copy model from assets
      final modelFileName = path.basename(assetPath);
      final localModelPath = path.join(modelDir.path, modelFileName);
      
      // Check if model already exists
      final localModelFile = File(localModelPath);
      if (!await localModelFile.exists()) {
        print('Copying model from assets to local storage...');
        final byteData = await rootBundle.load(assetPath);
        await localModelFile.writeAsBytes(byteData.buffer.asUint8List());
        print('Model copied to: $localModelPath');
      } else {
        print('Model already exists at: $localModelPath');
      }
      
      return localModelPath;
      
    } catch (e) {
      throw Exception('Failed to load model from assets: $e');
    }
  }

  /// Load model metadata from assets
  Future<TFLiteModelInfo> _loadModelMetadata(String metadataAssetPath) async {
    try {
      final jsonString = await rootBundle.loadString(metadataAssetPath);
      final jsonData = jsonString; // This would need JSON parsing
      
      // For now, return default metadata
      return TFLiteModelInfo(
        modelPath: '',
        modelName: 'P2_EffNetB0_Baseline',
        inputShape: [1, 224, 224, 3],
        outputShapes: [[1, 1]],
        preprocessing: 'EfficientNet',
        isQuantized: true,
        isMultiOutput: false,
        modelSizeMB: 10.0,
      );
      
    } catch (e) {
      throw Exception('Failed to load model metadata: $e');
    }
  }

  /// Print model information
  void _printModelInfo() {
    if (_interpreter == null) return;
    
    print('=== TFLite Model Info ===');
    
    // Input details
    final inputTensors = _interpreter!.getInputTensors();
    print('Input tensors: ${inputTensors.length}');
    for (int i = 0; i < inputTensors.length; i++) {
      final tensor = inputTensors[i];
      print('  Input $i: ${tensor.shape} (${tensor.type})');
    }
    
    // Output details
    final outputTensors = _interpreter!.getOutputTensors();
    print('Output tensors: ${outputTensors.length}');
    for (int i = 0; i < outputTensors.length; i++) {
      final tensor = outputTensors[i];
      print('  Output $i: ${tensor.shape} (${tensor.type})');
    }
    
    if (_modelInfo != null) {
      print('Model metadata:');
      print('  Name: ${_modelInfo!.modelName}');
      print('  Size: ${_modelInfo!.modelSizeMB} MB');
      print('  Quantized: ${_modelInfo!.isQuantized}');
      print('  Multi-output: ${_modelInfo!.isMultiOutput}');
    }
    
    print('========================');
  }
}
