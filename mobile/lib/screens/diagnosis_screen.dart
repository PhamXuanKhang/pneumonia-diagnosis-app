import 'dart:io';
import 'dart:typed_data';
import 'dart:ui' as ui;
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:flutter_spinkit/flutter_spinkit.dart';

import '../models/prediction_result.dart';
import '../services/inference_service.dart';
import '../services/image_service.dart';
import '../services/preprocessing_service.dart';
import '../services/visualization_service.dart';

class DiagnosisScreen extends StatefulWidget {
  const DiagnosisScreen({super.key});

  @override
  State<DiagnosisScreen> createState() => _DiagnosisScreenState();
}

class _DiagnosisScreenState extends State<DiagnosisScreen> {
  File? _selectedImage;
  PredictionResult? _predictionResult;
  ui.Image? _heatmapImage;
  bool _isLoading = false;
  bool _isModelInitialized = false;
  String _statusMessage = 'Khởi tạo model...';

  @override
  void initState() {
    super.initState();
    _initializeModel();
  }

  Future<void> _initializeModel() async {
    setState(() {
      _isLoading = true;
      _statusMessage = 'Đang khởi tạo TensorFlow Lite model...';
    });

    try {
      final inferenceService = context.read<InferenceService>();
      await inferenceService.initialize();
      
      setState(() {
        _isModelInitialized = true;
        _statusMessage = 'Model đã sẵn sàng';
        _isLoading = false;
      });

      // Show success message
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(
            content: Text('✅ Model đã được khởi tạo thành công'),
            backgroundColor: Colors.green,
          ),
        );
      }

    } catch (e) {
      setState(() {
        _isModelInitialized = false;
        _statusMessage = 'Lỗi khởi tạo model: $e';
        _isLoading = false;
      });

      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text('❌ Lỗi khởi tạo model: $e'),
            backgroundColor: Colors.red,
          ),
        );
      }
    }
  }

  Future<void> _selectImage() async {
    try {
      final imageService = context.read<ImageService>();
      final imageFile = await imageService.showImageSourceDialog(context);
      
      if (imageFile != null) {
        setState(() {
          _selectedImage = imageFile;
          _predictionResult = null;
          _heatmapImage = null;
        });
      }
    } catch (e) {
      _showErrorSnackBar('Lỗi chọn ảnh: $e');
    }
  }

  Future<void> _runDiagnosis() async {
    if (_selectedImage == null || !_isModelInitialized) return;

    setState(() {
      _isLoading = true;
      _statusMessage = 'Đang phân tích ảnh...';
    });

    try {
      // 1. Preprocessing
      setState(() => _statusMessage = 'Đang tiền xử lý ảnh...');
      final preprocessedData = await PreprocessingService.preprocessFromFile(
        _selectedImage!.path,
      );

      // 2. Inference
      setState(() => _statusMessage = 'Đang chạy dự đoán...');
      final inferenceService = context.read<InferenceService>();
      final result = await inferenceService.predict(
        preprocessedData,
        imagePath: _selectedImage!.path,
      );

      // 3. Visualization (nếu có feature maps)
      ui.Image? heatmap;
      if (result.featureMaps != null && result.featureMaps!.isNotEmpty) {
        setState(() => _statusMessage = 'Đang tạo visualization...');
        
        final visualizationService = context.read<VisualizationService>();
        heatmap = await visualizationService.generateHeatmap(
          result.featureMaps!,
          [7, 7, 1280], // EfficientNet feature map shape
          result.confidence,
        );
      }

      setState(() {
        _predictionResult = result;
        _heatmapImage = heatmap;
        _isLoading = false;
        _statusMessage = 'Phân tích hoàn tất';
      });

    } catch (e) {
      setState(() {
        _isLoading = false;
        _statusMessage = 'Lỗi phân tích';
      });
      _showErrorSnackBar('Lỗi phân tích: $e');
    }
  }

  Future<void> _runMedGemmaAnalysis() async {
    // Placeholder for MedGemma analysis
    setState(() {
      _isLoading = true;
      _statusMessage = 'Đang phân tích với MedGemma...';
    });

    // Simulate delay
    await Future.delayed(const Duration(seconds: 1));

    setState(() {
      _isLoading = false;
      _statusMessage = 'Phân tích MedGemma hoàn tất';
    });

    _showInfoDialog(
      'Phân tích MedGemma',
      'Đây là placeholder cho tính năng phân tích MedGemma.\n\n'
      'Trong phiên bản thực tế, đây sẽ là:\n'
      '• Phân tích chi tiết về vùng bất thường\n'
      '• Gợi ý chẩn đoán từ AI y tế\n'
      '• Mô tả các đặc điểm quan trọng\n'
      '• Khuyến nghị bước tiếp theo',
    );
  }

  void _showErrorSnackBar(String message) {
    if (mounted) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text(message),
          backgroundColor: Colors.red,
        ),
      );
    }
  }

  void _showInfoDialog(String title, String content) {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: Text(title),
        content: Text(content),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text('Đóng'),
          ),
        ],
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Chẩn Đoán Viêm Phổi'),
        backgroundColor: Theme.of(context).colorScheme.inversePrimary,
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            // Status Card
            _buildStatusCard(),
            
            const SizedBox(height: 16),
            
            // Image Selection Card
            _buildImageSelectionCard(),
            
            const SizedBox(height: 16),
            
            // Results Card
            if (_predictionResult != null) _buildResultsCard(),
            
            const SizedBox(height: 16),
            
            // Visualization Card
            if (_heatmapImage != null) _buildVisualizationCard(),
            
            const SizedBox(height: 16),
            
            // MedGemma Card
            _buildMedGemmaCard(),
          ],
        ),
      ),
    );
  }

  Widget _buildStatusCard() {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          children: [
            if (_isLoading)
              const SpinKitWave(
                color: Colors.blue,
                size: 30,
              ),
            const SizedBox(height: 8),
            Text(
              _statusMessage,
              style: Theme.of(context).textTheme.bodyMedium,
              textAlign: TextAlign.center,
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildImageSelectionCard() {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            Text(
              'Chọn Ảnh X-quang',
              style: Theme.of(context).textTheme.titleMedium,
            ),
            const SizedBox(height: 16),
            
            if (_selectedImage != null) ...[
              ClipRRect(
                borderRadius: BorderRadius.circular(8),
                child: Image.file(
                  _selectedImage!,
                  height: 200,
                  fit: BoxFit.cover,
                ),
              ),
              const SizedBox(height: 16),
            ],
            
            Row(
              children: [
                Expanded(
                  child: ElevatedButton.icon(
                    onPressed: _selectImage,
                    icon: const Icon(Icons.photo_camera),
                    label: const Text('Chọn Ảnh'),
                  ),
                ),
                const SizedBox(width: 16),
                Expanded(
                  child: ElevatedButton.icon(
                    onPressed: _selectedImage != null && _isModelInitialized && !_isLoading
                        ? _runDiagnosis
                        : null,
                    icon: const Icon(Icons.analytics),
                    label: const Text('Phân Tích'),
                  ),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildResultsCard() {
    final result = _predictionResult!;
    final isHighRisk = result.predictedClass == 'PNEUMONIA';
    
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            Text(
              'Kết Quả Chẩn Đoán',
              style: Theme.of(context).textTheme.titleMedium,
            ),
            const SizedBox(height: 16),
            
            // Main result
            Container(
              padding: const EdgeInsets.all(16),
              decoration: BoxDecoration(
                color: isHighRisk ? Colors.red.shade50 : Colors.green.shade50,
                borderRadius: BorderRadius.circular(8),
                border: Border.all(
                  color: isHighRisk ? Colors.red : Colors.green,
                  width: 2,
                ),
              ),
              child: Column(
                children: [
                  Icon(
                    isHighRisk ? Icons.warning : Icons.check_circle,
                    color: isHighRisk ? Colors.red : Colors.green,
                    size: 48,
                  ),
                  const SizedBox(height: 8),
                  Text(
                    result.predictedClass == 'PNEUMONIA' ? 'VIÊM PHỔI' : 'BÌNH THƯỜNG',
                    style: Theme.of(context).textTheme.headlineSmall?.copyWith(
                      color: isHighRisk ? Colors.red : Colors.green,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                  Text(
                    'Độ tin cậy: ${(result.confidence * 100).toStringAsFixed(1)}%',
                    style: Theme.of(context).textTheme.bodyMedium,
                  ),
                ],
              ),
            ),
            
            const SizedBox(height: 16),
            
            // Detailed probabilities
            Text(
              'Chi Tiết Xác Suất:',
              style: Theme.of(context).textTheme.titleSmall,
            ),
            const SizedBox(height: 8),
            
            _buildProbabilityBar('Bình thường', result.normalProbability, Colors.green),
            const SizedBox(height: 8),
            _buildProbabilityBar('Viêm phổi', result.pneumoniaProbability, Colors.red),
          ],
        ),
      ),
    );
  }

  Widget _buildProbabilityBar(String label, double probability, Color color) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          mainAxisAlignment: MainAxisAlignment.spaceBetween,
          children: [
            Text(label),
            Text('${(probability * 100).toStringAsFixed(1)}%'),
          ],
        ),
        const SizedBox(height: 4),
        LinearProgressIndicator(
          value: probability,
          backgroundColor: Colors.grey.shade300,
          valueColor: AlwaysStoppedAnimation<Color>(color),
        ),
      ],
    );
  }

  Widget _buildVisualizationCard() {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            Text(
              'Visualization (Heatmap)',
              style: Theme.of(context).textTheme.titleMedium,
            ),
            const SizedBox(height: 16),
            
            ClipRRect(
              borderRadius: BorderRadius.circular(8),
              child: RawImage(
                image: _heatmapImage,
                fit: BoxFit.cover,
                height: 200,
              ),
            ),
            
            const SizedBox(height: 8),
            Text(
              'Vùng màu đỏ cho thấy các khu vực mà model tập trung vào khi đưa ra dự đoán.',
              style: Theme.of(context).textTheme.bodySmall,
              textAlign: TextAlign.center,
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildMedGemmaCard() {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            Text(
              'Phân Tích MedGemma',
              style: Theme.of(context).textTheme.titleMedium,
            ),
            const SizedBox(height: 16),
            
            Container(
              padding: const EdgeInsets.all(16),
              decoration: BoxDecoration(
                color: Colors.blue.shade50,
                borderRadius: BorderRadius.circular(8),
                border: Border.all(color: Colors.blue.shade200),
              ),
              child: const Text(
                'Mô tả MedGemma hiển thị ở đây\n\n'
                'Đây là placeholder cho tính năng phân tích AI y tế nâng cao. '
                'Trong phiên bản thực tế sẽ cung cấp:\n'
                '• Phân tích chi tiết các vùng bất thường\n'
                '• Gợi ý chẩn đoán từ AI chuyên khoa\n'
                '• Khuyến nghị điều trị và theo dõi',
                style: TextStyle(color: Colors.blue),
              ),
            ),
            
            const SizedBox(height: 16),
            
            ElevatedButton.icon(
              onPressed: _isLoading ? null : _runMedGemmaAnalysis,
              icon: const Icon(Icons.psychology),
              label: const Text('Phân Tích MedGemma'),
            ),
          ],
        ),
      ),
    );
  }
}
