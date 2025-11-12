import 'dart:io';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:path_provider/path_provider.dart';
import 'package:path/path.dart' as path;

/// Service cho xử lý ảnh (camera, gallery)
class ImageService {
  final ImagePicker _picker = ImagePicker();

  /// Singleton instance
  static final ImageService _instance = ImageService._internal();
  factory ImageService() => _instance;
  ImageService._internal();

  /// Chọn ảnh từ camera
  Future<File?> pickFromCamera() async {
    try {
      // Check camera permission
      final cameraPermission = await Permission.camera.request();
      if (!cameraPermission.isGranted) {
        throw Exception('Camera permission denied');
      }

      // Pick image from camera
      final XFile? image = await _picker.pickImage(
        source: ImageSource.camera,
        imageQuality: 85,
        maxWidth: 1024,
        maxHeight: 1024,
      );

      if (image == null) return null;

      // Save to app directory
      final savedFile = await _saveImageToAppDirectory(image);
      return savedFile;

    } catch (e) {
      print('Error picking image from camera: $e');
      throw Exception('Failed to capture image: $e');
    }
  }

  /// Chọn ảnh từ gallery
  Future<File?> pickFromGallery() async {
    try {
      // Check storage permission (for older Android versions)
      if (Platform.isAndroid) {
        final storagePermission = await Permission.storage.request();
        if (!storagePermission.isGranted) {
          // Try photos permission for newer Android versions
          final photosPermission = await Permission.photos.request();
          if (!photosPermission.isGranted) {
            throw Exception('Storage permission denied');
          }
        }
      }

      // Pick image from gallery
      final XFile? image = await _picker.pickImage(
        source: ImageSource.gallery,
        imageQuality: 85,
        maxWidth: 1024,
        maxHeight: 1024,
      );

      if (image == null) return null;

      // Save to app directory
      final savedFile = await _saveImageToAppDirectory(image);
      return savedFile;

    } catch (e) {
      print('Error picking image from gallery: $e');
      throw Exception('Failed to select image: $e');
    }
  }

  /// Show image source selection dialog
  Future<File?> showImageSourceDialog(BuildContext context) async {
    return showModalBottomSheet<File?>(
      context: context,
      builder: (BuildContext context) {
        return SafeArea(
          child: Wrap(
            children: [
              ListTile(
                leading: const Icon(Icons.photo_camera),
                title: const Text('Chụp ảnh'),
                onTap: () async {
                  Navigator.pop(context);
                  final file = await pickFromCamera();
                  Navigator.pop(context, file);
                },
              ),
              ListTile(
                leading: const Icon(Icons.photo_library),
                title: const Text('Chọn từ thư viện'),
                onTap: () async {
                  Navigator.pop(context);
                  final file = await pickFromGallery();
                  Navigator.pop(context, file);
                },
              ),
              ListTile(
                leading: const Icon(Icons.cancel),
                title: const Text('Hủy'),
                onTap: () => Navigator.pop(context),
              ),
            ],
          ),
        );
      },
    );
  }

  /// Validate image file
  bool validateImage(File imageFile) {
    try {
      // Check file exists
      if (!imageFile.existsSync()) {
        print('Image file does not exist');
        return false;
      }

      // Check file size (max 10MB)
      final fileSizeBytes = imageFile.lengthSync();
      final fileSizeMB = fileSizeBytes / (1024 * 1024);
      if (fileSizeMB > 10) {
        print('Image file too large: ${fileSizeMB.toStringAsFixed(2)} MB');
        return false;
      }

      // Check file extension
      final extension = path.extension(imageFile.path).toLowerCase();
      final validExtensions = ['.jpg', '.jpeg', '.png'];
      if (!validExtensions.contains(extension)) {
        print('Invalid image format: $extension');
        return false;
      }

      return true;

    } catch (e) {
      print('Error validating image: $e');
      return false;
    }
  }

  /// Get image info
  Future<Map<String, dynamic>> getImageInfo(File imageFile) async {
    try {
      final stat = await imageFile.stat();
      final bytes = await imageFile.readAsBytes();
      
      return {
        'path': imageFile.path,
        'name': path.basename(imageFile.path),
        'size_bytes': stat.size,
        'size_mb': (stat.size / (1024 * 1024)),
        'modified': stat.modified.toIso8601String(),
        'extension': path.extension(imageFile.path),
        'data_length': bytes.length,
      };

    } catch (e) {
      print('Error getting image info: $e');
      return {};
    }
  }

  /// Delete image file
  Future<bool> deleteImage(File imageFile) async {
    try {
      if (await imageFile.exists()) {
        await imageFile.delete();
        print('Image deleted: ${imageFile.path}');
        return true;
      }
      return false;
    } catch (e) {
      print('Error deleting image: $e');
      return false;
    }
  }

  /// Get app images directory
  Future<Directory> getImagesDirectory() async {
    final appDir = await getApplicationDocumentsDirectory();
    final imagesDir = Directory(path.join(appDir.path, 'images'));
    
    if (!await imagesDir.exists()) {
      await imagesDir.create(recursive: true);
    }
    
    return imagesDir;
  }

  /// List all saved images
  Future<List<File>> listSavedImages() async {
    try {
      final imagesDir = await getImagesDirectory();
      final entities = await imagesDir.list().toList();
      
      final imageFiles = entities
          .whereType<File>()
          .where((file) {
            final extension = path.extension(file.path).toLowerCase();
            return ['.jpg', '.jpeg', '.png'].contains(extension);
          })
          .toList();

      // Sort by modification date (newest first)
      imageFiles.sort((a, b) {
        final aStat = a.statSync();
        final bStat = b.statSync();
        return bStat.modified.compareTo(aStat.modified);
      });

      return imageFiles;

    } catch (e) {
      print('Error listing saved images: $e');
      return [];
    }
  }

  /// Clear all saved images
  Future<int> clearAllImages() async {
    try {
      final imageFiles = await listSavedImages();
      int deletedCount = 0;

      for (final file in imageFiles) {
        if (await deleteImage(file)) {
          deletedCount++;
        }
      }

      print('Deleted $deletedCount images');
      return deletedCount;

    } catch (e) {
      print('Error clearing images: $e');
      return 0;
    }
  }

  /// Save XFile to app directory
  Future<File> _saveImageToAppDirectory(XFile xFile) async {
    try {
      final imagesDir = await getImagesDirectory();
      final timestamp = DateTime.now().millisecondsSinceEpoch;
      final extension = path.extension(xFile.path);
      final fileName = 'image_$timestamp$extension';
      final savedPath = path.join(imagesDir.path, fileName);

      // Copy file to app directory
      final bytes = await xFile.readAsBytes();
      final savedFile = File(savedPath);
      await savedFile.writeAsBytes(bytes);

      print('Image saved to: $savedPath');
      return savedFile;

    } catch (e) {
      throw Exception('Failed to save image: $e');
    }
  }

  /// Check permissions
  Future<Map<String, bool>> checkPermissions() async {
    final cameraStatus = await Permission.camera.status;
    final storageStatus = await Permission.storage.status;
    final photosStatus = await Permission.photos.status;

    return {
      'camera': cameraStatus.isGranted,
      'storage': storageStatus.isGranted,
      'photos': photosStatus.isGranted,
    };
  }

  /// Request all necessary permissions
  Future<bool> requestPermissions() async {
    try {
      final permissions = <Permission>[];
      
      // Camera permission
      if (!(await Permission.camera.isGranted)) {
        permissions.add(Permission.camera);
      }

      // Storage permissions
      if (Platform.isAndroid) {
        if (!(await Permission.storage.isGranted)) {
          permissions.add(Permission.storage);
        }
        if (!(await Permission.photos.isGranted)) {
          permissions.add(Permission.photos);
        }
      }

      if (permissions.isEmpty) return true;

      final statuses = await permissions.request();
      
      // Check if all permissions granted
      return statuses.values.every((status) => status.isGranted);

    } catch (e) {
      print('Error requesting permissions: $e');
      return false;
    }
  }
}
