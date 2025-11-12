import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'screens/diagnosis_screen.dart';
import 'services/inference_service.dart';
import 'services/image_service.dart';
import 'services/visualization_service.dart';

void main() {
  runApp(const PneumoniaDiagnosisApp());
}

class PneumoniaDiagnosisApp extends StatelessWidget {
  const PneumoniaDiagnosisApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MultiProvider(
      providers: [
        // Singleton services
        Provider<InferenceService>.value(value: InferenceService()),
        Provider<ImageService>.value(value: ImageService()),
        Provider<VisualizationService>.value(value: VisualizationService()),
      ],
      child: MaterialApp(
        title: 'Pneumonia Diagnosis',
        theme: ThemeData(
          colorScheme: ColorScheme.fromSeed(
            seedColor: Colors.blue,
            brightness: Brightness.light,
          ),
          useMaterial3: true,
          appBarTheme: const AppBarTheme(
            centerTitle: true,
            elevation: 0,
          ),
          cardTheme: CardTheme(
            elevation: 4,
            shape: RoundedRectangleBorder(
              borderRadius: BorderRadius.circular(12),
            ),
          ),
          elevatedButtonTheme: ElevatedButtonThemeData(
            style: ElevatedButton.styleFrom(
              padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 12),
              shape: RoundedRectangleBorder(
                borderRadius: BorderRadius.circular(8),
              ),
            ),
          ),
        ),
        darkTheme: ThemeData(
          colorScheme: ColorScheme.fromSeed(
            seedColor: Colors.blue,
            brightness: Brightness.dark,
          ),
          useMaterial3: true,
          appBarTheme: const AppBarTheme(
            centerTitle: true,
            elevation: 0,
          ),
          cardTheme: CardTheme(
            elevation: 4,
            shape: RoundedRectangleBorder(
              borderRadius: BorderRadius.circular(12),
            ),
          ),
          elevatedButtonTheme: ElevatedButtonThemeData(
            style: ElevatedButton.styleFrom(
              padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 12),
              shape: RoundedRectangleBorder(
                borderRadius: BorderRadius.circular(8),
              ),
            ),
          ),
        ),
        themeMode: ThemeMode.system,
        home: const DiagnosisScreen(),
        debugShowCheckedModeBanner: false,
      ),
    );
  }
}
