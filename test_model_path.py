#!/usr/bin/env python3
"""
Test script để kiểm tra đường dẫn model
"""

from pathlib import Path
import os

# Test đường dẫn model
BASE_DIR = Path(__file__).parent.absolute()
MODEL_PATH = BASE_DIR / "ml" / "models" / "saved_models" / "P2_EffNetB0_Baseline_final.keras"

print(f"Base directory: {BASE_DIR}")
print(f"Model path: {MODEL_PATH}")
print(f"Model path (string): {str(MODEL_PATH)}")
print(f"File exists: {os.path.exists(str(MODEL_PATH))}")

if os.path.exists(str(MODEL_PATH)):
    print(f"File size: {os.path.getsize(str(MODEL_PATH))} bytes")
else:
    print("❌ File not found!")
    # List contents of models directory
    models_dir = BASE_DIR / "ml" / "models" / "saved_models"
    if models_dir.exists():
        print(f"Contents of {models_dir}:")
        for item in models_dir.iterdir():
            print(f"  - {item.name}")
    else:
        print(f"Directory {models_dir} does not exist!")
