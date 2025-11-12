#!/usr/bin/env python3
"""
Script để test và chạy ứng dụng Streamlit Pipeline 2
"""

import sys
import os
from pathlib import Path

# Add current directory to Python path
sys.path.insert(0, str(Path(__file__).parent.absolute()))

def test_imports():
    """Test các import cần thiết"""
    print("Testing imports...")
    
    try:
        import streamlit as st
        print("✅ Streamlit imported successfully")
    except ImportError as e:
        print(f"❌ Streamlit import failed: {e}")
        return False
    
    try:
        import tensorflow as tf
        print(f"✅ TensorFlow {tf.__version__} imported successfully")
    except ImportError as e:
        print(f"❌ TensorFlow import failed: {e}")
        return False
    
    try:
        from streamlit_inference_p2 import Config
        print(f"✅ Config imported successfully")
        print(f"   Model path: {Config.MODEL_PATH}")
        print(f"   File exists: {os.path.exists(str(Config.MODEL_PATH))}")
    except ImportError as e:
        print(f"❌ Config import failed: {e}")
        return False
    
    return True

def main():
    """Main function"""
    print("=" * 50)
    print("Pipeline 2 - Streamlit App Test")
    print("=" * 50)
    
    if test_imports():
        print("\n✅ All tests passed! Ready to run Streamlit app.")
        print("\nTo run the app, use:")
        print("streamlit run streamlit_inference_p2.py")
    else:
        print("\n❌ Some tests failed. Please check the errors above.")

if __name__ == "__main__":
    main()
