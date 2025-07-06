#!/usr/bin/env python3
"""
Simple startup script for the Student Performance Predictor Flask app.
This script will start the app and show the model loading status.
"""

import os
import sys

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    # Import the Flask app
    from app import app, loaded_models, preprocessors_loaded_successfully, MODEL_DIR

    print("=" * 60)
    print("🚀 STUDENT PERFORMANCE PREDICTOR - STARTING UP")
    print("=" * 60)

    # Check model loading status
    print(f"📁 Models directory: {MODEL_DIR}")
    print(f"📊 Models loaded: {len(loaded_models)}")
    print(f"🔧 Preprocessors loaded: {preprocessors_loaded_successfully}")

    if loaded_models:
        print("\n✅ LOADED MODELS:")
        for filename, model in loaded_models.items():
            model_type = type(model).__name__
            print(f"   • {filename} ({model_type})")
    else:
        print("\n❌ NO MODELS LOADED!")
        print("   Check if model files exist in the models/ directory")

    # Check for XGBoost models specifically
    xgb_models = [f for f in loaded_models.keys() if 'xgb' in f.lower()]
    if xgb_models:
        print(f"\n🎯 XGBoost models found: {len(xgb_models)}")
        for xgb_model in xgb_models:
            print(f"   • {xgb_model}")
    else:
        print("\n⚠️  NO XGBOOST MODELS FOUND!")
        print("   The XGBoost prediction tab may not work properly.")

    print("\n" + "=" * 60)
    print("🌐 Starting Flask app...")
    print("   URL: http://127.0.0.1:5000")
    print("   Press Ctrl+C to stop")
    print("=" * 60)

    # Start the Flask app
    app.run(host='127.0.0.1', port=5000, debug=True)

except ImportError as e:
    print(f"❌ Error importing Flask app: {e}")
    print("Make sure you're in the correct directory and all dependencies are installed.")
except Exception as e:
    print(f"❌ Error starting app: {e}")
    import traceback
    traceback.print_exc()
