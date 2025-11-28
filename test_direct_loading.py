"""
Direct model loading test
"""

import joblib
import pickle
import os

def test_direct_loading():
    """Test loading models directly with different methods"""
    
    model_files = {
        'Main Model': 'models/best_price_prediction_model.pkl',
        'Feature Scaler': 'models/feature_scaler.pkl'
    }
    
    for name, filepath in model_files.items():
        print(f"\n🔍 Testing {name}: {filepath}")
        
        if not os.path.exists(filepath):
            print("   ❌ File does not exist")
            continue
        
        print(f"   File size: {os.path.getsize(filepath)} bytes")
        
        # Try joblib first
        try:
            print("   Trying joblib.load()...")
            obj = joblib.load(filepath)
            print(f"   ✅ joblib: Type = {type(obj)}")
            if hasattr(obj, 'predict'):
                print("   ✅ joblib: Has predict method")
            if hasattr(obj, 'transform'):
                print("   ✅ joblib: Has transform method")
        except Exception as e:
            print(f"   ❌ joblib error: {e}")
        
        # Try pickle
        try:
            print("   Trying pickle.load()...")
            with open(filepath, 'rb') as f:
                obj = pickle.load(f)
            print(f"   ✅ pickle: Type = {type(obj)}")
            if hasattr(obj, 'predict'):
                print("   ✅ pickle: Has predict method")
            if hasattr(obj, 'transform'):
                print("   ✅ pickle: Has transform method")
        except Exception as e:
            print(f"   ❌ pickle error: {e}")

if __name__ == "__main__":
    test_direct_loading()