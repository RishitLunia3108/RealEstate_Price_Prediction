"""
Model Diagnostic Tool
Check what's actually stored in the model pickle files
"""

import joblib
import os

def diagnose_models():
    """Diagnose the model files to understand what's stored"""
    print("🔍 DIAGNOSING MODEL FILES")
    print("=" * 50)
    
    model_files = {
        'Main Model': 'models/best_price_prediction_model.pkl',
        'Feature Scaler': 'models/feature_scaler.pkl',
        'Furnishing Encoder': 'models/furnishing_encoder.pkl',
        'Locality Encoder': 'models/locality_encoder.pkl',
        'BHK Area Combo Encoder': 'models/bhk_area_combo_encoder.pkl'
    }
    
    for name, filepath in model_files.items():
        print(f"\n📁 {name}: {filepath}")
        
        if not os.path.exists(filepath):
            print("   ❌ File does not exist")
            continue
            
        try:
            obj = joblib.load(filepath)
            
            print(f"   ✅ File loaded successfully")
            print(f"   📊 Type: {type(obj)}")
            
            # Check for common methods
            methods_to_check = ['predict', 'transform', 'fit', 'classes_']
            available_methods = []
            for method in methods_to_check:
                if hasattr(obj, method):
                    available_methods.append(method)
            
            if available_methods:
                print(f"   🔧 Available methods: {', '.join(available_methods)}")
            else:
                print("   ⚠️  No standard ML methods found")
            
            # Special checks
            if name == 'Main Model':
                if hasattr(obj, 'predict'):
                    print("   ✅ Model has predict method - GOOD")
                else:
                    print("   ❌ Model missing predict method - PROBLEM")
                    if hasattr(obj, 'shape'):
                        print(f"   📐 Shape: {obj.shape} (appears to be array)")
            
            elif 'Scaler' in name:
                if hasattr(obj, 'transform'):
                    print("   ✅ Scaler has transform method - GOOD")
                else:
                    print("   ❌ Scaler missing transform method - PROBLEM")
                    if hasattr(obj, 'shape'):
                        print(f"   📐 Shape: {obj.shape} (appears to be array)")
            
            elif 'Encoder' in name:
                if hasattr(obj, 'transform') or hasattr(obj, 'classes_'):
                    print("   ✅ Encoder looks valid - GOOD")
                else:
                    print("   ❌ Encoder missing expected methods - PROBLEM")
                    
        except Exception as e:
            print(f"   ❌ Error loading file: {e}")
    
    print("\n" + "=" * 50)
    print("💡 RECOMMENDATIONS:")
    print("If models appear to be arrays instead of ML objects:")
    print("1. Run main.py to retrain and save models properly")
    print("2. Check model_utils.py save_model() function")
    print("3. The Streamlit app will use dummy models as fallback")

if __name__ == "__main__":
    diagnose_models()