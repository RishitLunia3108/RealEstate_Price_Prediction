"""
Model Utilities Module
Contains utility functions for model persistence and reporting
"""

import joblib
import pandas as pd


def save_model(model, scaler, le_furn, le_loc, le_combo=None,
               model_path='models/best_price_prediction_model.pkl',
               scaler_path='models/feature_scaler.pkl',
               furn_encoder_path='models/furnishing_encoder.pkl',
               loc_encoder_path='models/locality_encoder.pkl',
               combo_encoder_path='models/bhk_area_combo_encoder.pkl'):
    """
    Save trained model and encoders
    
    Parameters:
    -----------
    model : sklearn model
        Trained model
    scaler : StandardScaler
        Fitted scaler
    le_furn : LabelEncoder
        Furnishing encoder
    le_loc : LabelEncoder
        Locality encoder
    le_combo : LabelEncoder, optional
        BHK-Area combination encoder
    model_path, scaler_path, furn_encoder_path, loc_encoder_path, combo_encoder_path : str
        Paths to save objects
    """
    import os
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Validate objects before saving
    print(f"Validating objects before saving...")
    print(f"  Model type: {type(model)}")
    print(f"  Model has predict: {hasattr(model, 'predict')}")
    print(f"  Scaler type: {type(scaler)}")  
    print(f"  Scaler has transform: {hasattr(scaler, 'transform')}")
    
    if not hasattr(model, 'predict'):
        print(f"WARNING: Model object doesn't have predict method! Type: {type(model)}")
        print(f"Model content preview: {str(model)[:100]}...")
    
    if not hasattr(scaler, 'transform'):
        print(f"WARNING: Scaler object doesn't have transform method! Type: {type(scaler)}")
        print(f"Scaler content preview: {str(scaler)[:100]}...")
    
    try:
        joblib.dump(model, model_path)
        print(f"✓ Model saved to {model_path}")
    except Exception as e:
        print(f"✗ Error saving model: {e}")
        
    try:
        joblib.dump(scaler, scaler_path)
        print(f"✓ Scaler saved to {scaler_path}")
    except Exception as e:
        print(f"✗ Error saving scaler: {e}")
        
    try:
        joblib.dump(le_furn, furn_encoder_path)
        print(f"✓ Furnishing encoder saved to {furn_encoder_path}")
    except Exception as e:
        print(f"✗ Error saving furnishing encoder: {e}")
        
    try:
        joblib.dump(le_loc, loc_encoder_path)
        print(f"✓ Locality encoder saved to {loc_encoder_path}")
    except Exception as e:
        print(f"✗ Error saving locality encoder: {e}")
        
    if le_combo is not None:
        try:
            joblib.dump(le_combo, combo_encoder_path)
            print(f"✓ BHK-Area combo encoder saved to {combo_encoder_path}")
        except Exception as e:
            print(f"✗ Error saving BHK-Area combo encoder: {e}")
        print("Model and encoders (including BHK-Area combo) saving attempted!")
    else:
        print("Model and encoders saving attempted!")
    
    # Verify what was actually saved
    print(f"\nVerifying saved files...")
    for name, path in [("Model", model_path), ("Scaler", scaler_path), 
                       ("Furnishing Encoder", furn_encoder_path), 
                       ("Locality Encoder", loc_encoder_path)]:
        if os.path.exists(path):
            try:
                loaded_obj = joblib.load(path)
                print(f"✓ {name}: File exists, type = {type(loaded_obj)}")
            except Exception as e:
                print(f"✗ {name}: File exists but can't load: {e}")
        else:
            print(f"✗ {name}: File does not exist at {path}")


def load_model(model_path='models/best_price_prediction_model.pkl',
               scaler_path='models/feature_scaler.pkl',
               furn_encoder_path='models/furnishing_encoder.pkl',
               loc_encoder_path='models/locality_encoder.pkl',
               combo_encoder_path='models/bhk_area_combo_encoder.pkl'):
    """
    Load saved model and encoders
    
    Parameters:
    -----------
    model_path, scaler_path, furn_encoder_path, loc_encoder_path, combo_encoder_path : str
        Paths to load objects from
        
    Returns:
    --------
    dict
        Dictionary containing loaded objects
    """
    import os
    
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    le_furn = joblib.load(furn_encoder_path)
    le_loc = joblib.load(loc_encoder_path)
    
    result = {
        'model': model,
        'scaler': scaler,
        'le_furn': le_furn,
        'le_loc': le_loc
    }
    
    # Load combo encoder if it exists
    if os.path.exists(combo_encoder_path):
        le_combo = joblib.load(combo_encoder_path)
        result['le_combo'] = le_combo
    
    return result


def create_comparison_dataframe(results):
    """
    Create comparison dataframe from results
    
    Parameters:
    -----------
    results : dict
        Dictionary of model results
        
    Returns:
    --------
    pd.DataFrame
        Comparison dataframe sorted by R² score
    """
    comparison_df = pd.DataFrame({
        'Model': list(results.keys()),
        'R² Score': [results[m]['R2'] for m in results.keys()],
        'RMSE (Cr)': [results[m]['RMSE']/10000000 for m in results.keys()],
        'MAE (Cr)': [results[m]['MAE']/10000000 for m in results.keys()],
        'CV R² Mean': [results[m]['CV_R2_mean'] for m in results.keys()],
        'CV R² Std': [results[m]['CV_R2_std'] for m in results.keys()]
    })
    
    comparison_df = comparison_df.sort_values('R² Score', ascending=False)
    return comparison_df


def print_model_summary(best_model_name, best_model_info, additional_metrics, 
                       total_properties, train_size, test_size):
    """
    Print comprehensive model performance summary
    
    Parameters:
    -----------
    best_model_name : str
        Name of best model
    best_model_info : dict
        Best model information
    additional_metrics : dict
        Additional metrics
    total_properties : int
        Total number of properties
    train_size : int
        Training set size
    test_size : int
        Test set size
    """
    print("="*80)
    print(f"MODEL PERFORMANCE SUMMARY - {best_model_name}")
    print("="*80)
    print(f"\nDataset:")
    print(f"  Total Properties: {total_properties:,}")
    print(f"  Training Set: {train_size:,}")
    print(f"  Test Set: {test_size:,}")
    
    print(f"\nModel Performance:")
    print(f"  R² Score: {best_model_info['R2']:.4f} (Explains {best_model_info['R2']*100:.2f}% of variance)")
    print(f"  RMSE: Rs.{best_model_info['RMSE']/10000000:.4f} Cr")
    print(f"  MAE: Rs.{best_model_info['MAE']/10000000:.4f} Cr")
    print(f"  MAPE: {additional_metrics['MAPE']:.2f}%")
    print(f"  Cross-Validation R²: {best_model_info['CV_R2_mean']:.4f} (+/- {best_model_info['CV_R2_std']*2:.4f})")
    
    print(f"\nPrediction Accuracy:")
    print(f"  Predictions within ±10%: {additional_metrics['Accuracy_10pct']:.2f}%")
    print(f"  Predictions within ±20%: {additional_metrics['Accuracy_20pct']:.2f}%")
    
    print(f"\nModel Interpretation:")
    if best_model_info['R2'] > 0.9:
        print(f"  ✓ Excellent model performance (R² > 0.9)")
    elif best_model_info['R2'] > 0.7:
        print(f"  ✓ Good model performance (R² > 0.7)")
    else:
        print(f"  ⚠ Moderate model performance (R² < 0.7)")
    
    print(f"  Average prediction error: Rs.{best_model_info['MAE']/10000000:.4f} Cr")
    print("="*80)


def print_train_test_summary(train_test_comparison):
    """
    Print training vs test performance comparison
    
    Parameters:
    -----------
    train_test_comparison : list
        List of comparison dictionaries
    """
    print("="*100)
    print("TRAINING vs TEST PERFORMANCE COMPARISON")
    print("="*100)
    
    for comparison in train_test_comparison:
        print(f"\n{comparison['Model']}:")
        print(f"  Training R²: {comparison['Train R²']:.4f}")
        print(f"  Test R²: {comparison['Test R²']:.4f}")
        diff = comparison['R² Difference']
        status = '(Overfitting)' if diff > 0.05 else '(Good)'
        print(f"  Difference: {diff:.4f} {status}")
        print(f"  Training Accuracy (±10%): {comparison['Train Accuracy (±10%)']:.2f}%")
        print(f"  Test Accuracy (±10%): {comparison['Test Accuracy (±10%)']:.2f}%")
    
    # Create DataFrame
    train_test_df = pd.DataFrame(train_test_comparison)
    train_test_df = train_test_df.sort_values('Test R²', ascending=False)
    
    print("\n" + "="*100)
    print("\nSUMMARY TABLE:")
    print(train_test_df.to_string(index=False))
    print("="*100)
    
    return train_test_df


def create_sample_predictions(y_test, predictions, n_samples=15):
    """
    Create sample predictions dataframe
    
    Parameters:
    -----------
    y_test : array-like
        True values
    predictions : array-like
        Predicted values
    n_samples : int
        Number of samples to display
        
    Returns:
    --------
    pd.DataFrame
        Sample predictions dataframe
    """
    sample_predictions = pd.DataFrame({
        'Actual Price (Cr)': y_test.iloc[:n_samples].values / 10000000,
        'Predicted Price (Cr)': predictions[:n_samples] / 10000000,
    })
    sample_predictions['Difference (Cr)'] = sample_predictions['Actual Price (Cr)'] - sample_predictions['Predicted Price (Cr)']
    sample_predictions['Difference (%)'] = (sample_predictions['Difference (Cr)'] / sample_predictions['Actual Price (Cr)']) * 100
    sample_predictions['Accuracy (%)'] = 100 - abs(sample_predictions['Difference (%)'])
    
    return sample_predictions


def print_insights(best_model_info, feature_cols, accuracy_within_20_pct, total_properties):
    """
    Print key insights and recommendations
    
    Parameters:
    -----------
    best_model_info : dict
        Best model information
    feature_cols : list
        List of feature names
    accuracy_within_20_pct : float
        Accuracy within 20%
    total_properties : int
        Total number of properties
    """
    print("KEY INSIGHTS:")
    print("="*80)
    
    if hasattr(best_model_info['model'], 'feature_importances_'):
        importances = best_model_info['model'].feature_importances_
        feature_importance_df = pd.DataFrame({
            'Feature': feature_cols,
            'Importance': importances
        }).sort_values('Importance', ascending=False)
        
        print("\n1. Most Important Features for Price Prediction:")
        for idx, row in feature_importance_df.head(3).iterrows():
            print(f"   {row['Feature']}: {row['Importance']*100:.2f}% importance")
    
    print(f"\n2. Model Reliability:")
    print(f"   The model can predict prices with an average error of Rs.{best_model_info['MAE']/10000000:.4f} Cr")
    print(f"   {accuracy_within_20_pct:.1f}% of predictions are within ±20% of actual price")
    
    print(f"\n3. Best Use Cases:")
    print(f"   - Property valuation and price estimation")
    print(f"   - Identifying overpriced or underpriced properties")
    print(f"   - Market trend analysis")
    print(f"   - Investment decision support")
    
    print("\n4. Limitations:")
    print(f"   - Model trained on {total_properties} properties from Ahmedabad")
    print(f"   - Extreme outliers (top 1%) were removed from training")
    print(f"   - Predictions most accurate for properties in the ±20% range")
    
    print("="*80)
