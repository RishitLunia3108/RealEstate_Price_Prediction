"""
Streamlit Utilities Module
Helper functions for the real estate price prediction Streamlit app
"""

import pandas as pd
import numpy as np
import joblib
import os
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from sklearn.preprocessing import StandardScaler, LabelEncoder


@st.cache_data
def load_data():
    """
    Load and cache the real estate data
    
    Returns:
    --------
    pd.DataFrame
        The loaded real estate dataset
    """
    try:
        # Try to load cleaned data first
        if os.path.exists('data/cleaned_data.csv'):
            data = pd.read_csv('data/cleaned_data.csv')
        elif os.path.exists('data/ahmedabad_real_estate_data.csv'):
            data = pd.read_csv('data/ahmedabad_real_estate_data.csv')
        else:
            # Create sample data if no files exist
            data = create_sample_data()
        
        return data
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return create_sample_data()


@st.cache_resource
def load_models():
    """
    Load and cache all trained models and encoders
    
    Returns:
    --------
    dict
        Dictionary containing model, scaler, and encoders
    """
    try:
        models = {}
        
        # Load main model
        if os.path.exists('models/best_price_prediction_model.pkl'):
            loaded_model = joblib.load('models/best_price_prediction_model.pkl')
            # Validate that it's actually a model with predict method
            if hasattr(loaded_model, 'predict'):
                models['model'] = loaded_model
            else:
                st.warning("⚠️ Loaded model object doesn't have predict method")
        
        # Load scaler
        if os.path.exists('models/feature_scaler.pkl'):
            loaded_scaler = joblib.load('models/feature_scaler.pkl')
            # Validate that it's actually a scaler with transform method
            if hasattr(loaded_scaler, 'transform'):
                models['scaler'] = loaded_scaler
            else:
                st.warning("⚠️ Loaded scaler object doesn't have transform method")
        
        # Load encoders
        if os.path.exists('models/furnishing_encoder.pkl'):
            loaded_encoder = joblib.load('models/furnishing_encoder.pkl')
            if hasattr(loaded_encoder, 'transform') or hasattr(loaded_encoder, 'classes_'):
                models['furnishing_encoder'] = loaded_encoder
        
        if os.path.exists('models/locality_encoder.pkl'):
            loaded_encoder = joblib.load('models/locality_encoder.pkl')
            if hasattr(loaded_encoder, 'transform') or hasattr(loaded_encoder, 'classes_'):
                models['locality_encoder'] = loaded_encoder
        
        if os.path.exists('models/bhk_area_combo_encoder.pkl'):
            loaded_encoder = joblib.load('models/bhk_area_combo_encoder.pkl')
            if hasattr(loaded_encoder, 'transform') or hasattr(loaded_encoder, 'classes_'):
                models['bhk_area_combo_encoder'] = loaded_encoder
        
        # If no valid model found, create dummy models
        if 'model' not in models:
            st.warning("⚠️ No valid trained model found - using dummy model for demo")
            dummy_models = create_dummy_models()
            models.update(dummy_models)
        
        return models
    
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return create_dummy_models()


def create_sample_data():
    """
    Create sample real estate data for demo purposes
    
    Returns:
    --------
    pd.DataFrame
        Sample real estate dataset
    """
    np.random.seed(42)
    
    localities = ['Thaltej', 'Ambli', 'Bopal', 'Shela', 'Science City', 'Vastrapur', 
                 'Sarkhej', 'Prahlad Nagar', 'Satellite', 'Navrangpura']
    
    furnishing_types = ['Furnished', 'Semi-Furnished', 'Unfurnished']
    
    n_samples = 1000
    
    data = []
    for i in range(n_samples):
        bhk = np.random.choice([1, 2, 3, 4, 5], p=[0.1, 0.3, 0.4, 0.15, 0.05])
        area = np.random.normal(bhk * 400 + 200, 150)
        area = max(300, min(area, 5000))  # Constrain area
        
        locality = np.random.choice(localities)
        furnishing = np.random.choice(furnishing_types)
        
        # Base price calculation with some randomness
        base_price_per_sqft = np.random.normal(5000, 1500)
        if locality in ['Thaltej', 'Ambli', 'Bopal']:
            base_price_per_sqft *= 1.5  # Premium localities
        elif locality in ['Science City', 'Vastrapur']:
            base_price_per_sqft *= 1.2  # Mid-premium
        
        price = area * base_price_per_sqft
        if furnishing == 'Furnished':
            price *= 1.1
        elif furnishing == 'Semi-Furnished':
            price *= 1.05
        
        # Add some noise
        price *= np.random.normal(1.0, 0.2)
        price = max(price, 500000)  # Minimum price
        
        data.append({
            'Property Title': f'{bhk} BHK Flat in {locality}',
            'Price': f'₹{price/10000000:.2f} Cr',
            'Area': f'{area:.0f} sqft',
            'BHK': f'{bhk} BHK',
            'Furnishing': furnishing,
            'Locality': locality,
            'Price_Clean': price,
            'Area_SqFt': area,
            'BHK_Num': bhk,
            'Price_Per_SqFt': price / area
        })
    
    return pd.DataFrame(data)


def create_dummy_models():
    """
    Create dummy models for demo purposes
    
    Returns:
    --------
    dict
        Dictionary with dummy models and encoders
    """
    try:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.preprocessing import StandardScaler, LabelEncoder
        
        # Create dummy model
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        
        # Create dummy training data to fit the model
        np.random.seed(42)  # For reproducible results
        X_dummy = np.random.rand(100, 18)  # 18 features
        y_dummy = np.random.rand(100) * 40000000 + 10000000  # Price range: 1-5 Cr
        
        model.fit(X_dummy, y_dummy)
        
        # Create dummy scaler
        scaler = StandardScaler()
        scaler.fit(X_dummy)
        
        # Create dummy encoders
        furnishing_encoder = LabelEncoder()
        furnishing_encoder.fit(['Furnished', 'Semi-Furnished', 'Unfurnished'])
        
        locality_encoder = LabelEncoder()
        localities = ['Thaltej', 'Ambli', 'Bopal', 'Shela', 'Science City', 'Vastrapur']
        locality_encoder.fit(localities)
        
        bhk_area_combo_encoder = LabelEncoder()
        combos = ['1_Small', '2_Medium', '3_Medium', '4_Large']
        bhk_area_combo_encoder.fit(combos)
        
        # Verify models have required methods
        assert hasattr(model, 'predict'), "Dummy model missing predict method"
        assert hasattr(scaler, 'transform'), "Dummy scaler missing transform method"
        
        return {
            'model': model,
            'scaler': scaler,
            'furnishing_encoder': furnishing_encoder,
            'locality_encoder': locality_encoder,
            'bhk_area_combo_encoder': bhk_area_combo_encoder
        }
        
    except Exception as e:
        # Ultra-simple fallback
        class DummyModel:
            def predict(self, X):
                # Simple price calculation based on area and BHK
                if hasattr(X, 'shape') and X.shape[1] >= 2:
                    area = X[:, 0] if len(X.shape) > 1 else X[0]
                    bhk = X[:, 1] if len(X.shape) > 1 else X[1]
                    return area * 5000 + bhk * 2000000  # Simple formula
                else:
                    return np.array([25000000])  # Default 2.5 Cr
        
        class DummyScaler:
            def transform(self, X):
                return X  # No scaling
        
        class DummyEncoder:
            def __init__(self):
                self.classes_ = ['Unknown']
            def transform(self, X):
                return [0] * len(X) if hasattr(X, '__len__') else [0]
        
        return {
            'model': DummyModel(),
            'scaler': DummyScaler(),
            'furnishing_encoder': DummyEncoder(),
            'locality_encoder': DummyEncoder(),
            'bhk_area_combo_encoder': DummyEncoder()
        }


def preprocess_input(area, bhk, furnishing, locality, models):
    """
    Preprocess user input for model prediction using exact training feature engineering
    
    Parameters:
    -----------
    area : float
        Property area in square feet
    bhk : int
        Number of BHK
    furnishing : str
        Furnishing status
    locality : str
        Property locality
    models : dict
        Dictionary containing encoders
        
    Returns:
    --------
    np.array
        Preprocessed feature array with 18 features
    """
    try:
        # The exact feature order from training:
        # ['Area_SqFt', 'BHK_Num', 'Furnishing_Encoded', 'Locality_Encoded', 
        #  'Locality_Tier', 'BHK_Area_Combo_Encoded', 'Area_per_BHK', 'Locality_Area', 
        #  'Locality_BHK', 'Locality_AreaPerBHK', 'Area_Squared', 'BHK_Squared', 
        #  'Area_BHK_Interaction', 'Is_Large_Property', 'Is_Small_Property', 
        #  'Is_Luxury_Config', 'Is_Compact_Config', 'Locality_PropertyCount']
        
        # Initialize feature dictionary for clarity
        feature_dict = {}
        
        # 1. Area_SqFt
        feature_dict['Area_SqFt'] = area
        
        # 2. BHK_Num
        feature_dict['BHK_Num'] = bhk
        
        # 3. Furnishing_Encoded
        try:
            furnishing_encoded = models['furnishing_encoder'].transform([furnishing])[0]
        except:
            # Fallback if locality not in training data
            furnishing_map = {'Furnished': 0, 'Semi-Furnished': 1, 'Unfurnished': 2, 'Unknown': 3}
            furnishing_encoded = furnishing_map.get(furnishing, 2)  # Default to Unfurnished
        feature_dict['Furnishing_Encoded'] = furnishing_encoded
        
        # 4. Locality_Encoded
        try:
            locality_encoded = models['locality_encoder'].transform([locality])[0]
        except:
            # Fallback for unknown localities - use average locality encoding
            locality_encoded = 47  # Approximate middle value for 95 localities
        feature_dict['Locality_Encoded'] = locality_encoded
        
        # 5. Locality_Tier (simplified tier calculation without full price data)
        # Premium tier localities (estimated from common Ahmedabad areas)
        premium_localities = {
            'Ambli', 'Thaltej', 'Bopal', 'Vastrapur', 'Bodakdev', 'Prahlad Nagar',
            'Science City', 'SG Highway', 'Satellite', 'Shela'
        }
        upper_mid_localities = {
            'Chandkheda', 'Gota', 'Nikol', 'Naroda', 'Vastral', 'Bhadaj'
        }
        if locality in premium_localities:
            locality_tier = 1  # Premium
        elif locality in upper_mid_localities:
            locality_tier = 2  # Upper-Mid
        else:
            locality_tier = 3  # Mid/Budget
        feature_dict['Locality_Tier'] = locality_tier
        
        # 6. BHK_Area_Combo_Encoded (using exact logic from training)
        def get_bhk_area_combo(bhk, area):
            if bhk == 1:
                if area < 550:
                    return '1_XSmall'
                elif area < 750:
                    return '1_Small'
                elif area < 950:
                    return '1_Medium'
                else:
                    return '1_Large'
            elif bhk == 2:
                if area < 850:
                    return '2_Small'
                elif area < 1100:
                    return '2_Medium'
                elif area < 1400:
                    return '2_Large'
                else:
                    return '2_XLarge'
            elif bhk == 3:
                if area < 1300:
                    return '3_Small'
                elif area < 1650:
                    return '3_Medium'
                elif area < 2200:
                    return '3_Large'
                else:
                    return '3_XLarge'
            else:  # bhk >= 4
                if area < 2000:
                    return '4+_Small'
                elif area < 2600:
                    return '4+_Medium'
                elif area < 3500:
                    return '4+_Large'
                else:
                    return '4+_XLarge'
        
        bhk_area_combo = get_bhk_area_combo(bhk, area)
        try:
            combo_encoded = models['bhk_area_combo_encoder'].transform([bhk_area_combo])[0]
        except:
            # Fallback encoding if combo not in training data
            combo_encoded = 5  # Middle value
        feature_dict['BHK_Area_Combo_Encoded'] = combo_encoded
        
        # 7. Area_per_BHK
        feature_dict['Area_per_BHK'] = area / bhk
        
        # 8. Locality_Area (interaction)
        feature_dict['Locality_Area'] = locality_encoded * area
        
        # 9. Locality_BHK (interaction)
        feature_dict['Locality_BHK'] = locality_encoded * bhk
        
        # 10. Locality_AreaPerBHK (interaction)
        feature_dict['Locality_AreaPerBHK'] = locality_encoded * (area / bhk)
        
        # 11. Area_Squared
        feature_dict['Area_Squared'] = area ** 2
        
        # 12. BHK_Squared
        feature_dict['BHK_Squared'] = bhk ** 2
        
        # 13. Area_BHK_Interaction
        feature_dict['Area_BHK_Interaction'] = area * bhk
        
        # 14. Is_Large_Property (using training quantiles - roughly 75th percentile)
        feature_dict['Is_Large_Property'] = 1 if area > 1800 else 0
        
        # 15. Is_Small_Property (using training quantiles - roughly 25th percentile)
        feature_dict['Is_Small_Property'] = 1 if area < 900 else 0
        
        # 16. Is_Luxury_Config
        feature_dict['Is_Luxury_Config'] = 1 if bhk >= 4 else 0
        
        # 17. Is_Compact_Config
        feature_dict['Is_Compact_Config'] = 1 if bhk <= 2 else 0
        
        # 18. Locality_PropertyCount (estimated based on locality tier)
        if locality_tier == 1:  # Premium areas have more properties
            property_count = 100
        elif locality_tier == 2:  # Mid-tier areas
            property_count = 50
        else:  # Budget areas have fewer properties in dataset
            property_count = 25
        feature_dict['Locality_PropertyCount'] = property_count
        
        # Convert to array in exact training order
        feature_order = [
            'Area_SqFt', 'BHK_Num', 'Furnishing_Encoded', 'Locality_Encoded',
            'Locality_Tier', 'BHK_Area_Combo_Encoded', 'Area_per_BHK', 'Locality_Area',
            'Locality_BHK', 'Locality_AreaPerBHK', 'Area_Squared', 'BHK_Squared',
            'Area_BHK_Interaction', 'Is_Large_Property', 'Is_Small_Property',
            'Is_Luxury_Config', 'Is_Compact_Config', 'Locality_PropertyCount'
        ]
        
        features = np.array([feature_dict[col] for col in feature_order])
        return features.reshape(1, -1)
        
    except Exception as e:
        st.error(f"Error preprocessing input: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        # Return fallback features with correct shape
        return np.zeros((1, 18))


def get_locality_stats(data, locality):
    """
    Get statistics for a specific locality
    
    Parameters:
    -----------
    data : pd.DataFrame
        Real estate dataset
    locality : str
        Locality name
        
    Returns:
    --------
    dict
        Dictionary with locality statistics
    """
    try:
        locality_data = data[data['Locality'] == locality]
        
        if len(locality_data) == 0:
            return None
        
        return {
            'avg_price': locality_data['Price_Clean'].mean(),
            'min_price': locality_data['Price_Clean'].min(),
            'max_price': locality_data['Price_Clean'].max(),
            'property_count': len(locality_data),
            'avg_area': locality_data['Area_SqFt'].mean(),
            'avg_price_per_sqft': locality_data['Price_Per_SqFt'].mean()
        }
    except:
        return None


def format_price(price):
    """
    Format price in Indian Cr/Lac format
    
    Parameters:
    -----------
    price : float
        Price in rupees
        
    Returns:
    --------
    str
        Formatted price string
    """
    if price >= 10000000:  # 1 Crore
        return f"₹{price/10000000:.2f} Cr"
    elif price >= 100000:  # 1 Lac
        return f"₹{price/100000:.2f} Lac"
    else:
        return f"₹{price:,.0f}"


def create_price_distribution_plot(data):
    """
    Create price distribution histogram
    """
    try:
        fig = px.histogram(
            data, 
            x='Price_Clean',
            nbins=30,
            title='Property Price Distribution',
            labels={'Price_Clean': 'Price (₹)', 'count': 'Number of Properties'}
        )
        
        # Add vertical lines for quartiles
        q25, q50, q75 = data['Price_Clean'].quantile([0.25, 0.5, 0.75])
        fig.add_vline(x=q25, line_dash="dash", line_color="red", 
                      annotation_text=f"Q1: {format_price(q25)}")
        fig.add_vline(x=q50, line_dash="dash", line_color="green", 
                      annotation_text=f"Median: {format_price(q50)}")
        fig.add_vline(x=q75, line_dash="dash", line_color="orange", 
                      annotation_text=f"Q3: {format_price(q75)}")
        
        fig.update_layout(height=400)
        return fig
    except Exception as e:
        # Fallback: simple histogram without quartiles
        fig = px.histogram(
            data, 
            x='Price_Clean',
            title='Property Price Distribution',
            labels={'Price_Clean': 'Price (₹)', 'count': 'Number of Properties'}
        )
        fig.update_layout(height=400)
        return fig


def create_bhk_analysis_plot(data):
    """
    Create BHK configuration analysis plot
    """
    try:
        bhk_stats = data.groupby('BHK_Num').agg({
            'Price_Clean': ['mean', 'count'],
            'Area_SqFt': 'mean'
        }).reset_index()
        
        bhk_stats.columns = ['BHK_Num', 'Avg_Price', 'Count', 'Avg_Area']
        bhk_stats = bhk_stats[bhk_stats['Count'] >= 5]  # Only show BHK with sufficient data
        
        if bhk_stats.empty:
            # Fallback: show all BHK data
            bhk_stats = data.groupby('BHK_Num')['Price_Clean'].mean().reset_index()
            bhk_stats.columns = ['BHK_Num', 'Avg_Price']
        
        fig = px.bar(
            bhk_stats,
            x='BHK_Num',
            y='Avg_Price',
            title='Average Price by BHK Configuration',
            labels={'BHK_Num': 'BHK Configuration', 'Avg_Price': 'Average Price (₹)'}
        )
        
        if 'Count' in bhk_stats.columns:
            fig.update_traces(text=bhk_stats['Count'], texttemplate='%{text} properties', textposition='outside')
        
        fig.update_layout(height=400)
        return fig
    except Exception as e:
        # Simple fallback chart
        bhk_avg = data.groupby('BHK_Num')['Price_Clean'].mean().reset_index()
        fig = px.bar(
            bhk_avg,
            x='BHK_Num',
            y='Price_Clean',
            title='Average Price by BHK Configuration',
            labels={'BHK_Num': 'BHK Configuration', 'Price_Clean': 'Average Price (₹)'}
        )
        fig.update_layout(height=400)
        return fig


def create_locality_comparison_plot(data, localities):
    """
    Create locality comparison plot
    """
    locality_data = data[data['Locality'].isin(localities)]
    
    fig = px.box(
        locality_data,
        x='Locality',
        y='Price_Clean',
        title='Price Distribution by Locality',
        labels={'Locality': 'Locality', 'Price_Clean': 'Price (₹)'}
    )
    
    fig.update_xaxes(tickangle=45)
    fig.update_layout(height=500)
    return fig


def create_model_comparison_plot(model_comparison):
    """
    Create model performance comparison plot
    """
    fig = px.bar(
        model_comparison,
        x='Model',
        y='R² Score',
        title='Model Performance Comparison (R² Score)',
        labels={'R² Score': 'R² Score', 'Model': 'Model'},
        text='R² Score'
    )
    
    fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
    fig.update_xaxes(tickangle=45)
    fig.update_layout(height=400)
    return fig


def create_feature_importance_plot(feature_importance):
    """
    Create feature importance plot
    """
    fig = px.bar(
        feature_importance.sort_values('Importance', ascending=True),
        x='Importance',
        y='Feature',
        orientation='h',
        title='Feature Importance Analysis',
        labels={'Importance': 'Importance Score', 'Feature': 'Feature'}
    )
    
    fig.update_layout(height=500)
    return fig


def validate_input(area, bhk, furnishing, locality):
    """
    Validate user input
    
    Parameters:
    -----------
    area : float
        Property area
    bhk : int
        Number of BHK
    furnishing : str
        Furnishing status
    locality : str
        Property locality
        
    Returns:
    --------
    tuple
        (is_valid, error_message)
    """
    errors = []
    
    # Area validation
    if area < 200 or area > 10000:
        errors.append("Area should be between 200 and 10,000 sq ft")
    
    # BHK validation
    if bhk < 1 or bhk > 6:
        errors.append("BHK should be between 1 and 6")
    
    # Area per BHK validation
    area_per_bhk = area / bhk
    if area_per_bhk < 150:
        errors.append("Area per BHK seems too small (< 150 sq ft per BHK)")
    elif area_per_bhk > 2000:
        errors.append("Area per BHK seems too large (> 2000 sq ft per BHK)")
    
    # Furnishing validation
    valid_furnishing = ['Furnished', 'Semi-Furnished', 'Unfurnished']
    if furnishing not in valid_furnishing:
        errors.append(f"Furnishing must be one of: {', '.join(valid_furnishing)}")
    
    # Locality validation (basic check)
    if not locality or len(locality.strip()) < 2:
        errors.append("Please enter a valid locality name")
    
    if errors:
        return False, "; ".join(errors)
    else:
        return True, "Input validation successful"


def get_prediction_confidence(area, bhk, locality, data):
    """
    Estimate prediction confidence based on similar properties in dataset
    
    Parameters:
    -----------
    area : float
        Property area
    bhk : int
        Number of BHK
    locality : str
        Property locality
    data : pd.DataFrame
        Historical data
        
    Returns:
    --------
    tuple
        (confidence_score, similar_properties_count)
    """
    try:
        # Find similar properties
        similar = data[
            (data['BHK_Num'] == bhk) &
            (data['Area_SqFt'].between(area * 0.8, area * 1.2)) &
            (data['Locality'] == locality)
        ]
        
        similar_count = len(similar)
        
        # Calculate confidence based on similar properties
        if similar_count >= 20:
            confidence = 0.9
        elif similar_count >= 10:
            confidence = 0.8
        elif similar_count >= 5:
            confidence = 0.7
        elif similar_count >= 2:
            confidence = 0.6
        else:
            confidence = 0.5
        
        return confidence, similar_count
    
    except:
        return 0.5, 0  # Default low confidence if error occurs