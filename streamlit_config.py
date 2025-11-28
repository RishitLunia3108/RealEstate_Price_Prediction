"""
Streamlit App Configuration
Configuration settings for the real estate price prediction app
"""

# App Configuration
APP_CONFIG = {
    "title": "Ahmedabad Real Estate Price Predictor",
    "icon": "🏠",
    "layout": "wide",
    "sidebar_state": "expanded"
}

# Model Configuration
MODEL_CONFIG = {
    "model_path": "models/best_price_prediction_model.pkl",
    "scaler_path": "models/feature_scaler.pkl",
    "furnishing_encoder_path": "models/furnishing_encoder.pkl",
    "locality_encoder_path": "models/locality_encoder.pkl",
    "bhk_area_combo_encoder_path": "models/bhk_area_combo_encoder.pkl"
}

# Data Configuration
DATA_CONFIG = {
    "cleaned_data_path": "data/cleaned_data.csv",
    "raw_data_path": "data/ahmedabad_real_estate_data.csv",
    "sample_size": 1000
}

# Feature Configuration
FEATURE_CONFIG = {
    "feature_names": [
        'Area_SqFt', 'BHK_Num', 'Furnishing_Encoded', 
        'Locality_Encoded', 'Locality_Tier', 'BHK_Area_Combo_Encoded',
        'Area_per_BHK', 'Locality_Area', 
        'Locality_BHK', 'Locality_AreaPerBHK',
        'Area_Squared', 'BHK_Squared', 'Area_BHK_Interaction',
        'Is_Large_Property', 'Is_Small_Property', 
        'Is_Luxury_Config', 'Is_Compact_Config', 'Locality_PropertyCount'
    ],
    "n_features": 18
}

# Input Validation Rules
VALIDATION_RULES = {
    "area": {
        "min": 200,
        "max": 10000,
        "default": 1200
    },
    "bhk": {
        "min": 1,
        "max": 6,
        "options": [1, 2, 3, 4, 5, 6],
        "default": 3
    },
    "furnishing": {
        "options": ['Furnished', 'Semi-Furnished', 'Unfurnished'],
        "default": 'Unfurnished'
    },
    "area_per_bhk": {
        "min": 150,
        "max": 2000
    }
}

# Locality Configuration
LOCALITY_CONFIG = {
    "premium_localities": ['Thaltej', 'Ambli', 'Bopal', 'Vastrapur', 'Satellite'],
    "mid_localities": ['Science City', 'Prahlad Nagar', 'Shela'],
    "budget_localities": ['Vastral', 'Narol', 'Odhav'],
    "default_locality": "Thaltej"
}

# UI Configuration
UI_CONFIG = {
    "colors": {
        "primary": "#1f77b4",
        "success": "#28a745",
        "warning": "#ffc107",
        "danger": "#dc3545",
        "info": "#17a2b8"
    },
    "chart_height": 400,
    "table_height": 300
}

# Price Configuration
PRICE_CONFIG = {
    "currency": "₹",
    "cr_threshold": 10000000,  # 1 Crore
    "lac_threshold": 100000,   # 1 Lac
    "price_segments": {
        "Budget": (0, 10000000),      # < 1 Cr
        "Mid-Range": (10000000, 25000000),  # 1-2.5 Cr
        "Premium": (25000000, 50000000),    # 2.5-5 Cr
        "Luxury": (50000000, float('inf'))  # > 5 Cr
    }
}

# Model Performance Metrics (Sample - Replace with actual values)
MODEL_METRICS = {
    "best_model": "LightGBM",
    "r2_score": 0.891,
    "rmse_cr": 0.94,
    "mae_cr": 0.61,
    "accuracy_10pct": 65.2,
    "accuracy_20pct": 78.5
}

# Navigation Configuration
NAVIGATION_CONFIG = {
    "pages": {
        "🔮 Price Prediction": {
            "description": "Predict property prices using ML models",
            "icon": "🔮"
        },
        "📊 Data Dashboard": {
            "description": "Explore market trends and data visualizations",
            "icon": "📊"
        },
        "🏆 Model Performance": {
            "description": "View model accuracy and performance metrics",
            "icon": "🏆"
        },
        "📈 Market Insights": {
            "description": "Get market analysis and investment insights",
            "icon": "📈"
        },
        "ℹ️ About": {
            "description": "Learn about the project and methodology",
            "icon": "ℹ️"
        }
    }
}

# Cache Configuration
CACHE_CONFIG = {
    "data_ttl": 3600,  # 1 hour
    "model_persist": True,
    "clear_cache_on_error": True
}