"""
Data Preprocessing Module
Contains functions for loading and preprocessing real estate data
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


def load_data(filepath='../cleaned_data.csv'):
    """
    Load real estate data from CSV file
    
    Parameters:
    -----------
    filepath : str
        Path to the CSV file
        
    Returns:
    --------
    pd.DataFrame
        Loaded dataframe
    """
    df = pd.read_csv(filepath)
    print(f"Dataset Shape: {df.shape}")
    return df


def remove_outliers(df, percentile=0.99):
    """
    Remove extreme outliers from the dataset
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    percentile : float
        Percentile threshold for outlier removal
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with outliers removed
    """
    df_clean = df.copy()
    
    # Remove extreme outliers (top 1%)
    price_threshold = df_clean['Price_Clean'].quantile(percentile)
    area_threshold = df_clean['Area_SqFt'].quantile(percentile)
    df_clean = df_clean[(df_clean['Price_Clean'] <= price_threshold) & 
                        (df_clean['Area_SqFt'] <= area_threshold)]
    
    print(f"Original size: {len(df)}")
    print(f"After removing top {(1-percentile)*100:.0f}% outliers: {len(df_clean)}")
    print(f"Properties removed: {len(df) - len(df_clean)}")
    
    return df_clean


def encode_furnishing(df):
    """
    Encode furnishing status using LabelEncoder
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
        
    Returns:
    --------
    pd.DataFrame, LabelEncoder
        Dataframe with encoded furnishing and the encoder object
    """
    df_encoded = df.copy()
    le_furn = LabelEncoder()
    df_encoded['Furnishing_Encoded'] = le_furn.fit_transform(df_encoded['Furnishing'].fillna('Unknown'))
    
    print("Furnishing Encoding:")
    for i, label in enumerate(le_furn.classes_):
        print(f"  {label}: {i}")
    
    return df_encoded, le_furn


def encode_locality(df, min_properties=5):
    """
    Encode locality using Label Encoding (NO target leakage)
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    min_properties : int
        Minimum number of properties to keep location separate
        
    Returns:
    --------
    pd.DataFrame, LabelEncoder
        Dataframe with encoded locality and the encoder object
    """
    df_encoded = df.copy()
    
    # Group rare locations as 'Other'
    location_counts = df_encoded['Locality'].value_counts()
    top_locations = location_counts[location_counts >= min_properties].index
    df_encoded['Locality_Grouped'] = df_encoded['Locality'].apply(
        lambda x: x if x in top_locations else 'Other'
    )
    
    print(f"Unique locations (original): {df_encoded['Locality'].nunique()}")
    print(f"Grouped locations (>={min_properties} properties): {len(top_locations)}")
    print(f"Properties in 'Other' category: {(df_encoded['Locality_Grouped'] == 'Other').sum()}")
    
    # Label Encoding: Simple categorical encoding (NO price leakage)
    le_loc = LabelEncoder()
    df_encoded['Locality_Encoded'] = le_loc.fit_transform(df_encoded['Locality_Grouped'])
    
    print(f"Label encoding complete: {len(le_loc.classes_)} unique locations")
    
    return df_encoded, le_loc


def create_locality_tier(df):
    """
    Create locality tier feature based on average property prices
    Tier 1 (Premium): Top 25% most expensive localities
    Tier 2 (Upper-Mid): 50-75th percentile
    Tier 3 (Mid): 25-50th percentile  
    Tier 4 (Budget): Bottom 25% localities
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with Locality column
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with Locality_Tier feature
    """
    df_tiered = df.copy()
    
    # Calculate mean price per locality (using training data pattern)
    locality_avg_price = df_tiered.groupby('Locality')['Price_Clean'].mean()
    
    # Calculate quartiles
    q25 = locality_avg_price.quantile(0.25)
    q50 = locality_avg_price.quantile(0.50)
    q75 = locality_avg_price.quantile(0.75)
    
    # Create tier mapping
    def assign_tier(locality):
        avg_price = locality_avg_price.get(locality, q50)  # Default to median
        if avg_price >= q75:
            return 1  # Premium (Tier 1)
        elif avg_price >= q50:
            return 2  # Upper-Mid (Tier 2)
        elif avg_price >= q25:
            return 3  # Mid (Tier 3)
        else:
            return 4  # Budget (Tier 4)
    
    df_tiered['Locality_Tier'] = df_tiered['Locality'].apply(assign_tier)
    
    print("\nLocality Tier Distribution:")
    print(f"  Tier 1 (Premium, ≥₹{q75/10000000:.2f}Cr avg): {(df_tiered['Locality_Tier']==1).sum()} properties")
    print(f"  Tier 2 (Upper-Mid, ₹{q50/10000000:.2f}-{q75/10000000:.2f}Cr): {(df_tiered['Locality_Tier']==2).sum()} properties")
    print(f"  Tier 3 (Mid, ₹{q25/10000000:.2f}-{q50/10000000:.2f}Cr): {(df_tiered['Locality_Tier']==3).sum()} properties")
    print(f"  Tier 4 (Budget, <₹{q25/10000000:.2f}Cr avg): {(df_tiered['Locality_Tier']==4).sum()} properties")
    
    return df_tiered


def create_bhk_area_combo(df):
    """
    Create BHK-Area combination feature to capture market segmentation
    
    Detects configurations like:
    - 1_Small: Studio/1BHK in compact space
    - 2_Small: Compact 2BHK
    - 2_Medium: Standard 2BHK
    - 3_Medium: Standard 3BHK
    - 3_Large: Spacious 3BHK
    - 4_Large: Luxury 4BHK
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with BHK_Num and Area_SqFt
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with BHK_Area_Combo feature
    """
    df_combo = df.copy()
    
    # Define area size categories per BHK
    def categorize_area_size(row):
        bhk = row['BHK_Num']
        area = row['Area_SqFt']
        
        # Area thresholds per BHK (based on market patterns)
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
        
        elif bhk >= 4:
            if area < 2000:
                return '4+_Small'
            elif area < 2600:
                return '4+_Medium'
            elif area < 3500:
                return '4+_Large'
            else:
                return '4+_XLarge'
        
        else:  # Fallback for any unexpected values
            return 'Unknown'
    
    df_combo['BHK_Area_Combo'] = df_combo.apply(categorize_area_size, axis=1)
    
    # Encode the combination
    le_combo = LabelEncoder()
    df_combo['BHK_Area_Combo_Encoded'] = le_combo.fit_transform(df_combo['BHK_Area_Combo'])
    
    print("\nBHK-Area Combination Feature Created:")
    combo_dist = df_combo['BHK_Area_Combo'].value_counts().head(10)
    for combo, count in combo_dist.items():
        print(f"  {combo}: {count} properties")
    print(f"  Total unique combinations: {df_combo['BHK_Area_Combo'].nunique()}")
    
    return df_combo, le_combo


def create_features(df):
    """
    Create additional engineered features
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with new features
    """
    df_featured = df.copy()
    
    # Basic features
    df_featured['Area_per_BHK'] = df_featured['Area_SqFt'] / df_featured['BHK_Num']
    df_featured['Area_per_BHK'] = df_featured['Area_per_BHK'].replace([np.inf, -np.inf], 0)
    
    # Locality interaction features (NO Price_Per_SqFt - data leakage!)
    df_featured['Locality_Area'] = df_featured['Locality_Encoded'] * df_featured['Area_SqFt']
    df_featured['Locality_BHK'] = df_featured['Locality_Encoded'] * df_featured['BHK_Num']
    df_featured['Locality_AreaPerBHK'] = df_featured['Locality_Encoded'] * df_featured['Area_per_BHK']
    
    # Polynomial features (Area and BHK interactions)
    df_featured['Area_Squared'] = df_featured['Area_SqFt'] ** 2
    df_featured['BHK_Squared'] = df_featured['BHK_Num'] ** 2
    df_featured['Area_BHK_Interaction'] = df_featured['Area_SqFt'] * df_featured['BHK_Num']
    
    # Area categories
    df_featured['Is_Large_Property'] = (df_featured['Area_SqFt'] > df_featured['Area_SqFt'].quantile(0.75)).astype(int)
    df_featured['Is_Small_Property'] = (df_featured['Area_SqFt'] < df_featured['Area_SqFt'].quantile(0.25)).astype(int)
    
    # BHK categories
    df_featured['Is_Luxury_Config'] = (df_featured['BHK_Num'] >= 4).astype(int)
    df_featured['Is_Compact_Config'] = (df_featured['BHK_Num'] <= 2).astype(int)
    
    # Add locality property count (NO price leakage - just count)
    locality_counts = df_featured.groupby('Locality_Grouped').size().reset_index(name='Locality_PropertyCount')
    df_featured = df_featured.merge(locality_counts, on='Locality_Grouped', how='left')
    
    # Create locality tier feature
    df_featured = create_locality_tier(df_featured)
    
    # Create BHK-Area combination feature
    df_featured, le_combo = create_bhk_area_combo(df_featured)
    
    print("\nNew Features Created:")
    print(f"  Area per BHK - Mean: {df_featured['Area_per_BHK'].mean():.2f} sq.ft")
    print(f"  Polynomial features: Area², BHK², Area×BHK")
    print(f"  Locality property count (no price leakage)")
    print(f"  Locality tier (4 tiers based on locality avg prices)")
    print(f"  BHK-Area combo (market segmentation)")
    print(f"  Property categories: Large/Small property, Luxury/Compact config")
    print(f"  Locality interactions: 3 features")
    
    return df_featured, le_combo


def prepare_features(df, feature_cols):
    """
    Prepare feature matrix and target variable
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    feature_cols : list
        List of feature column names
        
    Returns:
    --------
    X, y : pd.DataFrame, pd.Series
        Features and target variable
    """
    X = df[feature_cols].fillna(0)
    y = df['Price_Clean']
    
    print(f"\nFeatures: {feature_cols}")
    print(f"Target: Price_Clean")
    print(f"Dataset shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    
    return X, y


def preprocess_pipeline(filepath='data/cleaned_data.csv', percentile=0.99, min_properties=5):
    """
    Complete preprocessing pipeline
    
    Parameters:
    -----------
    filepath : str
        Path to the CSV file
    percentile : float
        Percentile threshold for outlier removal
    min_properties : int
        Minimum number of properties to keep location separate
        
    Returns:
    --------
    dict
        Dictionary containing processed data and encoders
    """
    # Load data
    df = load_data(filepath)
    
    # Store original before outlier removal (for visualization comparison)
    df_before_outliers = df.copy()
    
    # Remove outliers
    df = remove_outliers(df, percentile)
    
    # Encode furnishing
    df, le_furn = encode_furnishing(df)
    
    # Encode locality
    df, le_loc = encode_locality(df, min_properties)
    
    # Create features
    df, le_combo = create_features(df)
    
    # Define feature columns (including BHK-Area combo)
    feature_cols = ['Area_SqFt', 'BHK_Num', 'Furnishing_Encoded', 
                    'Locality_Encoded', 'Locality_Tier', 'BHK_Area_Combo_Encoded',
                    'Area_per_BHK', 'Locality_Area', 
                    'Locality_BHK', 'Locality_AreaPerBHK',
                    'Area_Squared', 'BHK_Squared', 'Area_BHK_Interaction',
                    'Is_Large_Property', 'Is_Small_Property', 
                    'Is_Luxury_Config', 'Is_Compact_Config', 'Locality_PropertyCount']
    
    # Prepare features
    X, y = prepare_features(df, feature_cols)
    
    return {
        'X': X,
        'y': y,
        'df': df,
        'df_before_outliers': df_before_outliers,
        'feature_cols': feature_cols,
        'le_furn': le_furn,
        'le_loc': le_loc,
        'le_combo': le_combo
    }
