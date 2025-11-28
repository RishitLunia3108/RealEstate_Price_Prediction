"""
Data Cleaning Module
Contains functions for cleaning scraped real estate data
"""

import pandas as pd
import numpy as np
import re


def clean_price(price_str):
    """
    Clean and convert price string to numeric value
    
    Parameters:
    -----------
    price_str : str
        Price string (e.g., '₹1.5 Cr', '₹45 Lac')
        
    Returns:
    --------
    float
        Numeric price value
    """
    if pd.isna(price_str) or price_str == 'N/A':
        return np.nan
    
    price_str = str(price_str).replace('₹', '').replace(',', '').strip()
    
    if 'Cr' in price_str:
        return float(re.search(r'[\d\.]+', price_str).group()) * 10000000
    elif 'Lac' in price_str or 'Lakh' in price_str:
        return float(re.search(r'[\d\.]+', price_str).group()) * 100000
    elif 'K' in price_str:
        return float(re.search(r'[\d\.]+', price_str).group()) * 1000
    else:
        try:
            return float(price_str)
        except:
            return np.nan


def clean_area(area_str):
    """
    Clean and convert area string to square feet
    
    Parameters:
    -----------
    area_str : str
        Area string (e.g., '1200 sqft', '150 sqyrd')
        
    Returns:
    --------
    float
        Area in square feet
    """
    if pd.isna(area_str) or area_str == 'N/A':
        return np.nan
    
    area_str = str(area_str).lower().replace(',', '')
    
    # Extract numeric part
    match = re.search(r'[\d\.]+', area_str)
    if not match:
        return np.nan
        
    value = float(match.group())
    
    if 'sqyrd' in area_str or 'sq.yrd' in area_str:
        return value * 9  # 1 sqyrd = 9 sqft
    elif 'sqm' in area_str or 'sq.m' in area_str:
        return value * 10.764  # 1 sqm = 10.764 sqft
    elif 'ground' in area_str:
        return value * 2400  # Approx 1 ground = 2400 sqft
    else:
        return value  # Assume sqft


def clean_bhk(bhk_str):
    """
    Extract BHK number from string
    
    Parameters:
    -----------
    bhk_str : str
        BHK string (e.g., '3 BHK', '2BHK')
        
    Returns:
    --------
    int
        Number of bedrooms
    """
    if pd.isna(bhk_str) or bhk_str == 'N/A':
        return np.nan
    match = re.search(r'\d+', str(bhk_str))
    if match:
        return int(match.group())
    return np.nan


def extract_main_locality(locality_str):
    """
    Extract main locality name from full locality string
    Removes property/society name and city name
    
    Parameters:
    -----------
    locality_str : str
        Full locality string (e.g., 'Aristo Anantam, Chharodi, Ahmedabad')
        
    Returns:
    --------
    str
        Main locality name (e.g., 'Chharodi')
    """
    if pd.isna(locality_str) or locality_str == 'N/A':
        return 'Ahmedabad'
    
    locality_str = str(locality_str).strip()
    
    # Split by comma
    parts = [p.strip() for p in locality_str.split(',')]
    
    # Remove 'Ahmedabad' if present
    parts = [p for p in parts if p.lower() != 'ahmedabad']
    
    # If we have multiple parts, take the second-to-last (usually the main locality)
    # Example: "Aristo Anantam, Chharodi, Ahmedabad" -> ["Aristo Anantam", "Chharodi"] -> "Chharodi"
    if len(parts) >= 2:
        return parts[-1]  # Take the last part after removing Ahmedabad
    elif len(parts) == 1:
        return parts[0]
    else:
        return 'Ahmedabad'


def clean_scraped_data(input_file='data/ahmedabad_real_estate_data.csv', 
                       output_file='data/cleaned_data.csv'):
    """
    Complete data cleaning pipeline for scraped data
    
    Parameters:
    -----------
    input_file : str
        Path to raw scraped CSV file
    output_file : str
        Path to save cleaned CSV file
        
    Returns:
    --------
    pd.DataFrame
        Cleaned dataframe
    """
    print("\n" + "="*80)
    print("DATA CLEANING PIPELINE")
    print("="*80)
    
    # Load data
    print(f"\n1. Loading data from {input_file}...")
    df = pd.read_csv(input_file)
    print(f"   Initial Shape: {df.shape}")
    print(f"   Source Distribution:")
    print(df['Source'].value_counts())
    
    # Remove duplicates
    print(f"\n2. Removing duplicates...")
    initial_count = len(df)
    df.drop_duplicates(inplace=True)
    print(f"   Removed: {initial_count - len(df)} duplicates")
    print(f"   Shape: {df.shape}")
    
    # Clean price
    print(f"\n3. Cleaning price data...")
    df['Price_Clean'] = df['Price'].apply(clean_price)
    before_price = len(df)
    df.dropna(subset=['Price_Clean'], inplace=True)
    print(f"   Removed: {before_price - len(df)} rows with invalid prices")
    print(f"   Price range: ₹{df['Price_Clean'].min()/10000000:.2f} Cr to ₹{df['Price_Clean'].max()/10000000:.2f} Cr")
    
    # Clean area
    print(f"\n4. Cleaning area data...")
    df['Area_SqFt'] = df['Area'].apply(clean_area)
    before_area = len(df)
    df.dropna(subset=['Area_SqFt'], inplace=True)
    print(f"   Removed: {before_area - len(df)} rows with invalid areas")
    print(f"   Area range: {df['Area_SqFt'].min():.0f} to {df['Area_SqFt'].max():.0f} sq.ft")
    
    # Clean BHK
    print(f"\n5. Cleaning BHK data...")
    df['BHK_Num'] = df['BHK'].apply(clean_bhk)
    before_bhk = len(df)
    df.dropna(subset=['BHK_Num'], inplace=True)
    print(f"   Removed: {before_bhk - len(df)} rows with invalid BHK")
    print(f"   BHK distribution:")
    print(df['BHK_Num'].value_counts().sort_index())
    
    # Calculate Price per SqFt
    print(f"\n6. Calculating price per square foot...")
    df['Price_Per_SqFt'] = df['Price_Clean'] / df['Area_SqFt']
    print(f"   Price/SqFt range: ₹{df['Price_Per_SqFt'].min():.0f} to ₹{df['Price_Per_SqFt'].max():.0f}")
    
    # Handle locality
    print(f"\n7. Processing locality data...")
    df['Locality'] = df['Locality'].apply(extract_main_locality)
    print(f"   Unique localities: {df['Locality'].nunique()}")
    print(f"   Top 10 localities:")
    print(df['Locality'].value_counts().head(10))
    
    # Handle furnishing
    print(f"\n8. Processing furnishing data...")
    df['Furnishing'] = df['Furnishing'].fillna('Unknown')
    print(f"   Furnishing distribution:")
    print(df['Furnishing'].value_counts())
    
    # Save cleaned data
    print(f"\n9. Saving cleaned data to {output_file}...")
    df.to_csv(output_file, index=False)
    
    print("\n" + "="*80)
    print("DATA CLEANING COMPLETE!")
    print("="*80)
    print(f"Final Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Output: {output_file}")
    print("="*80 + "\n")
    
    return df


if __name__ == "__main__":
    clean_scraped_data()
