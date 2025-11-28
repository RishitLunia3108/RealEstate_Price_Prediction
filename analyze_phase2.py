import pandas as pd
import numpy as np
from collections import Counter

# Load data
df = pd.read_csv('data/enriched_with_summaries.csv')
df_loc = pd.read_csv('data/locality_summaries.csv')

print("=" * 80)
print("PHASE 2 NLP INSIGHTS - COMPREHENSIVE ANALYSIS")
print("=" * 80)

# Dataset Info
print("\n=== DATASET OVERVIEW ===")
print(f"Total Properties: {len(df)}")
print(f"Total Features: {len(df.columns)}")

# NLP Features Coverage
print("\n=== NLP FEATURES COVERAGE ===")
nlp_cols = [
    'Amenities', 'Proximity_Info', 'Selling_Points',
    'Completeness_Score', 'Clarity_Score', 'Amenities_Score', 
    'Attractiveness_Score', 'Overall_Quality_Score',
    'Clean_Summary', 'Marketing_Description', 'Investor_Summary'
]
for col in nlp_cols:
    if col in df.columns:
        non_null = df[col].notna().sum()
        print(f"{col}: {non_null} ({non_null/len(df)*100:.1f}%)")

# Quality Scores Statistics
print("\n=== QUALITY SCORES STATISTICS ===")
quality_cols = ['Completeness_Score', 'Clarity_Score', 'Amenities_Score', 
                'Attractiveness_Score', 'Overall_Quality_Score']
for col in quality_cols:
    if col in df.columns:
        mean_val = df[col].mean()
        std_val = df[col].std()
        min_val = df[col].min()
        max_val = df[col].max()
        print(f"{col}:")
        print(f"  Mean: {mean_val:.2f}, Std: {std_val:.2f}, Min: {min_val:.2f}, Max: {max_val:.2f}")

# Summary Types
print("\n=== SUMMARY GENERATION ===")
if 'Clean_Summary' in df.columns:
    placeholder_count = df['Clean_Summary'].str.contains('property in', case=False, na=False).sum()
    gemini_count = len(df) - placeholder_count
    print(f"Gemini API Summaries: {gemini_count}")
    print(f"Placeholder Summaries: {placeholder_count}")
    print(f"Total Summaries: {len(df)}")

# Amenity Analysis
print("\n=== AMENITY EXTRACTION ANALYSIS ===")
all_amenities = []
for amenities_str in df['Amenities'].dropna():
    amenities = [a.strip() for a in str(amenities_str).split(',')]
    all_amenities.extend(amenities)

amenity_counts = Counter(all_amenities)
print(f"Total Unique Amenities: {len(amenity_counts)}")
print(f"Total Amenity Mentions: {sum(amenity_counts.values())}")
print(f"Avg Amenities per Property: {sum(amenity_counts.values())/len(df):.1f}")

print("\n=== TOP 15 AMENITIES ===")
for amenity, count in amenity_counts.most_common(15):
    print(f"{amenity}: {count} properties ({count/len(df)*100:.1f}%)")

# Locality Analysis
print("\n" + "=" * 80)
print("=== LOCALITY ANALYSIS ===")
print("=" * 80)
print(f"Total Localities Analyzed: {len(df_loc)}")
print(f"Total Properties Covered: {df_loc['property_count'].sum()}")
print(f"Avg Properties per Locality: {df_loc['property_count'].mean():.1f}")
print(f"Max Properties in Locality: {df_loc['property_count'].max()}")
print(f"Min Properties in Locality: {df_loc['property_count'].min()}")

print("\n=== TOP 10 LOCALITIES BY PROPERTY COUNT ===")
top10 = df_loc.nlargest(10, 'property_count')[['locality_name', 'property_count', 'avg_price', 'avg_area']]
for idx, row in top10.iterrows():
    print(f"{row['locality_name']}: {row['property_count']} props, ₹{row['avg_price']:.2f}Cr avg, {row['avg_area']:.0f} sqft")

print("\n=== TOP 10 MOST EXPENSIVE LOCALITIES (BY AVG PRICE) ===")
top10_price = df_loc.nlargest(10, 'avg_price')[['locality_name', 'property_count', 'avg_price', 'avg_area']]
for idx, row in top10_price.iterrows():
    print(f"{row['locality_name']}: ₹{row['avg_price']:.2f}Cr avg, {row['property_count']} props, {row['avg_area']:.0f} sqft")

print("\n=== LOCALITY PRICE DISTRIBUTION ===")
print(f"Cheapest Locality (avg): ₹{df_loc['avg_price'].min():.2f}Cr")
print(f"Most Expensive Locality (avg): ₹{df_loc['avg_price'].max():.2f}Cr")
print(f"Mean Locality Price: ₹{df_loc['avg_price'].mean():.2f}Cr")
print(f"Median Locality Price: ₹{df_loc['avg_price'].median():.2f}Cr")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
