"""
Script to visualize top 20 localities by median price
"""

import pandas as pd
from visualization import plot_top_localities_by_median_price

# Load the cleaned data
print("Loading cleaned data...")
df = pd.read_csv('data/cleaned_data.csv')

print(f"Total properties: {len(df)}")
print(f"Unique localities: {df['Locality'].nunique()}")

# Check if Science City variations exist
science_city_variations = df[df['Locality'].str.contains('Science City', case=False, na=False)]['Locality'].unique()
print(f"\nScience City variations found: {science_city_variations}")

# Create the visualization
print("\nGenerating visualization...")
top_localities_df = plot_top_localities_by_median_price(df, top_n=20)

# Display summary statistics
print("\n" + "="*60)
print("Top 20 Localities by Median Price - Summary")
print("="*60)
print(f"\n{top_localities_df.to_string(index=False)}")
