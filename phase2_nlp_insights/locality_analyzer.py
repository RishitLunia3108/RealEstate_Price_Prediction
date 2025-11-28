"""
NLP Task 4: Locality-Level Summary Generation
Aggregate insights and generate locality personality
"""

import pandas as pd
import google.generativeai as genai
from typing import Dict, List
from collections import Counter
from .config import GEMINI_API_KEY, LLM_MODEL, TEMPERATURE


class LocalityAnalyzer:
    """Analyze and generate locality-level summaries"""
    
    def __init__(self):
        genai.configure(api_key=GEMINI_API_KEY)
        self.model = genai.GenerativeModel(LLM_MODEL)
        self.generation_config = {
            'temperature': TEMPERATURE,
            'max_output_tokens': 1024
        }
    
    def _parse_price(self, price_series):
        """Parse price strings like '₹1.5 Cr', '₹85 Lac' to numeric (in Cr)"""
        import re
        def parse_single_price(price_str):
            if pd.isna(price_str):
                return 0
            price_str = str(price_str).replace('₹', '').replace(',', '').strip()
            if 'Cr' in price_str or 'Crore' in price_str:
                return float(re.findall(r'[\d.]+', price_str)[0])
            elif 'Lac' in price_str or 'Lakh' in price_str:
                return float(re.findall(r'[\d.]+', price_str)[0]) / 100
            else:
                try:
                    return float(price_str) / 10000000  # Assume raw value
                except:
                    return 0
        return price_series.apply(parse_single_price)
    
    def _parse_area(self, area_series):
        """Parse area strings like '1500 sqft', '200 sqyrd' to sqft"""
        import re
        def parse_single_area(area_str):
            if pd.isna(area_str):
                return 0
            area_str = str(area_str).replace(',', '').strip()
            try:
                value = float(re.findall(r'[\d.]+', area_str)[0])
                if 'sqyrd' in area_str or 'yard' in area_str:
                    return value * 9  # Convert to sqft
                elif 'sqm' in area_str:
                    return value * 10.764  # Convert to sqft
                else:
                    return value  # Assume sqft
            except:
                return 0
        return area_series.apply(parse_single_area)
    
    def _parse_bhk(self, bhk_series):
        """Parse BHK strings like '3 BHK', '2BHK' to numeric"""
        import re
        def parse_single_bhk(bhk_str):
            if pd.isna(bhk_str):
                return 0
            try:
                return int(re.findall(r'\d+', str(bhk_str))[0])
            except:
                return 0
        return bhk_series.apply(parse_single_bhk)
    
    def aggregate_locality_stats(self, locality_df: pd.DataFrame) -> Dict:
        """
        Calculate aggregate statistics for a locality
        
        Args:
            locality_df: DataFrame filtered for one locality
            
        Returns:
            Dict with locality statistics
        """
        stats = {
            'locality_name': locality_df['Locality'].iloc[0] if len(locality_df) > 0 else 'Unknown',
            'property_count': len(locality_df),
            'avg_price': self._parse_price(locality_df['Price']).mean() if 'Price' in locality_df.columns else 0,
            'min_price': self._parse_price(locality_df['Price']).min() if 'Price' in locality_df.columns else 0,
            'max_price': self._parse_price(locality_df['Price']).max() if 'Price' in locality_df.columns else 0,
            'avg_area': self._parse_area(locality_df['Area']).mean() if 'Area' in locality_df.columns else 0,
            'common_bhk': self._parse_bhk(locality_df['BHK']).mode()[0] if 'BHK' in locality_df.columns and len(locality_df) > 0 else 0,
            'furnishing_distribution': locality_df['Furnishing'].value_counts().to_dict() if 'Furnishing' in locality_df.columns else {}
        }
        
        # Aggregate amenities
        all_amenities = []
        for amenities_str in locality_df['Amenities']:
            if pd.notna(amenities_str) and amenities_str:
                try:
                    amenities = eval(amenities_str) if isinstance(amenities_str, str) else amenities_str
                    all_amenities.extend(amenities)
                except:
                    pass
        
        amenity_counts = Counter(all_amenities)
        stats['top_amenities'] = [item for item, count in amenity_counts.most_common(10)]
        stats['amenity_frequency'] = dict(amenity_counts.most_common(10))
        
        # Average quality score
        if 'Overall_Quality_Score' in locality_df:
            stats['avg_quality_score'] = locality_df['Overall_Quality_Score'].mean()
        
        return stats
    
    def identify_target_audience(self, stats: Dict) -> str:
        """
        Identify target audience based on property characteristics
        
        Returns:
            Target audience string
        """
        bhk = stats.get('common_bhk', 2)
        avg_price = stats.get('avg_price', 0)
        
        audience = []
        
        # Based on BHK
        if bhk >= 4:
            audience.append("Large families")
        elif bhk == 3:
            audience.append("Families with children")
        elif bhk == 2:
            audience.append("Small families, Working professionals")
        else:
            audience.append("Students, Single professionals")
        
        # Based on price
        if avg_price > 2.0:  # > 2 Cr
            audience.append("High-income buyers")
        elif avg_price > 1.0:  # 1-2 Cr
            audience.append("Upper-middle class")
        elif avg_price > 0.5:  # 50L-1Cr
            audience.append("Middle-income families")
        else:
            audience.append("First-time buyers, Budget-conscious")
        
        # Based on amenities
        amenities = stats.get('top_amenities', [])
        if 'gym' in amenities or 'swimming pool' in amenities:
            audience.append("Fitness enthusiasts")
        if 'kids play area' in amenities or 'school' in [a.lower() for a in amenities]:
            audience.append("Families with kids")
        
        return ", ".join(set(audience))
    
    def generate_locality_personality(self, stats: Dict) -> str:
        """
        Generate locality personality using LLM
        
        Returns:
            Personality description
        """
        prompt = f"""
Analyze this locality data and describe its personality in 3-4 sentences:

Locality: {stats['locality_name']}
Properties: {stats['property_count']}
Price Range: ₹{stats['min_price']:.2f} Cr - ₹{stats['max_price']:.2f} Cr (Avg: ₹{stats['avg_price']:.2f} Cr)
Average Area: {stats['avg_area']:.0f} sq ft
Most Common: {stats['common_bhk']} BHK
Top Amenities: {', '.join(stats['top_amenities'][:5])}

Describe the locality's character (premium/budget, family-friendly/bachelor-suitable, quiet/bustling, modern/traditional, etc.)
"""
        
        try:
            response = self.model.generate_content(
                prompt,
                generation_config=self.generation_config
            )
            return response.text.strip()
        except Exception as e:
            return f"Unable to generate personality: {str(e)}"
    
    def generate_locality_summary(self, locality_df: pd.DataFrame) -> Dict:
        """
        Generate complete locality summary
        
        Returns:
            Comprehensive locality summary dict
        """
        # Get statistics
        stats = self.aggregate_locality_stats(locality_df)
        
        # Identify audience
        target_audience = self.identify_target_audience(stats)
        
        # Generate personality
        personality = self.generate_locality_personality(stats)
        
        return {
            'locality_name': stats['locality_name'],
            'property_count': stats['property_count'],
            'price_range': f"₹{stats['min_price']:.2f} Cr - ₹{stats['max_price']:.2f} Cr",
            'avg_price': f"₹{stats['avg_price']:.2f} Cr",
            'avg_area': f"{stats['avg_area']:.0f} sq ft",
            'common_bhk': f"{stats['common_bhk']} BHK",
            'top_amenities': ', '.join(stats['top_amenities'][:5]),
            'target_audience': target_audience,
            'locality_personality': personality,
            'avg_quality_score': round(stats.get('avg_quality_score', 0), 2)
        }


def process_all_localities(input_file: str, output_file: str, min_properties: int = 5):
    """
    Generate summaries for all localities
    
    Args:
        input_file: Path to enriched data CSV
        output_file: Path to save locality summaries CSV
        min_properties: Minimum properties to analyze a locality
    """
    print("Loading enriched dataset...")
    df = pd.read_csv(input_file)
    
    # Group by locality
    localities = df['Locality'].unique()
    print(f"Found {len(localities)} unique localities")
    
    analyzer = LocalityAnalyzer()
    locality_summaries = []
    
    for idx, locality in enumerate(localities):
        locality_df = df[df['Locality'] == locality]
        
        if len(locality_df) < min_properties:
            print(f"  Skipping {locality} (only {len(locality_df)} properties)")
            continue
        
        print(f"\n[{idx+1}/{len(localities)}] Analyzing {locality} ({len(locality_df)} properties)...")
        
        summary = analyzer.generate_locality_summary(locality_df)
        locality_summaries.append(summary)
    
    # Create summary dataframe
    summary_df = pd.DataFrame(locality_summaries)
    
    # Sort by property count
    summary_df = summary_df.sort_values('property_count', ascending=False)
    
    # Save
    summary_df.to_csv(output_file, index=False)
    
    print(f"\n{'='*60}")
    print("✓ Locality summaries saved to:", output_file)
    print(f"✓ Generated summaries for {len(locality_summaries)} localities")
    print("="*60)
    
    # Print top 5 localities
    print("\nTOP 5 LOCALITIES BY PROPERTY COUNT:")
    print("="*60)
    for idx, row in summary_df.head(5).iterrows():
        print(f"\n{row['locality_name']} ({row['property_count']} properties)")
        print(f"  Price: {row['price_range']} (Avg: {row['avg_price']})")
        print(f"  Common: {row['common_bhk']}, Avg Area: {row['avg_area']}")
        print(f"  Top Amenities: {row['top_amenities']}")
        print(f"  Target: {row['target_audience']}")
        print(f"  Personality: {row['locality_personality'][:100]}...")
    print("="*60)
    
    return summary_df


if __name__ == "__main__":
    input_path = "../data/enriched_with_quality.csv"
    output_path = "../data/locality_summaries.csv"
    
    summary_df = process_all_localities(input_path, output_path)
    print("\n✓ Phase 2 - Task 4 Complete!")
