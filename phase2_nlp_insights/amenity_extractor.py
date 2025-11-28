"""
NLP Task 1: Amenities & Feature Extraction
Extract structured information from property descriptions
"""

import re
import pandas as pd
from typing import Dict, List, Set
from .config import AMENITY_KEYWORDS, PROXIMITY_KEYWORDS, SELLING_KEYWORDS


class AmenityExtractor:
    """Extract amenities and features from property descriptions"""
    
    def __init__(self):
        self.amenity_keywords = AMENITY_KEYWORDS
        self.proximity_keywords = PROXIMITY_KEYWORDS
        self.selling_keywords = SELLING_KEYWORDS
        
    def extract_amenities(self, text: str) -> List[str]:
        """
        Extract amenities from description text
        
        Args:
            text: Property description or raw details
            
        Returns:
            List of identified amenities
        """
        if not text or pd.isna(text):
            return []
        
        text_lower = text.lower()
        found_amenities = []
        
        for amenity in self.amenity_keywords:
            if amenity in text_lower:
                found_amenities.append(amenity)
                
        return list(set(found_amenities))  # Remove duplicates
    
    def extract_proximity_features(self, text: str) -> Dict[str, List[str]]:
        """
        Extract proximity information (near metro, hospital, etc.)
        
        Returns:
            Dict with 'locations' and 'distances'
        """
        if not text or pd.isna(text):
            return {'locations': [], 'distances': []}
        
        text_lower = text.lower()
        locations = []
        distances = []
        
        # Find proximity mentions
        for keyword in self.proximity_keywords:
            if keyword in text_lower:
                # Extract context around keyword
                pattern = rf'({keyword}[^.]*)'
                matches = re.findall(pattern, text_lower)
                locations.extend(matches)
        
        # Extract distance mentions (e.g., "2 km", "5 minutes")
        distance_pattern = r'(\d+)\s*(km|meter|minute|min|m)\s*(from|to|away)?'
        distance_matches = re.findall(distance_pattern, text_lower)
        distances = [f"{num} {unit}" for num, unit, _ in distance_matches]
        
        return {
            'locations': list(set(locations)),
            'distances': list(set(distances))
        }
    
    def extract_selling_points(self, text: str) -> List[str]:
        """
        Extract marketing/selling keywords
        
        Returns:
            List of selling points
        """
        if not text or pd.isna(text):
            return []
        
        text_lower = text.lower()
        selling_points = []
        
        for keyword in self.selling_keywords:
            if keyword in text_lower:
                selling_points.append(keyword)
                
        return list(set(selling_points))
    
    def extract_view_facing(self, text: str) -> Dict[str, str]:
        """
        Extract view and facing direction
        
        Returns:
            Dict with 'view' and 'facing'
        """
        if not text or pd.isna(text):
            return {'view': None, 'facing': None}
        
        text_lower = text.lower()
        
        # Extract view
        view_keywords = ['park view', 'garden view', 'lake view', 'city view', 
                        'pool view', 'mountain view', 'road view']
        view = None
        for v in view_keywords:
            if v in text_lower:
                view = v
                break
        
        # Extract facing
        facing_keywords = ['north facing', 'south facing', 'east facing', 
                          'west facing', 'north-east', 'south-west']
        facing = None
        for f in facing_keywords:
            if f in text_lower:
                facing = f
                break
        
        return {'view': view, 'facing': facing}
    
    def extract_lifestyle_highlights(self, text: str) -> List[str]:
        """
        Extract lifestyle-related features
        
        Returns:
            List of lifestyle highlights
        """
        if not text or pd.isna(text):
            return []
        
        text_lower = text.lower()
        lifestyle_keywords = [
            'family friendly', 'pet friendly', 'senior living', 
            'gated community', 'eco friendly', 'smart home',
            'modular kitchen', 'wooden flooring', 'marble flooring',
            'balcony', 'terrace', 'study room', 'servant room',
            'pooja room', 'store room'
        ]
        
        highlights = []
        for keyword in lifestyle_keywords:
            if keyword in text_lower:
                highlights.append(keyword)
                
        return list(set(highlights))
    
    def extract_all_features(self, text: str) -> Dict:
        """
        Extract all features in one go
        
        Returns:
            Complete feature dictionary
        """
        return {
            'amenities': self.extract_amenities(text),
            'proximity': self.extract_proximity_features(text),
            'selling_points': self.extract_selling_points(text),
            'view_facing': self.extract_view_facing(text),
            'lifestyle': self.extract_lifestyle_highlights(text),
            'amenity_count': len(self.extract_amenities(text)),
            'has_proximity_info': len(self.extract_proximity_features(text)['locations']) > 0
        }


def process_dataset(input_file: str, output_file: str):
    """
    Process entire dataset to extract features
    
    Args:
        input_file: Path to raw data CSV
        output_file: Path to save enriched CSV
    """
    print("Loading dataset...")
    df = pd.read_csv(input_file)
    
    print(f"Processing {len(df)} properties...")
    extractor = AmenityExtractor()
    
    # Extract features for each property
    features_list = []
    for idx, row in df.iterrows():
        # Combine title and raw details for better extraction
        text = f"{row.get('Property Title', '')} {row.get('Raw_Details', '')}"
        features = extractor.extract_all_features(text)
        features_list.append(features)
        
        if (idx + 1) % 500 == 0:
            print(f"  Processed {idx + 1}/{len(df)} properties...")
    
    # Add extracted features to dataframe
    df['Amenities'] = [f['amenities'] for f in features_list]
    df['Amenity_Count'] = [f['amenity_count'] for f in features_list]
    df['Proximity_Info'] = [f['proximity']['locations'] for f in features_list]
    df['Selling_Points'] = [f['selling_points'] for f in features_list]
    df['View'] = [f['view_facing']['view'] for f in features_list]
    df['Facing'] = [f['view_facing']['facing'] for f in features_list]
    df['Lifestyle_Features'] = [f['lifestyle'] for f in features_list]
    df['Has_Proximity_Info'] = [f['has_proximity_info'] for f in features_list]
    
    # Save enriched dataset
    df.to_csv(output_file, index=False)
    print(f"\n✓ Enriched dataset saved to: {output_file}")
    print(f"✓ Added 8 new feature columns")
    
    # Print summary
    print("\n" + "="*60)
    print("FEATURE EXTRACTION SUMMARY")
    print("="*60)
    print(f"Average amenities per property: {df['Amenity_Count'].mean():.2f}")
    print(f"Properties with proximity info: {df['Has_Proximity_Info'].sum()} ({df['Has_Proximity_Info'].mean()*100:.1f}%)")
    print(f"Most common amenities: {', '.join(pd.Series([a for sublist in df['Amenities'] for a in sublist]).value_counts().head(5).index.tolist())}")
    print("="*60)
    
    return df


if __name__ == "__main__":
    # Test extraction
    input_path = "../data/ahmedabad_real_estate_data.csv"
    output_path = "../data/enriched_data.csv"
    
    enriched_df = process_dataset(input_path, output_path)
    print("\n✓ Phase 2 - Task 1 Complete!")
