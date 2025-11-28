"""
NLP Task 3: Description Quality Scoring
Score property descriptions on multiple dimensions
"""

import pandas as pd
import re
from typing import Dict
from textblob import TextBlob


class QualityScorer:
    """Score property description quality"""
    
    def __init__(self):
        pass
    
    def score_completeness(self, text: str, has_amenities: bool = False) -> float:
        """
        Score completeness based on length and detail level
        
        Args:
            text: Property description
            has_amenities: Whether amenities are mentioned
            
        Returns:
            Completeness score (0-25)
        """
        if not text or pd.isna(text):
            return 0.0
        
        score = 0.0
        
        # Length scoring (0-10 points)
        word_count = len(text.split())
        if word_count >= 100:
            score += 10
        elif word_count >= 50:
            score += 7
        elif word_count >= 20:
            score += 4
        elif word_count >= 10:
            score += 2
        
        # Detail indicators (0-10 points)
        detail_keywords = ['located', 'near', 'featuring', 'includes', 'offers', 
                          'spacious', 'modern', 'renovated', 'newly', 'property']
        detail_score = sum(1 for keyword in detail_keywords if keyword in text.lower())
        score += min(detail_score, 10)
        
        # Amenities mentioned (0-5 points)
        if has_amenities:
            score += 5
        
        return min(score, 25)  # Cap at 25
    
    def score_clarity(self, text: str) -> float:
        """
        Score clarity based on readability and grammar
        
        Args:
            text: Property description
            
        Returns:
            Clarity score (0-25)
        """
        if not text or pd.isna(text):
            return 0.0
        
        score = 0.0
        
        # Sentence structure (0-10 points)
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) >= 3:
            score += 10
        elif len(sentences) >= 2:
            score += 6
        elif len(sentences) >= 1:
            score += 3
        
        # Grammar/sentiment analysis using TextBlob (0-10 points)
        try:
            blob = TextBlob(text)
            # Polarity closer to 0 = neutral/factual, closer to 1 = positive
            # We want slightly positive descriptions
            polarity = blob.sentiment.polarity
            if 0.1 <= polarity <= 0.5:  # Good range for property descriptions
                score += 10
            elif -0.1 <= polarity < 0.1:  # Too neutral
                score += 6
            elif polarity > 0.5:  # Too enthusiastic
                score += 7
            else:  # Negative
                score += 3
        except:
            score += 5  # Default if TextBlob fails
        
        # No excessive capitalization (0-5 points)
        upper_ratio = sum(1 for c in text if c.isupper()) / len(text) if text else 0
        if upper_ratio < 0.15:  # Less than 15% uppercase
            score += 5
        elif upper_ratio < 0.3:
            score += 3
        
        return min(score, 25)  # Cap at 25
    
    def score_amenities_count(self, amenities_list: list) -> float:
        """
        Score based on number of amenities
        
        Args:
            amenities_list: List of extracted amenities
            
        Returns:
            Amenities score (0-25)
        """
        if not amenities_list:
            return 0.0
        
        count = len(amenities_list)
        
        if count >= 10:
            return 25
        elif count >= 7:
            return 20
        elif count >= 5:
            return 15
        elif count >= 3:
            return 10
        elif count >= 1:
            return 5
        
        return 0
    
    def score_attractiveness(self, selling_points: list, lifestyle_features: list) -> float:
        """
        Score attractiveness based on selling keywords
        
        Args:
            selling_points: List of selling point keywords
            lifestyle_features: List of lifestyle features
            
        Returns:
            Attractiveness score (0-25)
        """
        total_features = len(selling_points) + len(lifestyle_features)
        
        if total_features >= 8:
            return 25
        elif total_features >= 6:
            return 20
        elif total_features >= 4:
            return 15
        elif total_features >= 2:
            return 10
        elif total_features >= 1:
            return 5
        
        return 0
    
    def calculate_overall_score(self, text: str, amenities: list, 
                               selling_points: list, lifestyle_features: list) -> Dict:
        """
        Calculate comprehensive quality score
        
        Returns:
            Dict with individual scores and overall score (0-100)
        """
        completeness = self.score_completeness(text, bool(amenities))
        clarity = self.score_clarity(text)
        amenity_score = self.score_amenities_count(amenities)
        attractiveness = self.score_attractiveness(selling_points, lifestyle_features)
        
        overall = completeness + clarity + amenity_score + attractiveness
        
        # Quality rating
        if overall >= 80:
            rating = "Excellent"
        elif overall >= 60:
            rating = "Good"
        elif overall >= 40:
            rating = "Average"
        elif overall >= 20:
            rating = "Poor"
        else:
            rating = "Very Poor"
        
        return {
            'completeness_score': round(completeness, 2),
            'clarity_score': round(clarity, 2),
            'amenities_score': round(amenity_score, 2),
            'attractiveness_score': round(attractiveness, 2),
            'overall_score': round(overall, 2),
            'quality_rating': rating
        }


def process_dataset(input_file: str, output_file: str):
    """
    Process dataset to calculate quality scores
    
    Args:
        input_file: Path to enriched data CSV
        output_file: Path to save with quality scores
    """
    print("Loading enriched dataset...")
    df = pd.read_csv(input_file)
    
    print(f"Scoring {len(df)} property descriptions...")
    scorer = QualityScorer()
    
    # Calculate scores
    scores_list = []
    
    for idx, row in df.iterrows():
        # Get text and features
        text = row.get('Raw_Details', '') or row.get('Property Title', '')
        
        # Parse lists safely
        amenities = eval(row['Amenities']) if pd.notna(row.get('Amenities')) and isinstance(row['Amenities'], str) else []
        selling_points = eval(row['Selling_Points']) if pd.notna(row.get('Selling_Points')) and isinstance(row['Selling_Points'], str) else []
        lifestyle = eval(row['Lifestyle_Features']) if pd.notna(row.get('Lifestyle_Features')) and isinstance(row['Lifestyle_Features'], str) else []
        
        # Calculate scores
        scores = scorer.calculate_overall_score(text, amenities, selling_points, lifestyle)
        scores_list.append(scores)
        
        if (idx + 1) % 500 == 0:
            print(f"  Scored {idx + 1}/{len(df)} properties...")
    
    # Add scores to dataframe
    df['Completeness_Score'] = [s['completeness_score'] for s in scores_list]
    df['Clarity_Score'] = [s['clarity_score'] for s in scores_list]
    df['Amenities_Score'] = [s['amenities_score'] for s in scores_list]
    df['Attractiveness_Score'] = [s['attractiveness_score'] for s in scores_list]
    df['Overall_Quality_Score'] = [s['overall_score'] for s in scores_list]
    df['Quality_Rating'] = [s['quality_rating'] for s in scores_list]
    
    # Save
    df.to_csv(output_file, index=False)
    print(f"\n{'='*60}")
    print("✓ Dataset with quality scores saved to:", output_file)
    print(f"✓ Added 6 quality score columns")
    print("="*60)
    
    # Print statistics
    print("\nQUALITY SCORE STATISTICS")
    print("="*60)
    print(f"Average Overall Score: {df['Overall_Quality_Score'].mean():.2f}/100")
    print(f"Average Completeness: {df['Completeness_Score'].mean():.2f}/25")
    print(f"Average Clarity: {df['Clarity_Score'].mean():.2f}/25")
    print(f"Average Amenities Score: {df['Amenities_Score'].mean():.2f}/25")
    print(f"Average Attractiveness: {df['Attractiveness_Score'].mean():.2f}/25")
    print("\nQuality Distribution:")
    print(df['Quality_Rating'].value_counts())
    print("="*60)
    
    return df


if __name__ == "__main__":
    input_path = "../data/enriched_data.csv"
    output_path = "../data/enriched_with_quality.csv"
    
    scored_df = process_dataset(input_path, output_path)
    print("\n✓ Phase 2 - Task 3 Complete!")
