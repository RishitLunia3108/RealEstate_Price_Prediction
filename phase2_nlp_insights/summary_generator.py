"""
NLP Task 2: Property Summary Generator
Generate three types of summaries using Gemini LLM
"""

import pandas as pd
import google.generativeai as genai
from typing import Dict, Optional
from .config import GEMINI_API_KEY, LLM_MODEL, TEMPERATURE, MAX_TOKENS


class SummaryGenerator:
    """Generate property summaries using Gemini LLM"""
    
    def __init__(self):
        genai.configure(api_key=GEMINI_API_KEY)
        self.model = genai.GenerativeModel(LLM_MODEL)
        self.generation_config = {
            'temperature': TEMPERATURE,
            'max_output_tokens': MAX_TOKENS
        }
    
    def generate_clean_summary(self, property_data: Dict) -> str:
        """
        Generate clean, factual summary (2-3 sentences)
        
        Args:
            property_data: Dict with property details
            
        Returns:
            Clean summary string
        """
        prompt = f"""
Generate a clean, factual 2-3 sentence summary for this property:

Location: {property_data.get('Location', 'N/A')}
BHK: {property_data.get('BHK', 'N/A')}
Area: {property_data.get('Area', 'N/A')}
Price: {property_data.get('Price', 'N/A')}
Furnishing: {property_data.get('Furnishing', 'N/A')}
Amenities: {', '.join(property_data.get('Amenities', [])) if property_data.get('Amenities') else 'N/A'}

Write a concise, neutral summary focusing on key facts only. No marketing language.
"""
        
        try:
            response = self.model.generate_content(
                prompt,
                generation_config=self.generation_config
            )
            return response.text.strip()
        except Exception as e:
            return f"Error generating summary: {str(e)}"
    
    def generate_marketing_description(self, property_data: Dict) -> str:
        """
        Generate persuasive marketing-style description
        
        Returns:
            Marketing description with USPs highlighted
        """
        prompt = f"""
Create a compelling marketing description for this property:

Location: {property_data.get('Location', 'N/A')}
BHK: {property_data.get('BHK', 'N/A')}
Area: {property_data.get('Area', 'N/A')}
Price: {property_data.get('Price', 'N/A')}
Furnishing: {property_data.get('Furnishing', 'N/A')}
Amenities: {', '.join(property_data.get('Amenities', [])) if property_data.get('Amenities') else 'N/A'}
Selling Points: {', '.join(property_data.get('Selling_Points', [])) if property_data.get('Selling_Points') else 'N/A'}

Write an attractive 3-4 sentence description that:
- Highlights unique selling points
- Uses persuasive language
- Appeals to potential buyers
- Emphasizes lifestyle benefits
"""
        
        try:
            response = self.model.generate_content(
                prompt,
                generation_config=self.generation_config
            )
            return response.text.strip()
        except Exception as e:
            return f"Error generating marketing description: {str(e)}"
    
    def generate_investor_summary(self, property_data: Dict) -> str:
        """
        Generate investor-focused summary (ROI potential)
        
        Returns:
            Investment analysis summary
        """
        prompt = f"""
Create an investor-focused summary for this property:

Location: {property_data.get('Location', 'N/A')}
BHK: {property_data.get('BHK', 'N/A')}
Area: {property_data.get('Area', 'N/A')}
Price: {property_data.get('Price', 'N/A')}
Price per SqFt: {property_data.get('Price_Per_SqFt', 'N/A')}
Amenities: {', '.join(property_data.get('Amenities', [])) if property_data.get('Amenities') else 'N/A'}

Write a 3-4 sentence investment analysis covering:
- Rental yield potential
- Capital appreciation prospects
- Location advantages for investment
- Target tenant demographics
"""
        
        try:
            response = self.model.generate_content(
                prompt,
                generation_config=self.generation_config
            )
            return response.text.strip()
        except Exception as e:
            return f"Error generating investor summary: {str(e)}"
    
    def generate_all_summaries(self, property_data: Dict) -> Dict[str, str]:
        """
        Generate all three summary types
        
        Returns:
            Dict with all summaries
        """
        print(f"  Generating summaries for property in {property_data.get('Location', 'Unknown')}...")
        
        return {
            'clean_summary': self.generate_clean_summary(property_data),
            'marketing_description': self.generate_marketing_description(property_data),
            'investor_summary': self.generate_investor_summary(property_data)
        }


def process_dataset(input_file: str, output_file: str, sample_size: Optional[int] = None, 
                   use_gemini_limit: int = 10):
    """
    Process dataset to generate summaries for all properties
    
    Args:
        input_file: Path to enriched data CSV
        output_file: Path to save with summaries
        sample_size: If provided, only process first N properties (for testing)
        use_gemini_limit: Number of properties to use Gemini API for (default: 10)
    """
    print("Loading enriched dataset...")
    df = pd.read_csv(input_file)
    
    if sample_size:
        df = df.head(sample_size)
        print(f"Processing first {sample_size} properties (sample mode)...")
    else:
        print(f"Processing all {len(df)} properties...")
    
    print(f"⚠️  Gemini API will be used for FIRST {use_gemini_limit} properties only")
    print(f"   Remaining {max(0, len(df) - use_gemini_limit)} properties will have placeholder summaries")
    
    generator = SummaryGenerator()
    
    # Generate summaries
    clean_summaries = []
    marketing_descriptions = []
    investor_summaries = []
    
    for idx, row in df.iterrows():
        print(f"\nProperty {idx + 1}/{len(df)}", end="")
        
        # Use Gemini API only for first 'use_gemini_limit' properties
        if idx < use_gemini_limit:
            print(" [Using Gemini API]")
            
            # Prepare property data
            property_data = {
                'Location': row.get('Location', 'N/A'),
                'BHK': row.get('BHK', 'N/A'),
                'Area': row.get('Area', 'N/A'),
                'Price': row.get('Price', 'N/A'),
                'Furnishing': row.get('Furnishing', 'N/A'),
                'Amenities': eval(row['Amenities']) if pd.notna(row.get('Amenities')) and isinstance(row['Amenities'], str) else [],
                'Selling_Points': eval(row['Selling_Points']) if pd.notna(row.get('Selling_Points')) and isinstance(row['Selling_Points'], str) else []
            }
            
            # Generate summaries using Gemini
            summaries = generator.generate_all_summaries(property_data)
            clean_summaries.append(summaries['clean_summary'])
            marketing_descriptions.append(summaries['marketing_description'])
            investor_summaries.append(summaries['investor_summary'])
        else:
            print(" [Skipping Gemini - API limit reached]")
            
            # Create placeholder summaries without using Gemini API
            location = row.get('Location', 'N/A')
            bhk = row.get('BHK', 'N/A')
            area = row.get('Area', 'N/A')
            price = row.get('Price', 'N/A')
            furnishing = row.get('Furnishing', 'N/A')
            
            clean_summaries.append(f"{bhk} property in {location}, {area}, priced at {price}. {furnishing} status.")
            marketing_descriptions.append(f"Beautiful {bhk} property available in {location}. Contact for details.")
            investor_summaries.append(f"Investment opportunity: {bhk} in {location} at {price}.")
        
        if (idx + 1) % 10 == 0:
            print(f"\n✓ Completed {idx + 1}/{len(df)} properties")
    
    # Add summaries to dataframe
    df['Clean_Summary'] = clean_summaries
    df['Marketing_Description'] = marketing_descriptions
    df['Investor_Summary'] = investor_summaries
    
    # Save
    df.to_csv(output_file, index=False)
    print(f"\n{'='*60}")
    print("✓ Dataset with summaries saved to:", output_file)
    print(f"✓ Added 3 summary columns")
    print(f"✓ Gemini API used for: {min(use_gemini_limit, len(df))} properties")
    print(f"✓ Placeholder summaries for: {max(0, len(df) - use_gemini_limit)} properties")
    print("="*60)
    
    return df


if __name__ == "__main__":
    # Test on sample (10 properties with Gemini)
    input_path = "../data/enriched_data.csv"
    output_path = "../data/enriched_with_summaries.csv"
    
    # Process with Gemini limit of 10 (change use_gemini_limit for more/less API calls)
    enriched_df = process_dataset(input_path, output_path, sample_size=None, use_gemini_limit=10)
    print("\n✓ Phase 2 - Task 2 Complete!")
