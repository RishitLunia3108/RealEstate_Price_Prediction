"""
Phase 2: NLP-Based Real Estate Insight Generation
Main orchestration pipeline for all NLP tasks
"""

import os
import sys
import pandas as pd
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phase2_nlp_insights.amenity_extractor import AmenityExtractor, process_dataset as extract_amenities
from phase2_nlp_insights.quality_scorer import QualityScorer, process_dataset as score_quality
from phase2_nlp_insights.locality_analyzer import LocalityAnalyzer, process_all_localities


def run_phase2_pipeline(input_data_path: str, output_dir: str = "../data", 
                        generate_summaries: bool = False, sample_size: int = None,
                        gemini_limit: int = 10):
    """
    Run complete Phase 2 NLP pipeline
    
    Args:
        input_data_path: Path to raw scraped data CSV
        output_dir: Directory to save outputs
        generate_summaries: Whether to generate LLM summaries (costs API calls)
        sample_size: If provided, only process first N properties
        gemini_limit: Number of properties to use Gemini API for (default: 10)
    
    Pipeline Steps:
        1. Extract amenities and features (NLP Task 1)
        2. Score description quality (NLP Task 3)
        3. Generate property summaries (NLP Task 2) - Optional
        4. Generate locality-level summaries (NLP Task 4)
    """
    
    print("\n" + "="*80)
    print(" "*20 + "PHASE 2: NLP INSIGHT GENERATION ENGINE")
    print("="*80)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Input Data: {input_data_path}")
    print(f"Output Directory: {output_dir}")
    print(f"Generate Summaries: {generate_summaries}")
    if sample_size:
        print(f"Sample Size: {sample_size} properties")
    if generate_summaries:
        print(f"Gemini API Limit: First {gemini_limit} properties only")
    print("="*80 + "\n")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Define file paths
    enriched_data_path = os.path.join(output_dir, "enriched_data.csv")
    quality_scored_path = os.path.join(output_dir, "enriched_with_quality.csv")
    summaries_path = os.path.join(output_dir, "enriched_with_summaries.csv")
    locality_summaries_path = os.path.join(output_dir, "locality_summaries.csv")
    
    # ==================== STEP 1: AMENITY EXTRACTION ====================
    print("\n" + "="*80)
    print("STEP 1: AMENITIES & FEATURE EXTRACTION")
    print("="*80)
    
    print("\nExtracting amenities, proximity info, selling points, and lifestyle features...")
    extract_amenities(input_data_path, enriched_data_path)
    
    # Load and optionally sample
    df = pd.read_csv(enriched_data_path)
    if sample_size and sample_size < len(df):
        print(f"\nSampling {sample_size} properties for faster processing...")
        df = df.head(sample_size)
        df.to_csv(enriched_data_path, index=False)
    
    print(f"\n✓ Step 1 Complete: {len(df)} properties enriched with NLP features")
    
    # ==================== STEP 2: QUALITY SCORING ====================
    print("\n" + "="*80)
    print("STEP 2: DESCRIPTION QUALITY SCORING")
    print("="*80)
    
    print("\nScoring descriptions on completeness, clarity, amenities, and attractiveness...")
    score_quality(enriched_data_path, quality_scored_path)
    
    print("\n✓ Step 2 Complete: Quality scores calculated")
    
    # ==================== STEP 3: PROPERTY SUMMARIES (Optional) ====================
    if generate_summaries:
        print("\n" + "="*80)
        print("STEP 3: PROPERTY SUMMARY GENERATION (Using Gemini LLM)")
        print("="*80)
        print(f"\n⚠️  Gemini API will be used for FIRST {gemini_limit} properties only")
        print("   Remaining properties will have placeholder summaries")
        
        from phase2_nlp_insights.summary_generator import process_dataset as generate_summaries_func
        
        generate_summaries_func(quality_scored_path, summaries_path, 
                              sample_size=sample_size, use_gemini_limit=gemini_limit)
        
        print("\n✓ Step 3 Complete: Property summaries generated")
        final_data_path = summaries_path
    else:
        print("\n" + "="*80)
        print("STEP 3: PROPERTY SUMMARY GENERATION - SKIPPED")
        print("="*80)
        print("  (Set generate_summaries=True to enable)")
        final_data_path = quality_scored_path
    
    # ==================== STEP 4: LOCALITY SUMMARIES ====================
    print("\n" + "="*80)
    print("STEP 4: LOCALITY-LEVEL SUMMARY GENERATION")
    print("="*80)
    
    print("\nGenerating locality personalities and aggregate insights...")
    process_all_localities(final_data_path, locality_summaries_path, min_properties=5)
    
    print("\n✓ Step 4 Complete: Locality summaries generated")
    
    # ==================== PIPELINE COMPLETE ====================
    print("\n" + "="*80)
    print(" "*25 + "PHASE 2 PIPELINE COMPLETE!")
    print("="*80)
    print("\nGenerated Files:")
    print(f"  1. {enriched_data_path}")
    print(f"  2. {quality_scored_path}")
    if generate_summaries:
        print(f"  3. {summaries_path}")
    print(f"  4. {locality_summaries_path}")
    print(f"\nEnd Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80 + "\n")
    
    return {
        'enriched_data': enriched_data_path,
        'quality_scored': quality_scored_path,
        'summaries': summaries_path if generate_summaries else None,
        'locality_summaries': locality_summaries_path
    }


def display_sample_insights(output_files: dict, num_samples: int = 3):
    """
    Display sample insights from generated data
    
    Args:
        output_files: Dict with paths to generated files
        num_samples: Number of sample properties to display
    """
    print("\n" + "="*80)
    print(" "*25 + "SAMPLE INSIGHTS")
    print("="*80)
    
    # Load quality scored data
    df = pd.read_csv(output_files['quality_scored'])
    
    print(f"\n📊 DATASET STATISTICS:")
    print(f"  Total Properties: {len(df)}")
    print(f"  Average Quality Score: {df['Overall_Quality_Score'].mean():.2f}/100")
    print(f"  Average Amenities per Property: {df['Amenity_Count'].mean():.2f}")
    print(f"  Properties with Proximity Info: {df['Has_Proximity_Info'].sum()} ({df['Has_Proximity_Info'].mean()*100:.1f}%)")
    
    print(f"\n🏆 TOP {num_samples} QUALITY PROPERTIES:")
    top_properties = df.nlargest(num_samples, 'Overall_Quality_Score')
    
    for idx, row in top_properties.iterrows():
        print(f"\n  Property {idx + 1}:")
        print(f"    Location: {row.get('Location', 'N/A')}")
        print(f"    BHK: {row.get('BHK', 'N/A')}, Area: {row.get('Area', 'N/A')}")
        print(f"    Price: {row.get('Price', 'N/A')}")
        print(f"    Quality Score: {row['Overall_Quality_Score']:.1f}/100 ({row['Quality_Rating']})")
        
        # Parse amenities
        amenities = eval(row['Amenities']) if pd.notna(row.get('Amenities')) and isinstance(row['Amenities'], str) else []
        print(f"    Amenities ({len(amenities)}): {', '.join(amenities[:5])}")
    
    # Load locality summaries
    if output_files.get('locality_summaries'):
        locality_df = pd.read_csv(output_files['locality_summaries'])
        
        print(f"\n🏘️  TOP 3 LOCALITIES:")
        for idx, row in locality_df.head(3).iterrows():
            print(f"\n  {row['locality_name']} ({row['property_count']} properties)")
            print(f"    Price Range: {row['price_range']}")
            print(f"    Common: {row['common_bhk']}, Avg Area: {row['avg_area']}")
            print(f"    Target: {row['target_audience']}")
            print(f"    Personality: {row['locality_personality'][:150]}...")
    
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    # Run Phase 2 pipeline
    # Get the correct path relative to this script's location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    input_path = os.path.join(parent_dir, "data", "ahmedabad_real_estate_data.csv")
    output_dir = os.path.join(parent_dir, "data")
    
    # Process with Gemini API for first 10 properties only
    # Set generate_summaries=True to enable, gemini_limit to control API usage
    output_files = run_phase2_pipeline(
        input_data_path=input_path,
        output_dir=output_dir,
        generate_summaries=True,  # Set to True for LLM summaries
        sample_size=None,  # Set to e.g., 100 for faster testing
        gemini_limit=10  # Only use Gemini API for first 10 properties
    )
    
    # Display insights
    display_sample_insights(output_files, num_samples=3)
    
    print("✓ Phase 2 Complete! Ready for Phase 3 (LangGraph Agents & RAG)")
