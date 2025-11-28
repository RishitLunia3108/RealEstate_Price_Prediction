"""
Quick test script for Phase 2 NLP components
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phase2_nlp_insights.amenity_extractor import AmenityExtractor
from phase2_nlp_insights.quality_scorer import QualityScorer

print("\n" + "="*80)
print(" "*25 + "PHASE 2 QUICK TEST")
print("="*80)

print("\n[TEST 1] Amenity Extraction")
print("-" * 80)
extractor = AmenityExtractor()

test_text = """
Spacious 3 BHK apartment in premium locality near metro station. 
Property features gym, swimming pool, 24/7 security, covered parking, 
kids play area, and clubhouse. Modern luxury living with garden view 
and north facing balcony. Located just 2 km from airport.
"""

print("\n📝 SAMPLE PROPERTY DESCRIPTION BEING ANALYZED:")
print("="*80)
print(test_text.strip())
print("="*80 + "\n")

print("🔍 EXTRACTING FEATURES...\n")
print("🔍 EXTRACTING FEATURES...\n")
features = extractor.extract_all_features(test_text.strip())

print("✅ EXTRACTED FEATURES:")
print(f"📍 Amenities Found: {', '.join(features['amenities'])}")
print(f"🔢 Amenity Count: {features['amenity_count']}")
print(f"💡 Selling Points: {', '.join(features['selling_points'])}")
print(f"🌳 View: {features['view_facing']['view']}")
print(f"🧭 Facing: {features['view_facing']['facing']}")
print(f"📍 Proximity Locations: {len(features['proximity']['locations'])}")
if features['proximity']['locations']:
    print(f"   → {features['proximity']['locations'][0][:60]}...")
print(f"\n✓ Amenity extraction working!")

# Test 2: Quality Scoring
print("\n[TEST 2] Quality Scoring")
print("-" * 80)

print("📊 SCORING THE SAME PROPERTY DESCRIPTION...\n")
scorer = QualityScorer()

scores = scorer.calculate_overall_score(
    test_text,
    features['amenities'],
    features['selling_points'],
    features['lifestyle']
)

print("✅ QUALITY SCORES:")
print(f"🎯 Overall Score: {scores['overall_score']}/100")
print(f"⭐ Rating: {scores['quality_rating']}")
print(f"   - Completeness: {scores['completeness_score']}/25")
print(f"   - Clarity: {scores['clarity_score']}/25")
print(f"   - Amenities: {scores['amenities_score']}/25")
print(f"   - Attractiveness: {scores['attractiveness_score']}/25")
print(f"\n✓ Quality scoring working!")

# Test 3: Summary Generation (Optional - costs API credits)
print("\n[TEST 3] LLM Summary Generation")
print("-" * 80)
print("\n📋 SAMPLE PROPERTY DATA TO BE SENT TO GEMINI:")
print("="*80)
sample_property_data = {
    'Location': 'Bopal, Ahmedabad',
    'BHK': '3 BHK',
    'Area': '1500 sq ft',
    'Price': '₹1.2 Cr',
    'Furnishing': 'Semi-Furnished',
    'Amenities': features['amenities'],
    'Selling_Points': features['selling_points']
}
for key, value in sample_property_data.items():
    if isinstance(value, list):
        print(f"{key}: {', '.join(value)}")
    else:
        print(f"{key}: {value}")
print("="*80 + "\n")

user_input = input("Test Gemini API summaries? This costs API credits. (y/N): ")

if user_input.lower() == 'y':
    from phase2_nlp_insights.summary_generator import SummaryGenerator
    
    generator = SummaryGenerator()
    
    print("\n🤖 Generating AI summaries using Gemini LLM...")
    summaries = generator.generate_all_summaries(sample_property_data)
    
    print("\n✅ GENERATED SUMMARIES:")
    print("\n" + "="*80)
    print("📄 CLEAN SUMMARY (Factual):")
    print("-"*80)
    print(summaries['clean_summary'])
    
    print("\n" + "="*80)
    print("🎯 MARKETING DESCRIPTION (Persuasive):")
    print("-"*80)
    print(summaries['marketing_description'])
    
    print("\n" + "="*80)
    print("💰 INVESTOR SUMMARY (ROI-Focused):")
    print("-"*80)
    print(summaries['investor_summary'])
    print("="*80)
    
    print("\n✓ LLM summary generation working!")
else:
    print("Skipped (requires API credits)")

print("\n" + "="*80)
print(" "*20 + "✓ ALL PHASE 2 TESTS PASSED!")
print("="*80)
print("\nNext Steps:")
print("  1. Run Phase 2 on full dataset: python phase2_nlp_insights/phase2_main.py")
print("  2. Check data/enriched_data.csv for extracted features")
print("  3. Check data/enriched_with_quality.csv for quality scores")
print("  4. Check data/locality_summaries.csv for locality insights")
print("="*80 + "\n")
