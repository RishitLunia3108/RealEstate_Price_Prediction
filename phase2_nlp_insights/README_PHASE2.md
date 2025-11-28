# Phase 2: NLP-Based Real Estate Insight Generation Engine

## Overview
Phase 2 extends the ML price prediction system with advanced NLP and LLM capabilities to extract insights from property descriptions, generate summaries, and provide natural language Q&A.

## Features

### 🔍 NLP Task 1: Amenity & Feature Extraction
- Extracts amenities (gym, pool, parking, etc.)
- Identifies proximity features (near metro, hospital, etc.)
- Detects selling points (premium, luxury, spacious, etc.)
- Extracts view/facing information
- Identifies lifestyle highlights

**Output Columns:**
- `Amenities`: List of extracted amenities
- `Amenity_Count`: Number of amenities
- `Proximity_Info`: Nearby locations
- `Selling_Points`: Marketing keywords
- `View`: View type (park view, garden view, etc.)
- `Facing`: Direction (north facing, south facing, etc.)
- `Lifestyle_Features`: Lifestyle highlights
- `Has_Proximity_Info`: Boolean flag

### 📝 NLP Task 2: Property Summary Generation (LLM-Powered)
Generates three types of summaries using Google Gemini:
1. **Clean Summary**: Factual 2-3 sentence description
2. **Marketing Description**: Persuasive description highlighting USPs
3. **Investor Summary**: ROI potential and investment analysis

**Output Columns:**
- `Clean_Summary`
- `Marketing_Description`
- `Investor_Summary`

### ⭐ NLP Task 3: Description Quality Scoring
Multi-dimensional quality scoring:
- **Completeness** (0-25): Length and detail level
- **Clarity** (0-25): Readability and grammar
- **Amenities Score** (0-25): Number of amenities mentioned
- **Attractiveness** (0-25): Selling keywords density

**Output Columns:**
- `Completeness_Score`
- `Clarity_Score`
- `Amenities_Score`
- `Attractiveness_Score`
- `Overall_Quality_Score` (0-100)
- `Quality_Rating` (Excellent/Good/Average/Poor)

### 🏘️ NLP Task 4: Locality-Level Summaries
Aggregates insights at locality level:
- Price ranges and averages
- Common BHK configurations
- Popular amenities
- Target audience identification
- Locality personality (premium/budget, family-friendly, etc.)

**Output File:** `locality_summaries.csv`

## Installation

### Prerequisites
- Python 3.8+
- Phase 1 completed (data cleaning and ML models)
- Google Gemini API key

### Install Dependencies
```bash
pip install -r requirements_phase2.txt
```

### Configure API Key
Edit `config.py` and add your Gemini API key:
```python
GEMINI_API_KEY = "your_api_key_here"
```

## Usage

### Run Complete Pipeline
```python
from phase2_nlp_insights.phase2_main import run_phase2_pipeline

output_files = run_phase2_pipeline(
    input_data_path="../data/ahmedabad_real_estate_data.csv",
    output_dir="../data",
    generate_summaries=True,  # Enable LLM summaries (costs API credits)
    sample_size=None  # Or set to e.g., 100 for testing
)
```

### Run Individual Components

#### 1. Amenity Extraction
```python
from phase2_nlp_insights.amenity_extractor import AmenityExtractor

extractor = AmenityExtractor()
features = extractor.extract_all_features("Property description text here")
print(features)
```

#### 2. Quality Scoring
```python
from phase2_nlp_insights.quality_scorer import QualityScorer

scorer = QualityScorer()
scores = scorer.calculate_overall_score(
    text="Description text",
    amenities=["gym", "parking"],
    selling_points=["premium", "spacious"],
    lifestyle_features=["balcony"]
)
print(f"Overall Score: {scores['overall_score']}/100")
```

#### 3. Summary Generation
```python
from phase2_nlp_insights.summary_generator import SummaryGenerator

generator = SummaryGenerator()
property_data = {
    'Location': 'Bopal',
    'BHK': '3 BHK',
    'Area': '1500 sq ft',
    'Price': '₹1.2 Cr',
    'Amenities': ['gym', 'pool', 'parking']
}

summaries = generator.generate_all_summaries(property_data)
print("Clean Summary:", summaries['clean_summary'])
print("Marketing:", summaries['marketing_description'])
print("Investor:", summaries['investor_summary'])
```

#### 4. Locality Analysis
```python
from phase2_nlp_insights.locality_analyzer import LocalityAnalyzer
import pandas as pd

df = pd.read_csv("../data/enriched_with_quality.csv")
analyzer = LocalityAnalyzer()

# Analyze one locality
bopal_df = df[df['Location'] == 'Bopal']
summary = analyzer.generate_locality_summary(bopal_df)
print(summary['locality_personality'])
```

## Output Files

After running the pipeline, the following files are generated:

1. **enriched_data.csv**: Original data + NLP features
2. **enriched_with_quality.csv**: + Quality scores
3. **enriched_with_summaries.csv**: + LLM-generated summaries (if enabled)
4. **locality_summaries.csv**: Locality-level aggregated insights

## Configuration

Edit `config.py` to customize:

```python
# LLM Settings
LLM_MODEL = "gemini-1.5-flash"  # Or "gemini-1.5-pro"
TEMPERATURE = 0.7
MAX_TOKENS = 2048

# NLP Keywords
AMENITY_KEYWORDS = ['gym', 'swimming pool', ...]
PROXIMITY_KEYWORDS = ['near metro', 'near hospital', ...]
SELLING_KEYWORDS = ['premium', 'luxury', ...]
```

## Performance & Costs

### Processing Time (approximate)
- **Amenity Extraction**: ~1 second per 100 properties
- **Quality Scoring**: ~2 seconds per 100 properties
- **LLM Summaries**: ~1-2 seconds per property (API dependent)
- **Locality Summaries**: ~2-3 seconds per locality

### API Costs (Google Gemini)
- **gemini-1.5-flash**: ~$0.075 per 1,000 properties (3 summaries each)
- **gemini-1.5-pro**: ~$0.525 per 1,000 properties

**Recommendation**: Test with `sample_size=10` first!

## Example Output

### Property with Quality Score
```
Location: Bopal, Ahmedabad
BHK: 3 BHK
Area: 1500 sq ft
Price: ₹1.2 Cr

Quality Score: 78/100 (Good)
  - Completeness: 22/25
  - Clarity: 20/25
  - Amenities: 18/25
  - Attractiveness: 18/25

Amenities (8): gym, swimming pool, parking, security, garden, clubhouse, lift, power backup

Clean Summary: "This 3 BHK apartment in Bopal offers 1500 sq ft of living space priced at ₹1.2 Cr. The property features modern amenities including a gym, swimming pool, and clubhouse."

Marketing Description: "Experience luxury living in this spacious 3 BHK residence in the heart of Bopal! With premium amenities like a swimming pool and gym, this property offers the perfect blend of comfort and convenience for modern families."
```

### Locality Summary
```
Locality: Bopal
Properties: 156
Price Range: ₹0.5 Cr - ₹3.2 Cr (Avg: ₹1.4 Cr)
Common: 3 BHK, Avg Area: 1580 sq ft
Top Amenities: gym, parking, security, swimming pool, power backup

Target Audience: Families with children, Upper-middle class, Fitness enthusiasts

Personality: "Bopal is a premium, family-friendly locality with excellent infrastructure and modern amenities. The area attracts upper-middle-class families seeking spacious homes with resort-style living. With abundant greenery and proximity to IT hubs, it's ideal for working professionals who value quality of life."
```

## Next Steps: Phase 3 (Coming Soon)

Phase 3 will implement:
- 🤖 **LangGraph Multi-Agent System**: Retrieval, Analysis, and Insight agents
- 🔍 **RAG System**: Natural language Q&A over property dataset using FAISS/ChromaDB
- 💬 **Conversational Interface**: Chat with your real estate data

## Troubleshooting

### API Errors
If you see Gemini API errors:
1. Check API key in `config.py`
2. Verify API quota at Google Cloud Console
3. Try reducing `sample_size` for testing

### Memory Issues
For large datasets (>5000 properties):
- Process in batches using `sample_size`
- Use `generate_summaries=False` to skip LLM calls
- Increase system memory or use cloud VM

### Import Errors
Ensure all dependencies are installed:
```bash
pip install -r requirements_phase2.txt
```

## Support
For issues or questions, check:
- Phase 1 README for data requirements
- `config.py` for configuration options
- Sample code in each module's `__main__` section

---
**Built with**: Google Gemini, LangChain, NLTK, TextBlob, spaCy
