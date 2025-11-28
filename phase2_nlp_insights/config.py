"""
Configuration for Phase 2 NLP Components
"""

# API Keys (Add your keys here)
GEMINI_API_KEY = "AIzaSyBK3HGz-RBDjhnO0qkcSfIupls6rI9PsAs"  # From Phase 2 doc

# Model Settings
LLM_MODEL = "gemini-2.5-flash"  # or "gemini-1.5-pro"
EMBEDDING_MODEL = "models/embedding-001"  # Gemini embeddings
TEMPERATURE = 0.7
MAX_TOKENS = 2048

# Data Paths
PHASE1_DATA_PATH = "../data/cleaned_data.csv"
RAW_DATA_PATH = "../data/ahmedabad_real_estate_data.csv"
VECTOR_STORE_PATH = "./vector_store"
OUTPUT_PATH = "./outputs"

# NLP Settings
MIN_DESCRIPTION_LENGTH = 50
AMENITY_KEYWORDS = [
    'gym', 'swimming pool', 'parking', 'security', 'garden', 'clubhouse',
    'lift', 'power backup', 'wifi', 'cctv', 'playground', 'jogging track',
    'community hall', 'restaurant', 'shopping mall', 'school', 'hospital',
    'metro', 'park', 'lake', 'temple', 'gated community', 'vastu compliant'
]

PROXIMITY_KEYWORDS = [
    'near metro', 'near hospital', 'near school', 'near mall', 'near airport',
    'walking distance', 'close to', 'minutes from', 'adjacent to'
]

SELLING_KEYWORDS = [
    'premium', 'luxury', 'spacious', 'modern', 'elegant', 'newly built',
    'ready to move', 'vastu', 'prime location', 'best deal', 'investment'
]
