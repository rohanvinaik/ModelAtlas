"""Configuration constants for ModelAtlas."""

from pathlib import Path

# Cache and storage
CACHE_DIR = Path.home() / ".cache" / "model-atlas"
MODEL_CARD_CACHE_DIR = CACHE_DIR / "model_cards"
NETWORK_DB_PATH = CACHE_DIR / "network.db"

# Cache TTL (seconds)
MODEL_CARD_TTL = 60 * 60 * 24  # 24 hours

# HuggingFace API defaults
DEFAULT_CANDIDATE_LIMIT = 500  # How many models to pull from HF API per query
DEFAULT_RESULT_LIMIT = 20  # How many results to return to the user

# Maximum model card text length for extraction (chars)
MAX_CARD_TEXT_LENGTH = 2000

# Index build defaults
DEFAULT_INDEX_SIZE = 2000  # Models per batch when building index

# Query scoring weights
WEIGHT_BANK_PROXIMITY = 0.35  # How close in bank-position space
WEIGHT_ANCHOR_OVERLAP = 0.45  # Jaccard similarity on anchor sets
WEIGHT_FUZZY = 0.20  # Fuzzy name-resolution score
