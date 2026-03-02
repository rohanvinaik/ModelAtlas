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
WEIGHT_BANK = 0.30  # How close in bank-position space
WEIGHT_ANCHOR = 0.35  # Jaccard similarity on anchor sets
WEIGHT_SPREAD = 0.15  # Spreading activation from seed models
WEIGHT_FUZZY = 0.20  # Fuzzy name-resolution score

# Spreading activation constants
SPREAD_DECAY = 0.8
SPREAD_MAX_DEPTH = 3
NEIGHBOR_SLICE = 20  # Max link neighbors per node
ANCHOR_SLICE = 15  # Max anchor co-occurrences per node

# Relation-specific link weights for spreading
LINK_WEIGHTS: dict[str, float] = {
    "fine_tuned_from": 0.9,
    "quantized_from": 0.85,
    "variant_of": 0.8,
    "same_family": 0.7,
    "predecessor": 0.6,
    "successor": 0.6,
}

# Navigate (structured scoring) constants
NAVIGATE_MISSING_BANK_PENALTY = 0.3  # Score for a bank when model has no position
NAVIGATE_AVOID_DECAY = 0.5  # Each avoided anchor multiplies score by this

# Ingest daemon settings
INGEST_DB_PATH = CACHE_DIR / "ingest_state.db"
INGEST_BATCH_SIZE = 50
INGEST_MIN_LIKES = 5
INGEST_VIBE_MIN_LIKES = 50
VIBE_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
VIBE_MAX_RETRIES = 3
