"""CAR-T Intelligence Agent configuration.

Follows the same Pydantic BaseSettings pattern as rag-chat-pipeline/config/settings.py.
"""

from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings


class CARTSettings(BaseSettings):
    """Configuration for CAR-T Intelligence Agent."""

    # ── Paths ──
    PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
    DATA_DIR: Path = PROJECT_ROOT / "data"
    CACHE_DIR: Path = DATA_DIR / "cache"
    REFERENCE_DIR: Path = DATA_DIR / "reference"

    # ── RAG Pipeline (reuse existing) ──
    RAG_PIPELINE_ROOT: Path = Path(
        "/home/adam/projects/hcls-ai-factory/rag-chat-pipeline"
    )

    # ── Milvus ──
    MILVUS_HOST: str = "localhost"
    MILVUS_PORT: int = 19530

    # Collection names
    COLLECTION_LITERATURE: str = "cart_literature"
    COLLECTION_TRIALS: str = "cart_trials"
    COLLECTION_CONSTRUCTS: str = "cart_constructs"
    COLLECTION_ASSAYS: str = "cart_assays"
    COLLECTION_MANUFACTURING: str = "cart_manufacturing"
    COLLECTION_GENOMIC: str = "genomic_evidence"  # Existing collection

    # ── Embeddings ──
    EMBEDDING_MODEL: str = "BAAI/bge-small-en-v1.5"
    EMBEDDING_DIMENSION: int = 384
    EMBEDDING_BATCH_SIZE: int = 32

    # ── LLM ──
    LLM_PROVIDER: str = "anthropic"
    LLM_MODEL: str = "claude-sonnet-4-20250514"
    ANTHROPIC_API_KEY: Optional[str] = None

    # ── RAG Search ──
    TOP_K_PER_COLLECTION: int = 5
    SCORE_THRESHOLD: float = 0.4

    # Collection search weights (must sum to 1.0)
    WEIGHT_LITERATURE: float = 0.30
    WEIGHT_TRIALS: float = 0.25
    WEIGHT_CONSTRUCTS: float = 0.20
    WEIGHT_ASSAYS: float = 0.15
    WEIGHT_MANUFACTURING: float = 0.10

    # ── PubMed ──
    NCBI_API_KEY: Optional[str] = None  # Optional, increases rate limit
    PUBMED_MAX_RESULTS: int = 5000

    # ── ClinicalTrials.gov ──
    CT_GOV_BASE_URL: str = "https://clinicaltrials.gov/api/v2"

    model_config = {"env_prefix": "CART_", "env_file": ".env"}


settings = CARTSettings()
