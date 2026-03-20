# app/config/config.py
"""
Centralized application configuration.

Architecture
------------
This module uses Pydantic Settings to provide a single, validated, typed
configuration object for the entire application.

How It Works
------------
1. Pydantic's BaseSettings reads values from environment variables
2. load_dotenv() is used to populate os.environ from .env files BEFORE
   instantiating Settings
3. Settings fields have defaults; env vars override them

Environment Files
-----------------
    .env                 → Base settings (mainly APP_ENV)
    .env.development     → Development overrides
    .env.staging         → Staging overrides  
    .env.production      → Production overrides

Value Resolution (lowest to highest priority)
---------------------------------------------
    1. Field default in Settings class
    2. Values from .env file
    3. Values from .env.{APP_ENV} file
    4. Actual OS environment variables

Usage
-----
    from app.config.config import settings
    
    db_url = settings.DB_SYNC_URL
    mt_backend = settings.MT_BACKEND

Important: Most configuration (except REQUIRED ENVIRONMENT config) should be accessed via `settings`, not via
os.environ.get(). This ensures type safety, validation, and consistency.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.utils.logger import logger



# ==========================================
# Path Constants (not environment-dependent)
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # repo root
OUTPUT_ROOT: Path = PROJECT_ROOT / "output"
OUTPUT_TEMP: Path = PROJECT_ROOT / "_output_temp"
PADDLEX_CONFIG_PATH = PROJECT_ROOT / "app/config/PP-StructureV3.yaml"



# ===================
# Environment Loading
# ===================

# Step 1: Load base .env to determine APP_ENV
BASE_ENV_PATH = Path(".env")
if BASE_ENV_PATH.exists():
    load_dotenv(BASE_ENV_PATH)
else:
    logger.warning("[config] .env not found; using APP_ENV=development")

# Step 2: Determine environment
APP_ENV = os.getenv("APP_ENV", "development").lower()
if APP_ENV not in {"development", "staging", "production"}:
    raise ValueError(f"Invalid APP_ENV: {APP_ENV}")

# Step 3: Load environment-specific overrides
ENV_FILES = {
    "development": ".env.development",
    "staging": ".env.staging",
    "production": ".env.production",
}
specific_env_file = ENV_FILES.get(APP_ENV)
if specific_env_file and os.path.exists(specific_env_file):
    load_dotenv(specific_env_file, override=True)
else:
    logger.warning(f"[config] {specific_env_file} not found; using defaults")



# ======================================
# Settings Model (Pydantic BaseSettings)
# ======================================

class Settings(BaseSettings):
    """
    Application settings with type validation.
    
    All configurable values should be defined here. Access via:
        from app.config.config import settings
        settings.FIELD_NAME
    """
    
    model_config = SettingsConfigDict(
        env_prefix="", # environment vars match field names exactly
        extra="allow", # allow other environment vars
        case_sensitive=False # appropriaate for .env files
        )
    
    # Application
    APP_ENV: str = APP_ENV
    PROJECT_NAME: str = "Legal AI Core"

    # The foollowing Should not be used because deprecated in Pydantic 2
    # class Config:
    #     env_file = None

    # =====================================================================
    # REQUIRED ENVIRONMENT config (typically accessed via os.environ.get())
    # They must/should exist in the environment, bacause they are supposed to be invoked as such (default value could even be omitted if they are critical ('must'), so that startup would fail if missing).
    # ======================================================================

    # Database
    DB_ASYNC_URL: str = Field("postgresql+asyncpg://admin:secret@localhost:5432/cm3070_ai_orchestration")
    DB_SYNC_URL: str = Field("postgresql+psycopg://admin:secret@localhost:5432/cm3070_ai_orchestration")

    # Paddle configs (not critical)
    PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK: bool = True



    # ===============================================================
    # OPTIONAL ENVIRONMENT config (should be accessed via `settings`)
    # (May exist in, and be overriden by, OS/.env)
    # ===============================================================

    # Redis / Job Queue
    REDIS_URL: str = "redis://localhost:6379/0"
    RQ_QUEUE_NAME: str = "pipeline"
    RQ_JOB_TIMEOUT_SECONDS: int = 3600

    # File storage
    STORAGE_PATH: str = "./data/uploads"

    # Security / API Key (default for dev only)
    API_KEY: str = "local-api-key"



    # ==============================
    # APP DEFAULTS / internal config
    # (Not expected to exist in, and be overriden by, OS/.env
    # ==============================

    # Machine Translation - Backend Selection
    MT_BACKEND: str = "gemma"  # "ct2" | "gemma"
    MT_BATCH_SIZE: int = 6

    # Machine Translation - CTranslate2 Backend
    MT_CT2_MODEL_BASE_DIR: Path = PROJECT_ROOT / "models"
    MT_CT2_DEVICE: str = "cuda"  # "cuda" | "cpu"
    MT_CT2_COMPUTE: str = "int8_float16"  # "default" | "float16" | "int8" | "int8_float16"
    
    # Machine Translation - HuggingFace Backend
    MT_HF_DEVICE: str = "auto"  # "auto" | "cpu" | GPU index as string
    

    # Machine Translation - TranslateGemma Backend
    MT_GEMMA_MODEL: str = "google/translategemma-4b-it" # "google/translategemma-4b-it" | "google/translategemma-12b-it"
    MT_GEMMA_DEVICE_MAP: str = "auto"  # "auto" | "cuda:0" | "cpu"
    MT_GEMMA_LOAD_IN_4BIT: bool = False  # Use 4-bit quantization
    MT_GEMMA_LOAD_IN_8BIT: bool = False  # Use 8-bit quantization
    MT_GEMMA_MAX_NEW_TOKENS: int = 1024  # Max tokens to generate
    
    # Machine Translation - Tuning & Debugging (These allow runtime override of token limits without code changes. Set to None/0 to use automatic model-based detection.)
    MT_MAX_TOKENS_OVERRIDE: int | None = None
    MT_CHARS_PER_TOKEN_OVERRIDE: float | None = None
    DEBUG_MT_CHUNKS: bool = False  # Set True to log chunk validation warnings
    

    # Language detection (S2)
    # Comma-separated ISO 639-1 codes used as candidates by Lingua + Tesseract.
    CANDIDATE_LANGS: list[str] = ["en", "fr", "nl", "it", "es", "uk", "ru"]


    # ------------------------------------------
    # Validators: enforce required settings in production (Pydantic v2)
    # ---------------------------------------------
    @field_validator("DB_SYNC_URL")
    def ensure_production_sync_db_not_local(cls, v):
        """Prevent dangerous misconfigurations."""
        if APP_ENV == "production" and "localhost" in v:
            raise ValueError(
                "Production DB_SYNC_URL cannot point to localhost. "
                "Update .env.production with a real database host."
            )
        return v
    

    @field_validator("DB_ASYNC_URL")
    def ensure_production_async_db_not_local(cls, v):
        """Prevent dangerous misconfigurations."""
        if APP_ENV == "production" and "localhost" in v:
            raise ValueError(
                "Production DB_ASYNC_URL cannot point to localhost. "
                "Update .env.production with a real database host."
            )
        return v


    @field_validator("MT_BACKEND")
    @classmethod
    def validate_mt_backend(cls, v):
        """Ensure MT_BACKEND is a valid option."""
        valid = {"ct2", "hf", "vllm", "gemma"}
        if v.lower() not in valid:
            raise ValueError(f"MT_BACKEND must be one of {valid}, got {v!r}")
        return v.lower()



# ==================
# Singleton Instance
# ==================

settings = Settings()