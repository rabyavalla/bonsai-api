"""
B.O.N.S.A.I. API — Configuration
"""

from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # Pinecone
    pinecone_api_key: str = ""
    pinecone_index_name: str = "bonsai-health"
    pinecone_environment: str = "us-east-1"

    # OpenAI (embeddings)
    openai_api_key: str = ""

    # Anthropic (Claude)
    anthropic_api_key: str = ""

    # x402 Payment
    bonsai_wallet_address: str = ""
    base_rpc_url: str = "https://mainnet.base.org"
    usdc_contract_address: str = "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913"

    # API config
    api_key_header: str = "X-API-Key"
    rate_limit_per_minute: int = 100
    log_level: str = "INFO"

    # Dev mode
    skip_payment_verification: bool = False

    # Pricing (USDC)
    tier_1_price: float = 0.02
    tier_2_price: float = 0.20
    tier_3_price: float = 0.45
    tier_4_price: float = 0.85

    # LLM models
    haiku_model: str = "claude-haiku-4-5-20251001"
    sonnet_model: str = "claude-sonnet-4-5-20250929"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


@lru_cache()
def get_settings() -> Settings:
    return Settings()
