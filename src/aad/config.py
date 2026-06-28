"""Application configuration via environment variables."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """AAD settings loaded from .env file and environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com/v1"
    deepseek_chat_model: str = "deepseek-chat"
    local_embedding_model: str = "BAAI/bge-small-zh-v1.5"
    embedding_dim: int = 512
    store_path: str = "data/aad_store.jsonl"
