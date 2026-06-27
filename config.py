from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # Meta WhatsApp configurations
    META_VERIFY_TOKEN: str
    META_ACCESS_TOKEN: str
    META_PHONE_NUMBER_ID: str
    
    # OpenAI / vLLM configurations
    OPENAI_BASE_URL: str = "https://10.101.7.72/v1"
    OPENAI_API_KEY: str
    OPENAI_MODEL_NAME: str = "meta-llama/Meta-Llama-3-8B-Instruct" # Fallback if not specified
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"
    
    # Qdrant configurations
    QDRANT_PATH: str = "local_qdrant_db"
    QDRANT_COLLECTION_NAME: str = "inventory"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

settings = Settings()
