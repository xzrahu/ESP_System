from pathlib import Path
from urllib.parse import quote_plus

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing_extensions import Self


class Settings(BaseSettings):
    SF_API_KEY: str | None = Field(default=None, description="SiliconFlow API key")
    SF_BASE_URL: str | None = Field(default=None, description="SiliconFlow base URL")

    AL_BAILIAN_API_KEY: str | None = Field(default=None, description="Bailian API key")
    AL_BAILIAN_BASE_URL: str | None = Field(default=None, description="Bailian base URL")

    MAIN_MODEL_NAME: str | None = Field(default="Qwen/Qwen3-32B", description="Main chat model")
    SUB_MODEL_NAME: str | None = Field(default="", description="Sub chat model")

    MYSQL_HOST: str | None = Field(default="localhost", description="MySQL host")
    MYSQL_PORT: int = Field(default=3306, description="MySQL port")
    MYSQL_USER: str | None = Field(default="root", description="MySQL user")
    MYSQL_PASSWORD: str | None = Field(default="", description="MySQL password")
    MYSQL_DATABASE: str | None = Field(default="its_db", description="MySQL database")
    MYSQL_CHARSET: str = Field(default="utf8mb4", description="MySQL charset")
    MYSQL_CONNECT_TIMEOUT: int = Field(default=10, description="MySQL connect timeout")
    MYSQL_MAX_CONNECTIONS: int = Field(default=5, description="MySQL max connections")

    USE_POSTGRES_MEMORY: bool = Field(
        default=True,
        description="Prefer PostgreSQL for LangGraph checkpoint/store memory backend",
    )
    POSTGRES_URI: str | None = Field(default=None, description="PostgreSQL DSN")
    POSTGRES_HOST: str | None = Field(default=None, description="PostgreSQL host")
    POSTGRES_PORT: int = Field(default=5432, description="PostgreSQL port")
    POSTGRES_USER: str | None = Field(default=None, description="PostgreSQL user")
    POSTGRES_PASSWORD: str | None = Field(default=None, description="PostgreSQL password")
    POSTGRES_DATABASE: str | None = Field(default=None, description="PostgreSQL database")
    POSTGRES_SSLMODE: str | None = Field(default=None, description="PostgreSQL sslmode")
    POSTGRES_POOL_MIN_SIZE: int = Field(default=1, description="PostgreSQL pool min size")
    POSTGRES_POOL_MAX_SIZE: int = Field(default=5, description="PostgreSQL pool max size")
    POSTGRES_POOL_TIMEOUT: int = Field(default=30, description="PostgreSQL pool timeout")

    MEMORY_EMBEDDING_MODEL: str = Field(
        default="text-embedding-3-small",
        description="Embedding model used by long-term memory recall",
    )
    MEMORY_EMBEDDING_DIMS: int = Field(default=1536, description="Memory embedding dimensions")
    MEMORY_EMBEDDING_BASE_URL: str | None = Field(
        default=None,
        description="OpenAI-compatible base URL for memory embeddings",
    )
    MEMORY_EMBEDDING_API_KEY: str | None = Field(
        default=None,
        description="OpenAI-compatible API key for memory embeddings",
    )

    KNOWLEDGE_BASE_URL: str | None = Field(default=None, description="Knowledge base service URL")
    DASHSCOPE_BASE_URL: str | None = Field(default=None, description="DashScope base URL")
    DASHSCOPE_API_KEY: str | None = Field(default=None, description="DashScope API key")
    BAIDUMAP_AK: str | None = Field(default=None, description="Baidu map access key")

    model_config = SettingsConfigDict(
        env_file=str(Path(__file__).parent.parent / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
        validate_default=True,
    )

    @model_validator(mode="after")
    def check_ai_service_configuration(self) -> Self:
        has_service = any(
            [
                self.SF_API_KEY and self.SF_BASE_URL,
                self.AL_BAILIAN_API_KEY and self.AL_BAILIAN_BASE_URL,
            ]
        )
        if not has_service:
            raise ValueError("At least one AI provider must be configured.")
        return self

    @property
    def postgres_conn_string(self) -> str | None:
        if self.POSTGRES_URI:
            return self.POSTGRES_URI

        if not (self.POSTGRES_HOST and self.POSTGRES_USER and self.POSTGRES_DATABASE):
            return None

        auth = quote_plus(self.POSTGRES_USER)
        if self.POSTGRES_PASSWORD:
            auth = f"{auth}:{quote_plus(self.POSTGRES_PASSWORD)}"

        database = quote_plus(self.POSTGRES_DATABASE)
        dsn = f"postgresql://{auth}@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{database}"
        if self.POSTGRES_SSLMODE:
            dsn = f"{dsn}?sslmode={quote_plus(self.POSTGRES_SSLMODE)}"
        return dsn


settings = Settings()
