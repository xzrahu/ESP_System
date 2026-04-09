import os


def _default_vector_store_uri(project_root: str) -> str:
    explicit_uri = os.environ.get("VECTOR_STORE_URI")
    if explicit_uri:
        return explicit_uri

    legacy_path = os.environ.get("VECTOR_STORE_PATH")
    if legacy_path:
        if legacy_path.lower().endswith(".db"):
            return legacy_path
        return os.path.join(legacy_path, "milvus.db")

    return os.path.join(project_root, "milvus_kb1.db")


def _default_bm25_index_name(collection_name: str) -> str:
    normalized = (collection_name or "its-knowledge").strip().lower()
    return f"{normalized}-bm25"

try:
    from pydantic_settings import BaseSettings, SettingsConfigDict
except ImportError:
    class BaseSettings:
        def __init__(self, *args, **kwargs):
            pass

    def SettingsConfigDict(**kwargs):
        return kwargs


class Settings(BaseSettings):
    API_KEY: str = os.environ.get("API_KEY")
    BASE_URL: str = os.environ.get("BASE_URL")
    MODEL: str = os.environ.get("MODEL")
    EMBEDDING_MODEL: str = os.environ.get("EMBEDDING_MODEL")
    BGE_RERANKER_MODEL: str = os.environ.get(
        "BGE_RERANKER_MODEL",
        "BAAI/bge-reranker-v2-m3",
    )
    BGE_RERANKER_API_URL: str = os.environ.get(
        "BGE_RERANKER_API_URL",
        f"{os.environ.get('BASE_URL', '').rstrip('/')}/rerank" if os.environ.get("BASE_URL") else "",
    )
    ENABLE_BGE_RERANKER: bool = os.environ.get(
        "ENABLE_BGE_RERANKER",
        "true",
    ).lower() in {"1", "true", "yes", "on"}
    BGE_RERANKER_TOP_N: int = int(os.environ.get("BGE_RERANKER_TOP_N", "3"))
    BGE_RERANKER_USE_FP16: bool = os.environ.get(
        "BGE_RERANKER_USE_FP16",
        "false",
    ).lower() in {"1", "true", "yes", "on"}

    KNOWLEDGE_BASE_URL: str = os.environ.get("KNOWLEDGE_BASE_URL")

    _current_dir = os.path.dirname(os.path.abspath(__file__))
    _project_root = os.path.dirname(_current_dir)

    VECTOR_STORE_URI: str = _default_vector_store_uri(_project_root)
    VECTOR_STORE_TOKEN: str = os.environ.get("VECTOR_STORE_TOKEN", "")
    VECTOR_STORE_DIM: int = int(os.environ.get("VECTOR_STORE_DIM", "0"))
    VECTOR_STORE_SIMILARITY_METRIC: str = os.environ.get(
        "VECTOR_STORE_SIMILARITY_METRIC",
        "COSINE",
    )
    VECTOR_STORE_COLLECTION_NAME: str = os.environ.get(
        "VECTOR_STORE_COLLECTION_NAME",
        "its-knowledge",
    )

    CRAWL_OUTPUT_DIR: str = os.path.join(_project_root, "data", "crawl")
    MD_FOLDER_PATH: str = CRAWL_OUTPUT_DIR
    TMP_MD_FOLDER_PATH: str = os.path.join(_project_root, "data", "tmp")
    BM25_STORAGE_DIR: str = os.path.join(_project_root, "data", "bm25")
    BM25_ELASTICSEARCH_URL: str = os.environ.get(
        "BM25_ELASTICSEARCH_URL",
        "http://localhost:9200",
    )
    BM25_ELASTICSEARCH_USERNAME: str = os.environ.get(
        "BM25_ELASTICSEARCH_USERNAME",
        "",
    )
    BM25_ELASTICSEARCH_PASSWORD: str = os.environ.get(
        "BM25_ELASTICSEARCH_PASSWORD",
        "",
    )
    BM25_ELASTICSEARCH_API_KEY: str = os.environ.get(
        "BM25_ELASTICSEARCH_API_KEY",
        "",
    )
    BM25_ELASTICSEARCH_CA_CERTS: str = os.environ.get(
        "BM25_ELASTICSEARCH_CA_CERTS",
        "",
    )
    BM25_ELASTICSEARCH_VERIFY_CERTS: bool = os.environ.get(
        "BM25_ELASTICSEARCH_VERIFY_CERTS",
        "true",
    ).lower() in {"1", "true", "yes", "on"}
    BM25_ELASTICSEARCH_TIMEOUT: int = int(
        os.environ.get("BM25_ELASTICSEARCH_TIMEOUT", "30")
    )
    BM25_ELASTICSEARCH_INDEX: str = os.environ.get(
        "BM25_ELASTICSEARCH_INDEX",
        _default_bm25_index_name(VECTOR_STORE_COLLECTION_NAME),
    )

    CHUNK_SIZE: int = 3000
    CHUNK_OVERLAP: int = 200

    TOP_ROUGH: int = 50
    TOP_FINAL: int = 5

    ENABLE_RAG_PROMPT_COMPRESSION: bool = True
    LLMLINGUA_MODEL_NAME: str | None = os.environ.get("LLMLINGUA_MODEL_NAME")
    LLMLINGUA_RATE: float = 0.55
    LLMLINGUA_DYNAMIC_CONTEXT_COMPRESSION_RATIO: float = 0.3
    LLMLINGUA_CONTEXT_BUDGET: str = "+100"
    LLMLINGUA_REORDER_CONTEXT: str = "sort"
    LLMLINGUA_CONDITION_IN_QUESTION: str = "after_condition"
    LLMLINGUA_CONDITION_COMPARE: bool = True
    LLMLINGUA_RANK_METHOD: str = "longllmlingua"

    model_config = SettingsConfigDict(
        env_file=os.path.join(_project_root, ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
