import logging
from typing import List

from langchain_core.documents import Document
from langchain_milvus import Milvus
from langchain_openai.embeddings import OpenAIEmbeddings

from config.settings import settings
from utils.vector_store import (
    get_vector_store_collection_name,
    get_vector_store_connection_args,
    get_vector_store_embedding_model_name,
)


logger = logging.getLogger(__name__)


class VectorStoreRepository:
    def __init__(self):
        self.embedding = OpenAIEmbeddings(
            model=get_vector_store_embedding_model_name(),
            openai_api_key=settings.API_KEY,
            openai_api_base=settings.BASE_URL,
        )
        self.database = Milvus(
            collection_name=get_vector_store_collection_name(),
            connection_args=get_vector_store_connection_args(),
            embedding_function=self.embedding,
            auto_id=True,
            enable_dynamic_field=True,
            consistency_level="Session",
            index_params={
                "index_type": "FLAT",
                "metric_type": settings.VECTOR_STORE_SIMILARITY_METRIC,
                "params": {},
            },
            search_params={
                "metric_type": settings.VECTOR_STORE_SIMILARITY_METRIC,
                "params": {},
            },
        )

    def add_documents(self, documents: list[Document], batch_size: int = 16) -> int:
        total_documents = len(documents)
        documents_added = 0

        try:
            for start in range(0, total_documents, batch_size):
                batch = documents[start : start + batch_size]
                self.database.add_documents(batch)
                documents_added += len(batch)
                logger.info(
                    "Saved document chunks to vector store: %s/%s",
                    documents_added,
                    total_documents,
                )
        except Exception:
            logger.exception("Failed to save document chunks into the vector store.")
            raise

        return documents_added

    def embedd_document(self, text: str) -> List[float]:
        return self.embedding.embed_query(text)

    def embedd_documents(self, texts: list[str]) -> List[List[float]]:
        return self.embedding.embed_documents(texts)

    def search_similarity_with_score(
        self,
        user_question: str,
        top_k: int = 5,
    ) -> list[tuple[Document, float]]:
        return self.database.similarity_search_with_score(user_question, k=top_k)
