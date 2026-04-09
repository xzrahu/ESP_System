import argparse
import json
import math
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean
from typing import Any

from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas import EvaluationDataset, evaluate
from ragas.run_config import RunConfig

with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore",
        message="Importing .* from 'ragas.metrics' is deprecated.*",
        category=DeprecationWarning,
    )
from ragas.metrics import (
        answer_correctness,
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )

from config.settings import settings
from repositories.file_repository import FileRepository
from services.llamaindex_bm25_retriever import build_nodes_from_documents
from services.llamaindex_bge_reranker_postprocessor import BGERerankerPostprocessor
from services.query_service import QueryService
from services.llamaindex_query_engine_service import (
    LlamaIndexQueryEngineService,
    QueryEngineResult,
)
from utils.markdown_utils import MarkDownUtils


METRIC_REGISTRY = {
    "answer_relevancy": answer_relevancy,
    "faithfulness": faithfulness,
    "context_precision": context_precision,
    "context_recall": context_recall,
    "answer_correctness": answer_correctness,
}

DEFAULT_METRICS = [
    "answer_relevancy",
    "faithfulness",
    "context_precision",
    "context_recall",
    "answer_correctness",
]
DEFAULT_QUERY_BACKEND = "query_service"


@dataclass
class RagasEvalSample:
    question: str
    reference: str
    reference_contexts: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class InMemoryLlamaIndexQueryService:
    """Evaluation-only fallback that builds a transient in-memory LlamaIndex."""

    def __init__(self):
        self._service = LlamaIndexQueryEngineService()
        self._query_engine = None

    def query(self, question: str) -> QueryEngineResult:
        normalized_question = (question or "").strip()
        if not normalized_question:
            raise ValueError("question must not be empty")

        response = self._get_query_engine().query(normalized_question)
        documents = self._service._source_nodes_to_documents(
            getattr(response, "source_nodes", [])
        )

        if not documents:
            return QueryEngineResult(
                answer="当前知识库中暂时没有找到该问题的解决方案。",
                documents=[],
            )

        answer = str(getattr(response, "response", "") or response).strip()
        if not answer:
            answer = "当前知识库中暂时没有找到该问题的解决方案。"

        return QueryEngineResult(answer=answer, documents=documents)

    def retrieve(self, question: str) -> list[Document]:
        return self.query(question).documents

    def retrieval(self, question: str) -> list[Document]:
        return self.retrieve(question)

    def _get_query_engine(self):
        if self._query_engine is None:
            from llama_index.core import Document as LlamaDocument
            from llama_index.core import VectorStoreIndex
            from llama_index.core.query_engine import RetrieverQueryEngine
            from llama_index.core.retrievers import QueryFusionRetriever
            from llama_index.core.retrievers.fusion_retriever import FUSION_MODES
            from llama_index.retrievers.bm25 import BM25Retriever

            source_documents = self._load_documents()
            documents = [
                LlamaDocument(text=document.page_content, metadata=document.metadata)
                for document in source_documents
            ]
            index = VectorStoreIndex.from_documents(
                documents,
                embed_model=self._service._create_embedding_model(),
            )

            vector_retriever = index.as_retriever(
                similarity_top_k=max(settings.TOP_FINAL, min(settings.TOP_ROUGH, 10)),
            )
            bm25_retriever = BM25Retriever.from_defaults(
                nodes=build_nodes_from_documents(source_documents),
                similarity_top_k=max(settings.TOP_FINAL, min(settings.TOP_ROUGH, 10)),
                skip_stemming=True,
            )
            hybrid_retriever = QueryFusionRetriever(
                retrievers=[bm25_retriever, vector_retriever],
                llm=None,
                mode=FUSION_MODES.RECIPROCAL_RANK,
                similarity_top_k=settings.TOP_FINAL,
                num_queries=1,
                use_async=False,
                retriever_weights=[0.35, 0.65],
            )

            self._query_engine = RetrieverQueryEngine.from_args(
                retriever=hybrid_retriever,
                llm=self._service._create_llm(),
                node_postprocessors=self._get_node_postprocessors(),
            )
        return self._query_engine

    @staticmethod
    def _get_node_postprocessors():
        if not settings.ENABLE_BGE_RERANKER:
            return None

        return [
            BGERerankerPostprocessor(
                model_name=settings.BGE_RERANKER_MODEL,
                top_n=settings.BGE_RERANKER_TOP_N,
                api_url=settings.BGE_RERANKER_API_URL,
                api_key=settings.API_KEY,
            )
        ]

    @staticmethod
    def _load_documents():
        file_repository = FileRepository()
        file_paths = file_repository.list_files(settings.CRAWL_OUTPUT_DIR)
        unique_file_paths = file_repository.remove_duplicate_files(file_paths)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=200,
            separators=["\n##", "\n**", "\n\n", "\n", " ", ""],
        )

        all_documents = []
        for md_path in unique_file_paths:
            documents = TextLoader(file_path=md_path, encoding="utf-8").load()

            for document in documents:
                document.metadata["title"] = MarkDownUtils.extract_title(md_path)

            final_document_chunks = []
            for document in documents:
                if len(document.page_content) < 3000:
                    final_document_chunks.append(document)
                    continue

                document_chunks = splitter.split_documents([document])
                for document_chunk in document_chunks:
                    title = Path(document_chunk.metadata["source"]).name
                    document_chunk.page_content = f"文档来源:{title}\n{document_chunk.page_content}"
                final_document_chunks.extend(document_chunks)

            clean_document_chunks = filter_complex_metadata(final_document_chunks)
            valid_document_chunks = [
                document
                for document in clean_document_chunks
                if document.page_content.strip()
            ]
            all_documents.extend(valid_document_chunks)

        return all_documents


def load_dataset(dataset_path: Path) -> list[RagasEvalSample]:
    samples: list[RagasEvalSample] = []

    with dataset_path.open("r", encoding="utf-8") as file:
        for line_number, raw_line in enumerate(file, start=1):
            line = raw_line.strip()
            if not line:
                continue

            payload = json.loads(line)
            question = str(payload.get("question") or "").strip()
            reference = str(payload.get("reference") or "").strip()
            reference_contexts = [
                str(item).strip()
                for item in payload.get("reference_contexts", [])
                if str(item).strip()
            ]
            metadata = payload.get("metadata") or {}

            if not question:
                raise ValueError(f"Line {line_number} is missing 'question'.")
            if not reference:
                raise ValueError(f"Line {line_number} is missing 'reference'.")
            if not isinstance(metadata, dict):
                raise ValueError(f"Line {line_number} has non-object 'metadata'.")

            samples.append(
                RagasEvalSample(
                    question=question,
                    reference=reference,
                    reference_contexts=reference_contexts,
                    metadata=metadata,
                )
            )

    if not samples:
        raise ValueError("Dataset is empty.")

    return samples


def parse_metrics(metrics: str | None) -> list[str]:
    if not metrics:
        return list(DEFAULT_METRICS)

    metric_names = [item.strip() for item in metrics.split(",") if item.strip()]
    unknown_metrics = [name for name in metric_names if name not in METRIC_REGISTRY]
    if unknown_metrics:
        raise ValueError(f"Unsupported metrics: {', '.join(unknown_metrics)}")

    return metric_names


def build_eval_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model_name=settings.MODEL,
        openai_api_key=settings.API_KEY,
        openai_api_base=settings.BASE_URL,
        temperature=0,
    )


def build_eval_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model=settings.EMBEDDING_MODEL or "text-embedding-3-large",
        openai_api_key=settings.API_KEY,
        openai_api_base=settings.BASE_URL,
    )


def build_prepared_records(samples: list[RagasEvalSample]) -> list[dict[str, Any]]:
    return build_prepared_records_with_backend(samples=samples, query_backend=DEFAULT_QUERY_BACKEND)


def build_query_runner(query_backend: str):
    if query_backend == "query_service":
        return QueryService()
    if query_backend == "in_memory_llamaindex":
        return InMemoryLlamaIndexQueryService()
    raise ValueError(f"Unsupported query backend: {query_backend}")


def build_prepared_records_with_backend(
    samples: list[RagasEvalSample],
    query_backend: str,
) -> list[dict[str, Any]]:
    query_runner = build_query_runner(query_backend)
    prepared_records: list[dict[str, Any]] = []

    for sample in samples:
        query_result = query_runner.query(sample.question)
        retrieved_contexts = [document.page_content for document in query_result.documents]
        retrieved_titles = [document.metadata.get("title") for document in query_result.documents]
        retrieved_paths = [document.metadata.get("path") for document in query_result.documents]

        prepared_records.append(
            {
                "user_input": sample.question,
                "response": query_result.answer,
                "retrieved_contexts": retrieved_contexts,
                "reference": sample.reference,
                "reference_contexts": sample.reference_contexts,
                "metadata": sample.metadata,
                "retrieved_titles": retrieved_titles,
                "retrieved_paths": retrieved_paths,
                "query_backend": query_backend,
            }
        )

    return prepared_records


def build_ragas_dataset(records: list[dict[str, Any]]) -> EvaluationDataset:
    dataset_rows = [
        {
            "user_input": record["user_input"],
            "response": record["response"],
            "retrieved_contexts": record["retrieved_contexts"],
            "reference": record["reference"],
            "reference_contexts": record["reference_contexts"],
        }
        for record in records
    ]
    return EvaluationDataset.from_list(dataset_rows)


def aggregate_scores(records: list[dict[str, Any]], metric_names: list[str]) -> dict[str, float | None]:
    aggregates: dict[str, float | None] = {}

    for metric_name in metric_names:
        values = [
            float(record[metric_name])
            for record in records
            if record.get(metric_name) is not None
        ]
        aggregates[metric_name] = mean(values) if values else None

    return aggregates


def sanitize_for_json(value: Any) -> Any:
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, dict):
        return {key: sanitize_for_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize_for_json(item) for item in value]
    return value


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run ragas evaluation for the knowledge service.")
    parser.add_argument(
        "--dataset",
        required=True,
        help="Path to the JSONL ragas dataset.",
    )
    parser.add_argument(
        "--output",
        default="evaluation/last_ragas_result.json",
        help="Where to write the evaluation result JSON.",
    )
    parser.add_argument(
        "--metrics",
        default=",".join(DEFAULT_METRICS),
        help="Comma-separated ragas metrics to run.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Optional ragas batch size.",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Only prepare query/contexts without calling ragas.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap on the number of samples to evaluate.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="ragas per-task timeout in seconds.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="ragas worker parallelism.",
    )
    parser.add_argument(
        "--query-backend",
        default=DEFAULT_QUERY_BACKEND,
        choices=["query_service", "in_memory_llamaindex"],
        help="How to execute queries when preparing ragas records.",
    )
    return parser


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()

    metric_names = parse_metrics(args.metrics)
    dataset_path = Path(args.dataset)
    output_path = Path(args.output)

    samples = load_dataset(dataset_path)
    if args.max_samples is not None:
        samples = samples[: args.max_samples]
        if not samples:
            raise ValueError("No samples left after applying --max-samples.")

    prepared_records = build_prepared_records_with_backend(
        samples=samples,
        query_backend=args.query_backend,
    )

    payload: dict[str, Any] = {
        "summary": {
            "dataset": str(dataset_path.as_posix()),
            "total_samples": len(samples),
            "metrics": metric_names,
            "prepare_only": bool(args.prepare_only),
            "query_backend": args.query_backend,
        },
        "records": prepared_records,
    }

    if not args.prepare_only:
        ragas_dataset = build_ragas_dataset(prepared_records)
        result = evaluate(
            dataset=ragas_dataset,
            metrics=[METRIC_REGISTRY[name] for name in metric_names],
            llm=build_eval_llm(),
            embeddings=build_eval_embeddings(),
            raise_exceptions=False,
            show_progress=True,
            batch_size=args.batch_size,
            run_config=RunConfig(timeout=args.timeout, max_workers=args.max_workers),
        )

        scored_records = []
        for prepared_record, score_record in zip(prepared_records, result.scores, strict=True):
            merged = dict(prepared_record)
            merged.update(score_record)
            scored_records.append(merged)

        payload["summary"]["aggregate_metrics"] = aggregate_scores(scored_records, metric_names)
        payload["records"] = scored_records

    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = sanitize_for_json(payload)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    print(f"Detailed result written to: {output_path.as_posix()}")


if __name__ == "__main__":
    main()
