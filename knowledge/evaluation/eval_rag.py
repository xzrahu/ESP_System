import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

from services.retrieval_service import RetrievalService
from evaluation.eval_ragas import InMemoryLlamaIndexQueryService


@dataclass
class EvalSample:
    question: str
    relevant_titles: set[str]
    relevant_paths: set[str]


def _normalize_text(value: str | None) -> str:
    return (value or "").strip().lower()


def _normalize_path(value: str | None) -> str:
    return str(Path(value).as_posix()).strip().lower() if value else ""


def load_dataset(dataset_path: Path) -> list[EvalSample]:
    samples: list[EvalSample] = []
    with dataset_path.open("r", encoding="utf-8") as file:
        for line_number, raw_line in enumerate(file, start=1):
            line = raw_line.strip()
            if not line:
                continue

            payload = json.loads(line)
            question = (payload.get("question") or "").strip()
            relevant_titles = {
                _normalize_text(item) for item in payload.get("relevant_titles", []) if item
            }
            relevant_paths = {
                _normalize_path(item) for item in payload.get("relevant_paths", []) if item
            }

            if not question:
                raise ValueError(f"Line {line_number} is missing 'question'.")
            if not relevant_titles and not relevant_paths:
                raise ValueError(
                    f"Line {line_number} must provide at least one of "
                    f"'relevant_titles' or 'relevant_paths'."
                )

            samples.append(
                EvalSample(
                    question=question,
                    relevant_titles=relevant_titles,
                    relevant_paths=relevant_paths,
                )
            )

    if not samples:
        raise ValueError("Dataset is empty.")

    return samples


def is_relevant(document: Document, sample: EvalSample) -> bool:
    title = _normalize_text(document.metadata.get("title"))
    path = _normalize_path(document.metadata.get("path"))
    return title in sample.relevant_titles or path in sample.relevant_paths


def _document_identity(document: Document) -> str:
    path = _normalize_path(document.metadata.get("path"))
    if path:
        return f"path:{path}"

    title = _normalize_text(document.metadata.get("title"))
    if title:
        return f"title:{title}"

    source = _normalize_path(document.metadata.get("source"))
    if source:
        return f"source:{source}"

    return f"content:{_normalize_text(document.page_content[:200])}"


def precision_at_k(matches: list[bool], top_k: int) -> float:
    if top_k <= 0:
        return 0.0
    return sum(matches[:top_k]) / top_k


def recall_at_k(matches: list[bool], total_relevant: int, top_k: int) -> float:
    if total_relevant <= 0:
        return 0.0
    return sum(matches[:top_k]) / total_relevant


def hit_rate_at_k(matches: list[bool], top_k: int) -> float:
    return 1.0 if any(matches[:top_k]) else 0.0


def mrr_at_k(matches: list[bool], top_k: int) -> float:
    for index, matched in enumerate(matches[:top_k], start=1):
        if matched:
            return 1.0 / index
    return 0.0


def ndcg_at_k(matches: list[bool], total_relevant: int, top_k: int) -> float:
    dcg = 0.0
    for index, matched in enumerate(matches[:top_k], start=1):
        if matched:
            dcg += 1.0 / math.log2(index + 1)

    ideal_hits = min(total_relevant, top_k)
    if ideal_hits == 0:
        return 0.0

    idcg = sum(1.0 / math.log2(index + 1) for index in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else 0.0


def evaluate_sample(
    retrieval_service: RetrievalService,
    sample: EvalSample,
    top_k: int,
) -> dict[str, Any]:
    raw_documents = retrieval_service.retrieval(sample.question)
    seen_identities: set[str] = set()
    documents: list[Document] = []
    for document in raw_documents:
        identity = _document_identity(document)
        if identity in seen_identities:
            continue
        seen_identities.add(identity)
        documents.append(document)

    matches = [is_relevant(document, sample) for document in documents]
    total_relevant = max(len(sample.relevant_titles), len(sample.relevant_paths), 1)

    return {
        "question": sample.question,
        "metrics": {
            f"precision@{top_k}": precision_at_k(matches, top_k),
            f"recall@{top_k}": recall_at_k(matches, total_relevant, top_k),
            f"hit_rate@{top_k}": hit_rate_at_k(matches, top_k),
            f"mrr@{top_k}": mrr_at_k(matches, top_k),
            f"ndcg@{top_k}": ndcg_at_k(matches, total_relevant, top_k),
        },
        "expected": {
            "relevant_titles": sorted(sample.relevant_titles),
            "relevant_paths": sorted(sample.relevant_paths),
        },
        "retrieved": [
            {
                "rank": rank,
                "title": document.metadata.get("title"),
                "path": document.metadata.get("path"),
                "matched": matched,
            }
            for rank, (document, matched) in enumerate(zip(documents, matches), start=1)
        ],
    }


def aggregate_results(results: list[dict[str, Any]], top_k: int) -> dict[str, float]:
    metric_names = [
        f"precision@{top_k}",
        f"recall@{top_k}",
        f"hit_rate@{top_k}",
        f"mrr@{top_k}",
        f"ndcg@{top_k}",
    ]
    return {
        name: sum(result["metrics"][name] for result in results) / len(results)
        for name in metric_names
    }


def build_retrieval_runner(query_backend: str):
    if query_backend == "retrieval_service":
        return RetrievalService()
    if query_backend == "in_memory_llamaindex":
        return InMemoryLlamaIndexQueryService()
    raise ValueError(f"Unsupported query backend: {query_backend}")


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run offline RAG retrieval evaluation.")
    parser.add_argument(
        "--dataset",
        required=True,
        help="Path to the JSONL evaluation dataset.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=1,
        help="Top-K used for evaluation metrics. Default is 1.",
    )
    parser.add_argument(
        "--output",
        default="evaluation/last_result.json",
        help="Where to write the evaluation result JSON.",
    )
    parser.add_argument(
        "--query-backend",
        default="retrieval_service",
        choices=["retrieval_service", "in_memory_llamaindex"],
        help="How to execute retrieval during evaluation.",
    )
    return parser


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    output_path = Path(args.output)

    samples = load_dataset(dataset_path)
    retrieval_service = build_retrieval_runner(args.query_backend)

    successful_results: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []

    for sample in samples:
        try:
            successful_results.append(
                evaluate_sample(retrieval_service=retrieval_service, sample=sample, top_k=args.top_k)
            )
        except Exception as exc:
            failures.append({"question": sample.question, "error": str(exc)})

    if not successful_results:
        raise RuntimeError("Evaluation failed for all samples.")

    summary = {
        "dataset": str(dataset_path.as_posix()),
        "top_k": args.top_k,
        "query_backend": args.query_backend,
        "total_samples": len(samples),
        "successful_samples": len(successful_results),
        "failed_samples": len(failures),
        "aggregate_metrics": aggregate_results(successful_results, args.top_k),
    }

    payload = {
        "summary": summary,
        "results": successful_results,
        "failures": failures,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Detailed result written to: {output_path.as_posix()}")


if __name__ == "__main__":
    main()
