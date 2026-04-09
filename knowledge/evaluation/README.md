# Knowledge Evaluation

This directory contains two evaluation flows for the knowledge service.

## 1. Retrieval Metrics

Use the existing retrieval-only script when you want to measure recall quality of the
retriever itself.

Supported metrics:

- `Precision@k`
- `Recall@k`
- `HitRate@k`
- `MRR@k`
- `NDCG@k`

Run from `its_multi_agent/backend/knowledge`:

```powershell
python -m evaluation.eval_rag --dataset evaluation/sample_eval_dataset.jsonl
```

## 2. RAGAS Answer Quality Evaluation

Use the `ragas` script when you want to evaluate the final answer produced by the
knowledge service together with the retrieved contexts.

Default metrics:

- `answer_relevancy`
- `faithfulness`
- `context_precision`
- `context_recall`
- `answer_correctness`

### Dataset Format

Use JSONL with one sample per line.

Required fields:

- `question`: user question
- `reference`: reference / ground-truth answer

Optional fields:

- `reference_contexts`: reference contexts for inspection only
- `metadata`: any extra tags you want to keep with the sample

Example:

```json
{
  "question": "开机蓝屏怎么办？",
  "reference": "可先记录蓝屏报错信息，进入安全模式排查驱动或最近安装的软件，并检查硬件连接情况。",
  "reference_contexts": [
    "蓝屏故障通常需要先记录错误代码，再按驱动、系统更新、硬件连接等方向排查。"
  ],
  "metadata": {
    "topic": "blue_screen"
  }
}
```

### Dry Run

Build the evaluation records without calling `ragas`.

```powershell
python -m evaluation.eval_ragas --dataset evaluation/sample_ragas_dataset.jsonl --prepare-only
```

### Full Evaluation

Run `ragas` scoring and write the result JSON.

```powershell
python -m evaluation.eval_ragas --dataset evaluation/sample_ragas_dataset.jsonl --output evaluation/last_ragas_result.json
```

Optional arguments:

- `--metrics answer_relevancy,faithfulness`
- `--batch-size 4`
- `--max-samples 5`
- `--timeout 300`
- `--max-workers 4`
- `--prepare-only`

## Notes

- The `ragas` flow evaluates final answer quality, not just retrieval quality.
- The knowledge base collection must contain indexed documents before evaluation, otherwise
  the query service will return fallback answers and the scores will not be meaningful.
- The `ragas` metrics use the configured LLM / embedding endpoints, so running the full
  evaluation may consume API quota.
