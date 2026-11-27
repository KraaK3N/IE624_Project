# SysBot evaluation

The [question set](questions.jsonl) contains 50 answerable questions and 10 outside the saved SysCon corpus. Each answerable question includes a reference answer and source URL. Time-sensitive facts describe the saved snapshot. See the [recorded results](RESULTS.md) for the retrieval ablation and sampled answer evaluation.

Follow the [project setup](../README.md#quick-start) and set `VECTOR_STORE_PATH` and `CHUNKS_PATH` in `.env` to the build you want to evaluate. Run the commands from the repository root. Each run creates an output directory under `data/evaluation/` unless you pass `--output-dir`. Set `HF_HUB_OFFLINE=1` if both retrieval models are already cached and you want an offline run.

## Retrieval

```sh
.venv313/bin/python evaluate.py retrieval
```

The evaluator compares dense, BM25, hybrid reciprocal rank fusion, and hybrid with cross-encoder reranking. It uses the 50 answerable questions and searches the `core` tier on CPU. Hit@5 measures whether the reference URL appears in the top five chunks. MRR@5 measures how high the first matching URL ranks, with zero for a miss. These are source-level metrics.

## Answers and latency

Configure your generation provider in `.env` before running this evaluation. To replicate the report's Ollama setup, use `llama3.1:8b` and check `ollama ps` after the first request to confirm it is running on CPU.

```sh
.venv313/bin/python evaluate.py answers
```

The evaluator records the streamed answer, retrieved context, and time to the first word and full response. Review the 10 unanswerable responses for appropriate abstention, including wording that differs from the standard message.

Grade each answerable response on a 0 to 1 scale. Enter one JSON object per line with the question `id` and these numeric fields.

| Field | Scoring |
|---|---|
| `correctness` | 1 for an accurate answer, 0.5 for mixed claims, 0 for an incorrect answer or abstention. |
| `completeness` | 1 for all reference facts, 0.5 for some, 0 for none. |
| `faithfulness` | 1 when the cited passages support all claims, 0.5 for some, 0 for none or abstention. |

```sh
.venv313/bin/python evaluate.py grade --answers data/evaluation/RUN/answers.jsonl --grades path/to/grades.jsonl
```
