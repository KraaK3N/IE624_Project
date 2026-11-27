# Evaluate the current SysBot code

The course presentation used a 25-question gold set with 20 answerable and 5 unanswerable questions. The slides record the protocol and results. The full question set and run logs for that study are unavailable in this repository. [Course presentation](../report/presentation.pdf#page=12) · [Presented results](RESULTS.md)

The evaluator defaults to a separate [60-question development fixture](questions.jsonl) with 50 answerable and 10 unanswerable questions. Each answerable question includes a reference answer and source URL. Use `--questions PATH` to supply a different JSONL set in the same format.

Follow the [project setup](../README.md#quick-start) and set `VECTOR_STORE_PATH` and `CHUNKS_PATH` in `.env` to the build you want to evaluate. Run the commands from the repository root. Each run creates an output directory under `data/evaluation/` unless you pass `--output-dir`. Set `HF_HUB_OFFLINE=1` if both retrieval models are already cached and you want an offline run.

## Retrieval

```sh
.venv313/bin/python evaluate.py retrieval
```

The evaluator compares dense, BM25, hybrid reciprocal rank fusion, and hybrid with cross-encoder reranking. It searches the `core` tier on CPU for questions with reference URLs. It reports Hit@5 and MRR@5 over the ranked chunks by matching their source URLs. The presentation used page-level Hit@1, Hit@5, and MRR@10.

## Answers and latency

Configure your generation provider in `.env` before running this evaluation. The presentation used Llama 3.1 8B through Ollama on CPU. Its final two runs used a 0.6 abstention threshold. The current configuration defaults to 0.2 and supports overriding it with `MIN_RERANK_SCORE` in `.env`.

```sh
.venv313/bin/python evaluate.py answers
```

The evaluator records the streamed answer, retrieved context, and time to the first word and full response. Review the 10 unanswerable responses for appropriate abstention, including wording that differs from the standard message.

For manual review of the current code, grade each answerable response on a 0 to 1 scale. Enter one JSON object per line with the question `id` and these numeric fields. The presentation tracked key-fact recall and whether citations matched the annotated source pages.

| Field | Scoring |
|---|---|
| `correctness` | 1 for an accurate answer, 0.5 for mixed claims, 0 for an incorrect answer or abstention. |
| `completeness` | 1 for all reference facts, 0.5 for some, 0 for none. |
| `faithfulness` | 1 when the cited passages support all claims, 0.5 for some, 0 for none or abstention. |

```sh
.venv313/bin/python evaluate.py grade --answers data/evaluation/RUN/answers.jsonl --grades path/to/grades.jsonl
```
