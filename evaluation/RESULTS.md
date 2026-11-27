# SysBot evaluation results

The evaluation used the saved SysCon index in `data/processed/` with 8,248 chunks, including 683 core chunks. The run used the Intel Core i5-10310U laptop CPU, BAAI/bge-base-en-v1.5 embeddings, and BAAI/bge-reranker-v2-m3. Latency was measured while the offline index rebuild was also running.

## Retrieval ablation

All 50 answerable questions use the default core tier. A hit means that the reference page appeared among the top five returned chunks. MRR@5 is the mean reciprocal rank of the first matching result, with zero for a miss.

| Configuration | Hits | Hit@5 | MRR@5 | Median retrieval |
|---|---:|---:|---:|---:|
| Dense | 47/50 | 94% | 0.902 | 0.294 s |
| BM25 | 48/50 | 96% | 0.905 | 0.070 s |
| Hybrid RRF | 49/50 | 98% | 0.930 | 0.348 s |
| Hybrid RRF and cross-encoder | 50/50 | 100% | 0.990 | 83.66 s |

Dense, BM25, and hybrid RRF missed the course catalogue on the lab-courses question. The cross-encoder put it first. A source-level hit identifies a relevant page. Passage-level support is scored separately below.

The original run saved four per-question ranked lists and a summary under the local `data/evaluation/retrieval-ablation-20261121/` directory.

## End-to-end answers and abstention

The CPU answer sample covered all five answerable categories and two unanswerable questions. Ollama ran Llama 3.1 8B (Q4_K_M) with an 8,192-token context and reported 100% CPU use. Each question used the normal streaming chat path.

| Question | First word | Full reply | Correctness | Completeness | Faithfulness |
|---|---:|---:|---:|---:|---:|
| Ph.D. application deadline | 423.2 s | 442.4 s | 1 | 1 | 1 |
| SC 605 prerequisites | 363.7 s | 389.4 s | 1 | 1 | 1 |
| NMR spectroscopy faculty | 327.2 s | 338.9 s | 1 | 1 | 1 |
| IDDDP leave form and recommendation | 330.0 s | 362.3 s | 0.5 | 0.5 | 0.5 |
| Main office contact | 294.3 s | 320.4 s | 1 | 1 | 0 |
| Saturday office hours | 77.4 s | 77.4 s | — | — | — |
| Future SC 601 exam questions | 86.6 s | 86.6 s | — | — | — |

Across the five graded answers, mean correctness was **0.90**, completeness was **0.90**, and cited faithfulness was **0.70**. The leave answer cited an admission requirement for the leave recommendation. The main office contact answer gave matching details but omitted inline citations. Both sampled unanswerable questions elicited explicit refusals without guesses.

The median across all seven cases was **327.2 s** to the first word and **338.9 s** to the full reply. Across answerable cases it was **330.0 s** and **362.3 s**, respectively. These measurements describe the sampled queries and this workload. The 60-question set remains available for a broader run.

The original run saved answers, passages, and grades under the local `data/evaluation/cpu-answer-sample-20261121/` directory. The [evaluation guide](README.md) shows how to run the evaluation and score the results.
