# Course presentation results

The presented study used the November 2026 SysCon crawl snapshot, with 301 URLs fetched, 271 documents in the corpus, and 8,248 indexed chunks (683 core and 7,565 research). The gold set contained 25 questions. The 20 answerable questions covered nine categories and had 87 annotated key facts. The five unanswerable questions tested abstention. [Crawl and index, slides 5–8](../report/presentation.pdf#page=5) · [Evaluation setup, slide 12](../report/presentation.pdf#page=12)

## Retrieval ablation

The ablation used the 20 answerable questions on CPU. Hit@k measures how often a gold source page appears in the first k results. MRR@10 averages the reciprocal rank of the first gold page in the first ten results, with zero for a miss.

| Configuration | Hit@1 | Hit@5 | MRR@10 | Seconds per question |
| --- | ---: | ---: | ---: | ---: |
| A. Dense | 70% | 90% | 0.797 | 0.05 |
| B. BM25 | 75% | 100% | 0.839 | 0.01 |
| C. Hybrid with reciprocal rank fusion | 75% | 90% | 0.838 | 0.05 |
| D. Hybrid with cross-encoder reranking | 95% | 100% | 0.975 | 19.80 |

Adding the cross-encoder to hybrid retrieval improved Hit@1 by 20 percentage points and brought CPU retrieval time to 19.80 seconds per question. [Slides 13–14](../report/presentation.pdf#page=13)

## Answer quality and abstention

All three runs used configuration D and Llama 3.1 8B through Ollama. An answer was complete if it included every annotated key fact. Overall correctness was reported across the 25-question set. [Answer quality, slide 15](../report/presentation.pdf#page=15)

| Run | Complete answers | Key-fact recall | Overall correct |
| --- | ---: | ---: | ---: |
| 1. Baseline | 75% | 84.2% | 80% |
| 2. After fixes | 90% | 94.2% | 92% |
| 3. Repeat of run 2 | 85% | 92.5% | 88% |

The changes after run 1 stopped expanding “how many” questions to whole pages, strengthened the prompt for numbers, dates and lists, and raised the abstention threshold from 0.2 to 0.6. The slides report 100% citation accuracy and gold-page coverage in every run. All five unanswerable questions were declined in every run, and no answerable question was declined. In run 2, the mean top reranker score was 0.97 for answerable questions and 0.13 for unanswerable ones. [Slides 15–16](../report/presentation.pdf#page=15)

## Latency and errors

In run 2, mean CPU time was 71.1 seconds for answerable questions (59 seconds median) and 25.4 seconds for unanswerable questions. The two incomplete answers omitted part of the SC 625 outline and an academic-year update, despite both having the relevant passage in context. Results reflect this crawl snapshot and the five abstention cases in the gold set. [Slide 17](../report/presentation.pdf#page=17)

For commands and the development fixture shipped with the current implementation, see the [evaluation guide](README.md).
