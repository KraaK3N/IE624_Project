"""Run SysBot's retrieval ablation or record end-to-end answers for review."""

import argparse
import json
import platform
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from time import perf_counter

from src.knowledge import source_identity

ROOT = Path(__file__).resolve().parent
DEFAULT_QUESTIONS = ROOT / "evaluation" / "questions.jsonl"
CONFIGURATIONS = {
    "dense": ("dense", False),
    "bm25": ("bm25", False),
    "hybrid": ("hybrid", False),
    "hybrid_rerank": ("hybrid", True),
}


@dataclass
class Grade:
    id: str
    # Score factual accuracy independently of how much of the reference answer was included.
    correctness: float
    completeness: float
    faithfulness: float


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def retrieve_result(item: dict, hits: list, seconds: float) -> dict:
    expected = {source_identity(url, legacy=True) for url in item["source_urls"]}
    urls = [hit.meta["url"] for hit in hits]
    rank = next((i for i, url in enumerate(urls, 1) if source_identity(url, legacy=True) in expected), None)
    return {
        "id": item["id"], "expected_urls": item["source_urls"], "retrieved_urls": urls,
        "rank": rank, "seconds": round(seconds, 3),
    }


def retrieval_summary(rows: list[dict]) -> dict:
    ranks = [row["rank"] for row in rows]
    return {
        "questions": len(rows),
        "hits_at_5": sum(rank is not None for rank in ranks),
        "hit_at_5": mean(rank is not None for rank in ranks),
        "mrr_at_5": mean(1 / rank if rank else 0 for rank in ranks),
        "median_retrieval_seconds": median(row["seconds"] for row in rows),
    }


def retrieval_run(retriever, questions: list[dict], output: Path) -> dict:
    results = {}
    answerable = [item for item in questions if item["source_urls"]]
    if not answerable:
        raise ValueError("The retrieval ablation requires at least one answerable question.")
    for label, (mode, rerank) in CONFIGURATIONS.items():
        rows = []
        with (output / f"{label}.jsonl").open("w", encoding="utf-8", buffering=1) as file:
            for item in answerable:
                start = perf_counter()
                hits = retriever.search(item["question"], k=5, mode=mode, rerank=rerank)
                row = retrieve_result(item, hits, perf_counter() - start)
                rows.append(row)
                file.write(json.dumps(row, ensure_ascii=False) + "\n")
        results[label] = retrieval_summary(rows)
        print(f"{label}: {results[label]}", flush=True)
    return results


def answer_run(retriever, questions: list[dict], output: Path) -> dict:
    from src.bot import NO_INFO, DepartmentChatbot

    class ObservedRetriever:
        last_search = None

        def search(self, question, k):
            start = perf_counter()
            hits = retriever.search(question, k=k)
            self.last_search = (question, k, hits, perf_counter() - start)
            return hits

        def doc_chunks(self, doc_id):
            return retriever.doc_chunks(doc_id)

    class EvaluationBot(DepartmentChatbot):
        # Keep the passages used by the actual streaming response for citation review.
        evidence = ()

        def build_prompt(self, question, hits):
            self.evidence = hits
            return super().build_prompt(question, hits)

    observed = ObservedRetriever()
    bot = EvaluationBot(retriever=observed)
    results = []
    with (output / "answers.jsonl").open("w", encoding="utf-8") as file:
        for item in questions:
            bot.evidence = ()
            observed.last_search = None
            text = ""
            first_word = None
            start = perf_counter()
            for piece in bot.respond_stream(item["question"]):
                text += piece
                if first_word is None and re.search(r"\w+\W", text):
                    first_word = perf_counter() - start
            elapsed = perf_counter() - start
            answer = text.partition("\n\n**Sources**\n")[0]
            result = {
                **item,
                "answer": answer,
                "abstained": NO_INFO.split(". ", 1)[0].lower() in answer.lower(),
                "first_word_seconds": round(first_word if first_word is not None else elapsed, 3),
                "full_answer_seconds": round(elapsed, 3),
                "passages": [{"number": i, "url": hit.meta["url"], "section": hit.meta["section_path"], "text": hit.text}
                             for i, hit in enumerate(bot.evidence, 1)],
            }
            if item["source_urls"] and observed.last_search is not None:
                query, k, hits, seconds = observed.last_search
                if query == item["question"] and k == 5:
                    result["hybrid_rerank_at_5"] = retrieve_result(item, hits, seconds)
            results.append(result)
            file.write(json.dumps(result, ensure_ascii=False) + "\n")
            file.flush()
            print(f"{item['id']}: {elapsed:.2f}s", flush=True)
    unknown = [result for result in results if not result["source_urls"]]
    answerable = [result for result in results if result["source_urls"]]
    summary = {
        "provider": bot.llm.provider,
        "model": bot.llm.model,
        "questions": len(results),
        "unanswerable_refusals": sum(result["abstained"] for result in unknown),
        "unanswerable_questions": len(unknown),
        "answerable_refusals": sum(result["abstained"] for result in answerable),
        "answerable_questions": len(answerable),
        "median_first_word_seconds": median(result["first_word_seconds"] for result in results),
        "median_full_answer_seconds": median(result["full_answer_seconds"] for result in results),
    }
    if answerable and all("hybrid_rerank_at_5" in row for row in answerable):
        summary["hybrid_rerank_at_5"] = retrieval_summary([row["hybrid_rerank_at_5"] for row in answerable])
    return summary


def grade_run(answers_path: Path, grades_path: Path) -> dict:
    answers = {row["id"]: row for row in read_jsonl(answers_path) if row["source_urls"]}
    grades = {}
    for row in read_jsonl(grades_path):
        grade = Grade(**row)
        if grade.id in grades or grade.id not in answers:
            raise ValueError(f"Duplicate or unknown answerable question ID: {grade.id}")
        for field, value in asdict(grade).items():
            if field != "id" and not 0 <= value <= 1:
                raise ValueError(f"{grade.id}: {field} must be between 0 and 1")
        grades[grade.id] = grade
    if not answers or grades.keys() != answers.keys():
        raise ValueError("Provide one grade per answerable question in the answer run.")
    return {
        "graded_questions": len(grades),
        **{field: mean(getattr(grade, field) for grade in grades.values())
           for field in ("correctness", "completeness", "faithfulness")},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["retrieval", "answers", "grade"], help="evaluation to run")
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--answers", type=Path, help="answer run JSONL for grade mode")
    parser.add_argument("--grades", type=Path, help="reviewer grades JSONL for grade mode")
    args = parser.parse_args()
    if args.mode == "grade":
        if args.answers is None or args.grades is None:
            parser.error("grade mode requires --answers and --grades")
        print(json.dumps(grade_run(args.answers, args.grades), indent=2))
        return

    from src.config import CHUNKS_PATH, COLLECTION_NAME, RERANKER_MODEL, VECTOR_STORE_PATH
    from src.retriever import Retriever

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = args.output_dir or ROOT / "data" / "evaluation" / timestamp
    questions = read_jsonl(args.questions)
    if not questions:
        parser.error("question file is empty")
    output.mkdir(parents=True, exist_ok=False)
    print(f"Writing results to {output.resolve()}", flush=True)
    retriever = Retriever(device="cpu")
    evaluation = retrieval_run if args.mode == "retrieval" else answer_run
    metrics = evaluation(retriever, questions, output)
    write_json(output / "summary.json", {
        "mode": args.mode,
        "questions": str(args.questions),
        "chroma": str(VECTOR_STORE_PATH),
        "chunks": str(CHUNKS_PATH),
        "collection": COLLECTION_NAME,
        "embedding_model": retriever.col.metadata["embedding_model"],
        "chunks_sha256": retriever.col.metadata.get("chunks_sha256"),
        "reranker": RERANKER_MODEL,
        "retrieval_device": "cpu",
        "platform": platform.platform(),
        "metrics": metrics,
    })


if __name__ == "__main__":
    main()
