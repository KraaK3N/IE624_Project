"""SysBot - SysCon chatbot: routing -> hybrid retrieval -> grounded, cited, streamed answer.

    python -m src.bot "How do I apply for leave as a PhD student?"
"""

import re
import sys
import time
from collections.abc import Iterator
from dataclasses import dataclass

from src.config import (ANTHROPIC_API_KEY, LLM_PROVIDER, MIN_RERANK_SCORE, MODEL_NAME, OLLAMA_HOST,
                        OPENAI_API_KEY, TOP_K)
from src.retriever import Hit, Retriever

SITE = "https://www.sc.iitb.ac.in/"
NO_INFO = (
    "I don't have that information in the SysCon website content. "
    f"Please check {SITE} or contact the SysCon office."
)

SYSTEM_PROMPT = f"""You are SysBot, the assistant for the Centre for Systems and Control (SysCon), IIT Bombay.
Answer questions from students, applicants and visitors using ONLY the numbered context passages provided.

Rules:
1. Use only facts stated in the context. Never use outside knowledge or guess.
2. Cite every fact with its passage number in square brackets, e.g. [1] or [2][3].
3. If the context does not answer the question, reply exactly: "{NO_INFO}"
   Give a partial answer only if part of the question is answered, and say what is missing.
4. Institute-wide documents (e.g. IIT Bombay brochures) are not SysCon-specific; say so when you use them.
5. Mention dates or academic years when the context gives them, since information may be outdated.
6. Be concise: a short paragraph or a few bullet points. Do not mention "passages" or "context" in the answer."""

ROUTER_PROMPT = """You are SysBot, the friendly assistant of SysCon (Centre for Systems and Control, IIT Bombay).
Decide how to handle the user's latest message, using the conversation for context.

- SEARCH whenever the message asks for, or follows up on, ANY information about SysCon or IIT Bombay
  (courses, curriculum, course work, credits, faculty, research, labs, admissions, eligibility, rules,
  forms, fees, scholarships, events, contacts, ...). Rewrite it as one standalone question that includes
  context from the conversation. Output:  SEARCH: <question>
  When in doubt, choose SEARCH. Never ask the user to clarify a department question; search instead.
- Only for greetings, small talk ("how are you"), thanks, feelings or questions about you: reply naturally
  and warmly in 1-2 sentences and invite a SysCon question. Output:  CHAT: <your reply>
- For clearly off-topic requests (general knowledge, coding, weather, ...): politely say you only help
  with SysCon matters. Output:  CHAT: <your reply>

Output exactly one line starting with SEARCH: or CHAT:"""

GREETING_RE = re.compile(r"^(hi+|hello+|hey+|namaste|good\s(morning|afternoon|evening))[\s!.,]*(there)?[\s!.]*$", re.I)
THANKS_RE = re.compile(r"^(thanks?|thank\s*you|thx|ty)( (so much|a lot))?[\s!.]*$", re.I)
DOMAIN_RE = re.compile(
    r"\b(syscon|iit|course|courses|coursework|curriculum|syllabus|credit|credits|m\.?\s?tech|ph\.?\s?d|mtech|phd|"
    r"faculty|professor|prof|staff|admission|admissions|eligib\w*|apply|application|exam|semester|thesis|"
    r"seminar|colloquium|lab|labs|research|project|form|forms|fee|fees|scholarship|stipend|leave|rule|rules|"
    r"deadline|timetable|contact|office|convener|coordinator|ta|ra|gate|interview|apsc?|rpc|synopsis|"
    r"sc\s?-?\d{3})\b",
    re.I,
)


def small_talk(query: str) -> str | None:
    q = query.strip()
    if GREETING_RE.match(q):
        return "Hello! I'm SysBot. What would you like to know about SysCon?"
    if THANKS_RE.match(q):
        return "You're welcome! Anything else you'd like to know about SysCon?"
    return None


@dataclass
class Answer:
    text: str
    sources: list[tuple[int, Hit]]  # (citation number used in the text, hit)
    query_used: str
    abstained: bool


class LLM:
    def __init__(self, provider: str = LLM_PROVIDER, model: str = MODEL_NAME):
        self.provider, self.model = provider, model
        if provider == "ollama":
            import ollama
            self.client = ollama.Client(host=OLLAMA_HOST)
        elif provider == "anthropic":
            import anthropic
            self.client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        elif provider == "openai":
            import openai
            self.client = openai.OpenAI(api_key=OPENAI_API_KEY)
        else:
            raise ValueError(f"Unknown LLM_PROVIDER: {provider}")

    def stream(self, system: str, user: str, max_tokens: int = 600) -> Iterator[str]:
        if self.provider == "ollama":
            for part in self.client.chat(
                model=self.model, stream=True, keep_alive="30m",
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                options={"temperature": 0.1, "num_ctx": 8192, "num_predict": max_tokens},
            ):
                yield part["message"]["content"]
        elif self.provider == "anthropic":
            with self.client.messages.stream(
                model=self.model, system=system, max_tokens=max_tokens, temperature=0.1,
                messages=[{"role": "user", "content": user}],
            ) as s:
                yield from s.text_stream
        else:
            for part in self.client.chat.completions.create(
                model=self.model, stream=True,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            ):
                if part.choices and part.choices[0].delta.content:
                    yield part.choices[0].delta.content

    def chat(self, system: str, user: str, max_tokens: int = 600) -> str:
        return "".join(self.stream(system, user, max_tokens)).strip()


class DepartmentChatbot:
    def __init__(self, retriever: Retriever | None = None, llm: LLM | None = None):
        self.retriever = retriever or Retriever()
        self.llm = llm or LLM()

    # ---------- steps ----------
    def route(self, query: str, history: list[dict]) -> tuple[str, str]:
        """("search", question) or ("chat", reply). Clear department questions skip the LLM call."""
        turns = [m for m in history if m["role"] in {"user", "assistant"}][-4:]
        has_context = any(m["role"] == "user" for m in turns)
        if DOMAIN_RE.search(query) and (not has_context or len(query.split()) >= 6):
            return "search", query
        convo = "\n".join(f"{m['role'].title()}: {m['content'][:400]}" for m in turns) or "(none)"
        out = self.llm.chat(ROUTER_PROMPT, f"Conversation so far:\n{convo}\n\nLatest message: {query}", 150)
        m = re.search(r"\b(SEARCH|CHAT)\s*:\s*(.+)", out, re.S | re.I)
        if not m:
            return "search", query
        return m.group(1).lower(), m.group(2).strip().strip('"') or query

    @staticmethod
    def build_prompt(question: str, hits: list[Hit]) -> str:
        context = "\n\n".join(
            f"[{i}] {h.meta['title']}" + (f" - {h.meta['section_path']}" if h.meta.get("section_path") else "")
            + f"\nURL: {h.meta['url']}\n{h.text}"
            for i, h in enumerate(hits, 1)
        )
        return f"Context:\n\n{context}\n\nQuestion: {question}"

    @staticmethod
    def cited(text: str, hits: list[Hit]) -> list[tuple[int, Hit]]:
        if NO_INFO[:40] in text:
            return []
        nums = sorted({int(n) for n in re.findall(r"\[(\d+)\]", text) if 0 < int(n) <= len(hits)})
        return [(n, hits[n - 1]) for n in nums]

    @staticmethod
    def format_sources(sources: list[tuple[int, Hit]]) -> str:
        by_url: dict[str, tuple[list[int], str]] = {}
        for n, h in sources:
            by_url.setdefault(h.meta["url"], ([], h.meta["title"]))[0].append(n)
        refs = "\n".join(f"- {''.join(f'[{n}]' for n in ns)} [{title}]({url.replace(' ', '%20')})"
                          for url, (ns, title) in by_url.items())
        return f"\n\n**Sources**\n{refs}" if refs else ""

    # ---------- public API ----------
    def respond_stream(self, query: str, history: list[dict] | None = None) -> Iterator[str]:
        """Yield the reply in pieces (for the UI)."""
        if not query.strip():
            yield "How can I help you with SysCon today?"
            return
        if reply := small_talk(query):
            yield reply
            return
        t0 = time.perf_counter()
        kind, q = self.route(query, history or [])
        t1 = time.perf_counter()
        if kind == "chat":
            print(f"[timing] route {t1 - t0:.1f}s (chat)", flush=True)
            yield q
            return
        hits = self.retriever.search(q, k=TOP_K)
        t2 = time.perf_counter()
        if not hits or (hits[0].rerank_score is not None and hits[0].rerank_score < MIN_RERANK_SCORE):
            print(f"[timing] route {t1 - t0:.1f}s | retrieve {t2 - t1:.1f}s | abstained", flush=True)
            yield NO_INFO
            return
        text, first = "", None
        for piece in self.llm.stream(SYSTEM_PROMPT, self.build_prompt(q, hits)):
            first = first or time.perf_counter()
            text += piece
            yield piece
        t3 = time.perf_counter()
        print(f"[timing] route {t1 - t0:.1f}s | retrieve {t2 - t1:.1f}s | first word {(first or t3) - t2:.1f}s "
              f"| generate {t3 - t2:.1f}s | query: {q}", flush=True)
        yield self.format_sources(self.cited(text, hits))

    def respond(self, query: str, history: list[dict] | None = None) -> str:
        return "".join(self.respond_stream(query, history))

    def answer(self, query: str, history: list[dict] | None = None) -> Answer:
        """Structured result (for evaluation scripts)."""
        kind, q = self.route(query, history or [])
        if kind == "chat":
            return Answer(q, [], query, False)
        hits = self.retriever.search(q, k=TOP_K)
        if not hits or (hits[0].rerank_score is not None and hits[0].rerank_score < MIN_RERANK_SCORE):
            return Answer(NO_INFO, [], q, True)
        text = self.llm.chat(SYSTEM_PROMPT, self.build_prompt(q, hits))
        return Answer(text, self.cited(text, hits), q, NO_INFO[:40] in text)


if __name__ == "__main__":
    bot = DepartmentChatbot()
    questions = [" ".join(sys.argv[1:])] if len(sys.argv) > 1 else [
        "How are you?",
        "Who works on stochastic control?",
        "What is the eligibility for M.Tech admission?",
        "How do I apply for leave as a PhD student?",
    ]
    for q in questions:
        print(f"\nQ: {q}")
        for piece in bot.respond_stream(q):
            print(piece, end="", flush=True)
        print("\n" + "-" * 80)
