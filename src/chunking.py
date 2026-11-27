"""Split normalized documents into section-aware chunks for indexing."""

import re
from collections import Counter
from collections.abc import Iterator

TARGET_TOKENS = 350
OVERLAP_TOKENS = 50
MIN_CHUNK_TOKENS = 20
MIN_ALNUM_RATIO = 0.5
MAX_EMBEDDING_TOKENS = 512

HEADING_RE = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")
SEP_RE = re.compile(r"^\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)*\|?\s*$")
SENT_RE = re.compile(r"(?<=[.!?;])\s+(?=[A-Z0-9(\[\"'])|(?<=\S)\n(?=\s*(?:[-*+]\s|\d+[.)]\s))")
GENERIC_CTX_RE = re.compile(r"^\W*(click\s*here|here|link|download|more|read\s*more|details?|view|pdf)\W*$", re.I)
RESEARCH_PAGE_RE = re.compile(r"pub|paper|gallery|lecture|thesis|slides", re.I)
FAQ_RE = re.compile(r"\bfaq(?:s|'s)?\b|frequently asked questions", re.I)


def clean(text: str) -> str:
    text = re.sub(r"<!--.*?-->", "", text, flags=re.S)
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text)
    text = re.sub(r"\*\*==>.*?<==\*\*", "", text)
    text = re.sub(r"<br\s*/?>", " ", text, flags=re.I)
    text = text.replace("�", "").replace("\u00a0", " ")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def tier_of(doc: dict) -> str:
    if "tier" in doc:
        tier = doc["tier"]
        if tier not in ("core", "research"):
            raise ValueError(f"Invalid tier {tier!r} for {doc['url']}")
        return tier
    url = doc["url"]
    if "/archiveTalks/" in url:
        return "research"
    if "/~" in url and (doc["source_kind"] != "html" or RESEARCH_PAGE_RE.search(url)):
        return "research"
    return "core"


def blocks(text: str) -> Iterator[tuple[str, int, str]]:
    """Yield headings, prose paragraphs, and Markdown tables in source order."""
    lines = text.splitlines()
    paragraph = []

    def flush():
        if paragraph:
            is_list = any(re.match(r"(?:[-*+]\s|\d+[.)]\s)", line) for line in paragraph)
            yield "para", 0, ("\n" if is_list else " ").join(paragraph)
            paragraph.clear()

    index = 0
    while index < len(lines):
        line = lines[index].strip()
        index += 1
        heading = HEADING_RE.match(line)
        if heading:
            yield from flush()
            title = re.sub(r"\s+#+$", "", heading[2])
            title = re.sub(r"<[^>]+>|\*\*|__", "", title).strip("* ")
            if title:
                yield "heading", len(heading[1]), title
        elif line.startswith("|") or (
            "|" in line and index < len(lines) and SEP_RE.match(lines[index].strip())
        ):
            yield from flush()
            rows = [line]
            while index < len(lines):
                row = lines[index].strip()
                if "|" not in row or HEADING_RE.match(row):
                    break
                rows.append(row)
                index += 1
            yield "table", 0, "\n".join(rows)
        elif line:
            paragraph.append(line)
        else:
            yield from flush()
    yield from flush()


def sections(text: str) -> Iterator[tuple[str, str, list[tuple[str, str]]]]:
    stack, content = [], []
    heading = ""
    for kind, level, value in blocks(text):
        if kind == "heading":
            if content:
                yield " > ".join(title for _, title in stack), heading, content
                content = []
            stack = [(depth, title) for depth, title in stack if depth < level]
            stack.append((level, value))
            heading = f"{'#' * level} {value}"
        else:
            content.append((kind, value))
    if content:
        yield " > ".join(title for _, title in stack), heading, content


class Chunker:
    def __init__(self, tokenizer, max_length: int = MAX_EMBEDDING_TOKENS):
        self.tok = tokenizer
        self.max_length = min(max_length, MAX_EMBEDDING_TOKENS)

    def n(self, text: str, *, special_tokens: bool = False) -> int:
        return len(self.tok.encode(text, add_special_tokens=special_tokens, verbose=False))

    def fit_text(self, text: str, limit: int, *, tail: bool = False) -> str:
        """Find a source-text prefix or suffix within the budget without decoding tokens."""
        def take(size: int) -> str:
            return text[-size:] if tail and size else text[:size]

        low, high = 0, min(len(text), max(limit * 4, 1))
        while self.n(take(high)) <= limit:
            if high == len(text):
                return text.strip()
            low, high = high, min(len(text), high * 2)
        while low < high:
            mid = (low + high + 1) // 2
            if self.n(take(mid)) <= limit:
                low = mid
            else:
                high = mid - 1
        piece = take(low)
        if 0 < low < len(text):
            if tail and not text[-low - 1].isspace() and not piece[0].isspace():
                boundary = re.search(r"\s", piece)
                if boundary:
                    piece = piece[boundary.end():]
            elif not tail and not text[low].isspace() and not piece[-1].isspace():
                boundary = re.search(r"\s\S+$", piece)
                if boundary:
                    piece = piece[:boundary.start()]
        return piece.strip()

    def embedding_header(self, title: str, section: str, context: str = "") -> str:
        parts = section.split(" > ") if section else []
        while parts and parts[0].casefold() == title.casefold():
            parts.pop(0)
        names = [part for part in (title, *parts) if part]
        header = " > ".join(names)
        limit = self.max_length // 2
        if self.n(header, special_tokens=True) > limit:
            # Deep or very long headings still retain a title and the most specific section.
            special = self.n("", special_tokens=True)
            title_limit = min(self.n(title), (limit - special) // 2) if parts else limit - special
            short_title = self.fit_text(title, title_limit)
            section_limit = limit - self.n(f"{short_title} > ", special_tokens=True)
            short_section = self.fit_text(parts[-1], section_limit) if parts else ""
            header = " > ".join(part for part in (short_title, short_section) if part)
        if context:
            with_context = f"{header}\n(Linked on the website as: {context[:200]})"
            if self.n(with_context, special_tokens=True) <= limit:
                header = with_context
        return header

    def windows(self, text: str, limit: int) -> Iterator[str]:
        while text:
            piece = self.fit_text(text, limit)
            if not piece:
                raise ValueError("The token budget cannot fit the first character of a chunk")
            yield piece
            text = text[len(piece):].lstrip()

    def split_prose(self, text: str, limit: int) -> Iterator[str]:
        start = end = 0
        boundaries = [match.start() for match in SENT_RE.finditer(text)] + [len(text)]
        for boundary in boundaries:
            candidate = text[start:boundary].strip()
            if self.n(candidate) > limit:
                if end > start:
                    yield text[start:end].strip()
                    start = end
                sentence = text[start:boundary].strip()
                if self.n(sentence) > limit:
                    yield from self.windows(sentence, limit)
                    start = boundary
            end = boundary
        if end > start:
            yield text[start:end].strip()

    def split_table(self, text: str, limit: int) -> Iterator[str]:
        rows = text.splitlines()
        header = rows[:2] if len(rows) > 1 and SEP_RE.match(rows[1]) else []
        if header and self.n("\n".join(header)) >= limit:
            raise ValueError("A table header exceeds the chunk budget. Split the table into narrower tables.")

        current = header.copy()
        for row in rows[len(header):]:
            if self.n("\n".join([*current, row])) <= limit:
                current.append(row)
                continue
            if current != header:
                yield "\n".join(current)
            if self.n("\n".join([*header, row])) <= limit:
                current = [*header, row]
                continue

            # Split a single unusually long row into cell fragments, retaining column position.
            current = header.copy()
            cells = re.split(r"(?<!\\)\|", row.strip("|"))
            skeleton = "| " + " | ".join("" for _ in cells) + " |"
            row_limit = limit - self.n("\n".join([*header, skeleton]))
            if row_limit < 1:
                raise ValueError("A table is too wide to keep its header and cells in one chunk.")
            for column, cell in enumerate(cells):
                if not cell.strip():
                    continue
                for piece in self.windows(cell.strip(), row_limit):
                    fragment = [""] * len(cells)
                    fragment[column] = piece
                    yield "\n".join([*header, "| " + " | ".join(fragment) + " |"])
        if current != header or rows == header:
            yield "\n".join(current)

    def prose_tail(self, text: str, limit: int) -> str:
        if limit <= 0:
            return ""
        end = len(text)
        for start in reversed([0, *(match.end() for match in SENT_RE.finditer(text))]):
            if self.n(text[start:]) > limit:
                break
            end = start
        return text[end:] if end < len(text) else self.fit_text(text, limit, tail=True)

    def section_chunks(self, content: list[tuple[str, str]], limit: int) -> Iterator[str]:
        overlap = min(OVERLAP_TOKENS, limit // 4)
        current = []
        last_prose = ""
        for kind, text in content:
            if self.n(text) <= limit:
                pieces = [text]
            elif kind == "table":
                pieces = self.split_table(text, limit)
            else:
                pieces = self.split_prose(text, limit - overlap)
            for piece in pieces:
                if current and self.n("\n\n".join([*current, piece])) > limit:
                    yield "\n\n".join(current)
                    carry = self.prose_tail(last_prose, min(overlap, limit - self.n(piece))) if kind == "para" else ""
                    current = [carry] if carry and self.n(f"{carry}\n\n{piece}") <= limit else []
                current.append(piece)
                last_prose = piece if kind == "para" else ""
        if current:
            yield "\n\n".join(current)

    def chunk(self, text: str, title: str = "", context: str = "") -> Iterator[tuple[str, str]]:
        """Yield the heading path and body for each chunk."""
        for path, heading, content in sections(text):
            header = self.embedding_header(title, path, context)
            limit = min(TARGET_TOKENS, self.max_length - self.n(f"{header}\n\n", special_tokens=True))
            if limit < 1:
                raise ValueError(f"No token budget left for body text in {title!r} > {path!r}")
            try:
                for index, body in enumerate(self.section_chunks(content, limit)):
                    if index == 0 and heading and self.n(f"{heading}\n\n{body}") <= limit:
                        body = f"{heading}\n\n{body}"
                    yield path, body
            except ValueError as error:
                raise ValueError(f"{title!r} > {path!r}: {error}") from error


def alnum_ratio(text: str) -> float:
    # Blank fields and rules in PDF forms are layout, not extracted text noise.
    text = re.sub(r"_{4,}|-{4,}", "", text)
    chars = [char for char in text if not char.isspace()]
    return sum(char.isalnum() for char in chars) / max(len(chars), 1)


def build_chunks(docs: list[dict], chunker: Chunker, tiers: set[str]) -> tuple[list[dict], Counter]:
    chunks, stats = [], Counter()
    for doc in docs:
        tier = tier_of(doc)
        if tier not in tiers:
            stats[f"skipped_tier_{tier}"] += 1
            continue
        title = doc["title"].strip()
        context = ""
        if doc["source_kind"] != "html":
            context = next(
                (item for item in doc.get("link_context") or []
                 if item and item.casefold() != title.casefold() and not GENERIC_CTX_RE.match(item)),
                "",
            )
        is_faq = bool(FAQ_RE.search(f"{doc['doc_type']} {title} {doc['url']}"))
        curated = "tier" in doc or doc["source_kind"] in {"markdown", "manual"}
        seen, kept = set(), []
        for path, body in chunker.chunk(clean(doc["text"]), title, context):
            norm = re.sub(r"\W+", " ", body).casefold().strip()
            key = (path, norm)
            if key in seen:
                stats["dropped_duplicate"] += 1
                continue
            tokens = chunker.n(body)
            if tokens < MIN_CHUNK_TOKENS:
                allow_short = is_faq or path.endswith("?") or curated
                heading, separator, answer = body.partition("\n\n")
                content = answer if separator and HEADING_RE.match(heading) else body
                useful = allow_short and any(char.isalnum() for char in content)
            else:
                useful = alnum_ratio(body) >= MIN_ALNUM_RATIO
            if not useful:
                stats["dropped_junk"] += 1
                continue
            seen.add(key)
            header = chunker.embedding_header(title, path, context)
            embed_text = f"{header}\n\n{body}"
            embedding_tokens = chunker.n(embed_text, special_tokens=True)
            if embedding_tokens > chunker.max_length:
                raise ValueError(
                    f"{doc['url']} > {path!r} needs {embedding_tokens} embedding tokens, "
                    f"above the limit of {chunker.max_length}"
                )
            kept.append((path, body, tokens, embed_text, embedding_tokens))
        if "tier" in doc and not kept:
            raise ValueError(f"No searchable text in registered source {doc['url']}")
        for index, (path, body, tokens, embed_text, embedding_tokens) in enumerate(kept):
            chunks.append({
                "id": f"{doc['id']}-{index:04d}",
                "doc_id": doc["id"],
                "url": doc["url"],
                "title": title,
                "section_path": path,
                "doc_type": doc["doc_type"],
                "tier": tier,
                "source_kind": doc["source_kind"],
                "chunk_index": index,
                "n_chunks": len(kept),
                "latest_year": doc.get("latest_year") or -1,
                "last_modified": doc.get("last_modified") or "",
                "fetched_at": doc.get("fetched_at") or "",
                "n_tokens": tokens,
                "embedding_tokens": embedding_tokens,
                "text": body,
                "embed_text": embed_text,
            })
    return chunks, stats
