"""Read the local source register and merge it into an optional corpus."""

import hashlib
import json
import re
from pathlib import Path
from string import ascii_letters, digits
from urllib.parse import parse_qsl, quote, urlencode, urlsplit, urlunsplit

SITE_HOST = "www.sc.iitb.ac.in"
SITE_ALIASES = {SITE_HOST, "sc.iitb.ac.in"}
UNRESERVED = set(ascii_letters + digits + "-._~")
ESCAPE_RE = re.compile(r"%[0-9a-fA-F]{2}")
TRACKING_RE = re.compile(r"^(utm_\w+|fbclid|gclid|_ga)$", re.I)
CORE_TYPES = {"admission", "course", "rules"}


def _normalize_escape(match: re.Match) -> str:
    escape = match[0]
    char = chr(int(escape[1:], 16))
    return char if char in UNRESERVED else escape.upper()


def source_identity(url: str, *, legacy: bool = False) -> str:
    """Canonicalize the identity while preserving the supplied citation URL."""
    if legacy and url.startswith("manual://"):
        return url
    invalid = f"expected an HTTP(S) citation URL, got {url!r}"
    try:
        parsed = urlsplit(url)
        host, port = parsed.hostname, parsed.port
    except ValueError as exc:
        raise ValueError(invalid) from exc
    if (
        parsed.scheme not in {"http", "https"}
        or not host
        or any(char.isspace() for char in host)
        or any(ord(char) < 32 for char in url)
        or parsed.username is not None
    ):
        raise ValueError(invalid)

    scheme = parsed.scheme
    same_site = host in SITE_ALIASES
    if port == {"http": 80, "https": 443}[scheme]:
        port = None
    if same_site:
        host, scheme = SITE_HOST, "https"
    netloc = f"[{host}]" if ":" in host else host
    if port is not None:
        netloc = f"{netloc}:{port}"

    path = re.sub(r"/{2,}", "/", parsed.path).rstrip("/") or "/"
    path = quote(ESCAPE_RE.sub(_normalize_escape, path), safe="/%:@!$&'()*+,;=")
    query = urlencode(sorted(
        (key, value)
        for key, value in parse_qsl(parsed.query, keep_blank_values=True)
        if not TRACKING_RE.fullmatch(key)
    ))
    # The site parser emits its boilerplate as a separate synthetic document.
    fragment = "site-wide" if same_site and path == "/" and parsed.fragment == "site-wide" else ""
    return urlunsplit((scheme, netloc, path, query, fragment))


def _required_text(record: dict, field: str) -> str:
    value = record.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field!r} must be a nonempty string")
    return value.strip()


def _load_baseline(path: Path) -> dict[str, dict]:
    documents = {}
    ids = set()
    with path.open(encoding="utf-8-sig") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                doc = json.loads(line)
                if not isinstance(doc, dict):
                    raise ValueError("expected a document object")
                for field in ("id", "url", "title", "doc_type", "source_kind", "text"):
                    _required_text(doc, field)
                identity = source_identity(doc["url"].strip(), legacy=True)
                if doc["id"] in ids:
                    raise ValueError(f"duplicate document ID {doc['id']!r}")
                previous = documents.get(identity)
                if previous is not None:
                    # Legacy parser outputs could include a transcription alongside its landing page.
                    if doc["source_kind"] == "manual" and previous["source_kind"] != "manual":
                        selected = {**doc, "id": previous["id"]}
                    elif previous["source_kind"] == "manual" and doc["source_kind"] != "manual":
                        selected = {**previous, "id": doc["id"]}
                    else:
                        raise ValueError(f"duplicate source URL {doc['url']!r}")
                else:
                    selected = doc
            except ValueError as exc:
                raise ValueError(f"{path}:{number}: {exc}") from exc
            documents[identity] = selected
            ids.add(doc["id"])
    return documents


def _read_source(path: Path, url: str) -> tuple[str, str, dict]:
    suffix = path.suffix.lower()
    if suffix in {".md", ".markdown"}:
        text = path.read_text(encoding="utf-8-sig").strip()
        if not text:
            raise ValueError(f"{path} is empty")
        return text, "markdown", {}
    if suffix != ".pdf":
        raise ValueError(f"unsupported file type for {path}; use Markdown or PDF")

    from .pdf_extract import parse_pdf

    try:
        text, metadata = parse_pdf(path)
    except ImportError as exc:
        raise ValueError("PDF extraction needs the packages in requirements-ingest.txt") from exc
    except (OSError, RuntimeError, ValueError) as exc:
        raise ValueError(f"could not extract text from {path}: {exc}") from exc
    if metadata["needs_ocr"] or not text.strip():
        pages = metadata.get("image_only_pages", [])
        detail = f" (image-only pages: {', '.join(map(str, pages))})" if pages else ""
        raise ValueError(
            f"PDF extraction needs review for {path}{detail}. "
            f"Register a Markdown transcription with the same official URL {url!r}."
        )
    return text, "pdf", {"pages": metadata["pages"]}


def _curated_document(entry: dict, root: Path, url: str) -> dict:
    filename = Path(_required_text(entry, "file"))
    path = (root / filename).resolve()
    if filename.is_absolute() or not path.is_relative_to(root):
        raise ValueError("'file' must be relative to the source register and stay inside its directory")
    title = _required_text(entry, "title")
    doc_type = _required_text(entry, "doc_type")
    tier = _required_text(entry, "tier")
    if tier not in {"core", "research"}:
        raise ValueError("'tier' must be 'core' or 'research'")
    if doc_type in CORE_TYPES and tier != "core":
        raise ValueError(f"{doc_type!r} documents require tier 'core'")
    text, source_kind, metadata = _read_source(path, url)
    return {
        "url": url,
        "title": title,
        "doc_type": doc_type,
        "tier": tier,
        "source_kind": source_kind,
        "text": text,
        "n_chars": len(text),
        **metadata,
    }


def load_documents(baseline: Path | None, sources: Path | None) -> list[dict]:
    """Load baseline records and replace matching URLs with registered local files."""
    documents = _load_baseline(baseline) if baseline is not None else {}
    if sources is None:
        return list(documents.values())

    try:
        entries = json.loads(sources.read_text(encoding="utf-8-sig"))
    except ValueError as exc:
        raise ValueError(f"{sources}: {exc}") from exc
    if not isinstance(entries, list):
        raise ValueError(f"{sources}: expected an array of source objects")

    root = sources.resolve().parent
    registered = set()
    ids = {doc["id"] for doc in documents.values()}
    for number, entry in enumerate(entries, start=1):
        try:
            if not isinstance(entry, dict):
                raise ValueError("expected a source object")
            url = _required_text(entry, "url")
            identity = source_identity(url)
            if identity in registered:
                raise ValueError(f"duplicate registered source URL {url!r}")
            doc = _curated_document(entry, root, url)
            if identity in documents:
                doc["id"] = documents[identity]["id"]
            else:
                doc["id"] = hashlib.sha1(identity.encode()).hexdigest()[:12]
                if doc["id"] in ids:
                    raise ValueError(f"document ID collision for {url!r}")
        except (OSError, ValueError) as exc:
            raise ValueError(f"{sources}, entry {number}: {exc}") from exc
        documents[identity] = doc
        registered.add(identity)
        ids.add(doc["id"])
    return list(documents.values())
