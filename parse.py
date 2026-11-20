"""Stage 2 - turn data/raw (from scripts/crawl.py) into a clean Markdown corpus.

    python scripts/parse.py

Outputs (data/processed/):
    markdown/*.md      one clean document per page / file, with metadata header (for reading)
Also includes hand-written files from data/manual/*.md (e.g. timetables that exist only as images).
Put "# Title" and "Source: <page url>" at the top of each manual file.
    corpus.jsonl       one record per document: text + metadata (input to chunking)
    parse_report.json  thin pages, scanned PDFs needing OCR, unparsed files
"""

import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import urljoin, urlparse

import pymupdf
import pymupdf4llm
from bs4 import BeautifulSoup, Comment
from markdownify import markdownify
from markitdown import MarkItDown

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent if HERE.name == "scripts" else HERE
RAW, OUT, MANUAL = ROOT / "data" / "raw", ROOT / "data" / "processed", ROOT / "data" / "manual"
MD_DIR = OUT / "markdown"
SITE_HOSTS = {"www.sc.iitb.ac.in", "sc.iitb.ac.in"}
DOC_EXT = {".pdf", ".doc", ".docx", ".ppt", ".pptx", ".xls", ".xlsx", ".odt", ".ods", ".odp", ".rtf"}

DROP_TAGS = ["script", "style", "noscript", "template", "svg", "canvas", "iframe", "form", "button",
             "input", "select", "textarea", "nav", "aside", "footer", "img", "picture", "video", "audio", "map"]
BOILER_RE = re.compile(
    r"(^|[-_\s])(nav|navbar|navigation|menu|menubar|footer|sidebar|side-bar|breadcrumbs?|cookie|social|share|"
    r"skip|banner|topbar|top-bar|copyright|pagination|pager|search|header-top|top-header|marquee|ticker)([-_\s]|$)",
    re.I,
)
MAIN_SELECTORS = ["main", "[role=main]", "article", "#content", "#main-content", "#maincontent", "#main",
                  ".main-content", ".content", "#page-content", ".page-content", "#primary"]
GENERIC_ANCHOR = re.compile(r"^(click\s*here|here|link|download|more|read\s*more|details?|view|pdf|this)\W*$", re.I)
TITLE_SPLIT = re.compile(r"\s+[|–—:-]\s+")
MONTH = r"(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?"
DATE_RE = re.compile(
    rf"\b\d{{1,2}}(?:st|nd|rd|th)?[\s-]+{MONTH},?[\s-]+\d{{4}}\b|\b{MONTH}\s+\d{{1,2}}(?:st|nd|rd|th)?,?\s+\d{{4}}\b|"
    r"\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}[/.]\d{1,2}[/.]\d{4}\b",
    re.I,
)
SEP_RE = re.compile(r"\|(\s*:?-{3,}:?\s*\|)+")
YEAR_RE = re.compile(r"\b(19[89]\d|20[0-4]\d)\b")
DOC_TYPES = [
    ("course", r"course|curricul|syllab|sc\s?-?\d{3}\b|credit|timetable|time-table"),
    ("admission", r"admission|apply|application|eligib|interview|selection|cutoff|cut-off"),
    ("circular", r"circular|notice|notification|office order"),
    ("rules", r"rule|regulation|polic|ordinance|guideline|handbook|manual|procedure|form"),
    ("faculty", r"faculty|professor|\bprof\b|people"),
    ("staff", r"\bstaff\b|office|administrat"),
    ("students", r"student|alumni|scholar|thesis|\bphd\b|mtech|m\.tech"),
    ("event_news", r"news|event|seminar|colloquium|talk|workshop|announcement|conference|lecture"),
    ("research", r"research|\blab\b|laborator|publication|project|facilit"),
    ("contact", r"contact|address|reach|location"),
    ("about", r"about|history|overview|vision|mission|home"),
]
MIN_CHARS = 80


def normalize_url(url: str) -> str:
    p = urlparse(url)
    host = "www.sc.iitb.ac.in" if p.netloc in SITE_HOSTS else p.netloc
    path = p.path if p.path in {"", "/"} else p.path.rstrip("/")
    return f"https://{host}{path}" + (f"?{p.query}" if p.query else "") if p.netloc in SITE_HOSTS else url


def doc_id(url: str) -> str:
    return hashlib.sha1(url.encode()).hexdigest()[:12]


def classify(url: str, title: str) -> str:
    if urlparse(url).path in {"", "/"}:
        return "about"
    hay = f"{urlparse(url).path} {title}".lower()
    return next((name for name, pat in DOC_TYPES if re.search(pat, hay)), "other")


def dates_in(text: str) -> tuple[list[str], int | None]:
    dates = list(dict.fromkeys(m.group(0).strip() for m in DATE_RE.finditer(text)))[:25]
    years = [int(y) for y in YEAR_RE.findall(text)]
    return dates, (max(years) if years else None)


def tidy(md: str) -> str:
    md = md.replace(" ", " ").replace("​", "")
    lines, src = [], md.splitlines()
    for i, line in enumerate(src):
        line = line.rstrip()
        empty_row = re.fullmatch(r"\|(\s*\|)+\s*", line)
        if empty_row and i + 2 < len(src) and SEP_RE.fullmatch(src[i + 1].strip()):
            src[i + 1], src[i + 2] = src[i + 2], src[i + 1]
            continue
        if empty_row or re.fullmatch(r"[-*_=\s]{3,}", line):
            continue
        lines.append(re.sub(r"[ \t]{2,}", " ", line) if not line.lstrip().startswith("|") else line)
    md = "\n".join(lines)
    md = re.sub(r"\[\s*\]\([^)]*\)", "", md)
    md = re.sub(r"\n{3,}", "\n\n", md)
    return md.strip()


# ---------------- HTML ----------------
def link_density(tag) -> float:
    text = len(tag.get_text(" ", strip=True)) or 1
    return sum(len(a.get_text(" ", strip=True)) for a in tag.find_all("a")) / text


def unwrap_layout_tables(soup: BeautifulSoup) -> None:
    for table in soup.find_all("table"):
        rows = table.find_all("tr")
        max_cols = max((len(r.find_all(["td", "th"], recursive=False)) for r in rows), default=0)
        is_layout = (
            table.find("table") is not None
            or max_cols <= 1
            or table.get("role") == "presentation"
            or any(c.find(["p", "div", "ul", "ol", "h1", "h2", "h3"]) for c in table.find_all(["td", "th"]))
            and len(rows) <= 3
        )
        if is_layout:
            own = [e for e in table.find_all(["thead", "tbody", "tfoot", "tr", "td", "th"]) if e.find_parent("table") is table]
            for t in [table, *own]:
                t.name = "div"
                t.attrs = {}


def clean_html(raw: bytes, encoding: str | None, base_url: str) -> tuple[str, str, list[str], list[str]]:
    soup = BeautifulSoup(raw, "lxml", from_encoding=encoding)
    page_title = soup.title.get_text(" ", strip=True) if soup.title else ""
    images = list(dict.fromkeys(urljoin(base_url, i["src"]) for i in soup.find_all("img", src=True)))
    for c in soup.find_all(string=lambda s: isinstance(s, Comment)):
        c.extract()
    chrome = []
    for t in soup.find_all(["footer", "header"]) + soup.find_all(id=re.compile(r"footer|header|contact", re.I)) \
            + soup.find_all(class_=re.compile(r"footer|header|contact|address", re.I)):
        if t.name != "nav":
            for el in t.find_all(["nav", "ul", "script", "style"]):
                el.decompose()
            chrome += [l.strip(" |") for l in t.get_text("\n", strip=True).splitlines() if len(l.strip(" |")) > 3]
    for t in soup.find_all(DROP_TAGS):
        t.decompose()
    for t in soup.find_all(True):
        if getattr(t, "decomposed", False) or t.parent is None:
            continue
        ident = " ".join([t.get("id") or "", *(t.get("class") or [])]) if t.attrs else ""
        hidden = "display:none" in (t.get("style") or "").replace(" ", "").lower() or t.get("aria-hidden") == "true"
        if hidden or (ident and BOILER_RE.search(ident) and (link_density(t) > 0.4 or len(t.get_text(strip=True)) < 400)):
            t.decompose()
    for h in soup.find_all("header"):
        if not h.find_parent(["main", "article"]) and link_density(h) > 0.3:
            h.decompose()

    body = soup.body or soup
    body_len, main = len(body.get_text(" ", strip=True)), None
    for sel in MAIN_SELECTORS:
        cand = body.select_one(sel)
        if cand and len(cand.get_text(" ", strip=True)) >= min(200, 0.5 * body_len):
            main = cand
            break
    main = main or body

    unwrap_layout_tables(main)
    for a in main.find_all("a"):
        href = (a.get("href") or "").strip()
        text = a.get_text(" ", strip=True)
        if not href or href.startswith(("#", "javascript:")):
            a.unwrap()
            continue
        url = urljoin(base_url, href)
        p = urlparse(url)
        is_doc = Path(p.path).suffix.lower() in DOC_EXT
        if href.startswith("mailto:"):
            a.replace_with(f"{text} ({href[7:]})" if href[7:] not in text else text)
        elif is_doc or p.netloc not in SITE_HOSTS or GENERIC_ANCHOR.match(text or ""):
            a.attrs = {"href": url}
        else:
            a.unwrap()

    h1 = main.find("h1")
    md = markdownify(str(main), heading_style="ATX", bullets="-", table_infer_header=True,
                     escape_underscores=False, escape_asterisks=False, escape_misc=False)
    return tidy(md), (h1.get_text(" ", strip=True) if h1 else page_title), chrome, images


def strip_common_title_parts(titles: dict[str, str]) -> dict[str, str]:
    parts = {u: [p.strip() for p in TITLE_SPLIT.split(t) if p.strip()] for u, t in titles.items()}
    counts = Counter(p for ps in parts.values() for p in set(ps))
    n = len(titles)
    common = {p for p, c in counts.items() if n >= 4 and c / n > 0.5}
    return {u: " - ".join(p for p in ps if p not in common) or titles[u] for u, ps in parts.items()}


def remove_sitewide_lines(docs: dict[str, str]) -> tuple[dict[str, str], list[str]]:
    n = len(docs)
    if n < 5:
        return docs, []
    counts = Counter(l for md in docs.values() for l in {x.strip() for x in md.splitlines()} if l and not l.startswith("|"))
    common = {l for l, c in counts.items() if c >= max(4, 0.3 * n)}
    cleaned = {u: tidy("\n".join(l for l in md.splitlines() if l.strip() not in common)) for u, md in docs.items()}
    order = {l: i for md in docs.values() for i, l in enumerate(x.strip() for x in md.splitlines()) if l in common}
    return cleaned, sorted(common, key=lambda l: order.get(l, 0))


# ---------------- documents ----------------
def parse_pdf(path: Path) -> tuple[str, dict]:
    with pymupdf.open(path) as doc:
        pages, meta_title = doc.page_count, (doc.metadata or {}).get("title", "")
    md = tidy(pymupdf4llm.to_markdown(str(path), show_progress=False))
    return md, {"pages": pages, "meta_title": meta_title, "needs_ocr": len(md) < 100 * max(pages, 1)}


def parse_office(path: Path, converter: MarkItDown) -> tuple[str, dict]:
    return tidy(converter.convert(str(path)).text_content or ""), {}


def best_doc_title(rec: dict, md: str, meta_title: str) -> str:
    for lt in rec.get("link_texts", []):
        t = re.sub(r"[\s.,:;-]*\b(please\s+)?click\s*here\b.*$", "", lt.get("text", ""), flags=re.I).strip(" .,:;-")
        if 4 < len(t) <= 200 and not GENERIC_ANCHOR.match(t):
            return t
    if meta_title and len(meta_title) > 4 and not re.match(r"(microsoft|untitled|document\d*)", meta_title, re.I):
        return meta_title
    heading = re.search(r"^#+\s+(.+)$", md, re.M)
    return heading.group(1).strip()[:200] if heading else Path(urlparse(rec["url"]).path).name


# ---------------- main ----------------
def main() -> None:
    manifest: dict[str, dict] = json.loads((RAW / "manifest.json").read_text(encoding="utf-8"))
    for rec in manifest.values():
        target = rec.get("duplicate_of")
        if target in manifest and rec.get("link_texts"):
            manifest[target].setdefault("link_texts", []).extend(rec["link_texts"])

    MD_DIR.mkdir(parents=True, exist_ok=True)
    for f in MD_DIR.glob("*.md"):
        f.unlink()
    report = defaultdict(list)

    html_md, html_titles, chrome_counts, page_images = {}, {}, Counter(), {}
    for url, rec in manifest.items():
        if rec.get("kind") != "html" or not (ROOT / rec["path"]).exists():
            continue
        try:
            md, title, chrome, images = clean_html((ROOT / rec["path"]).read_bytes(), rec.get("encoding"), rec.get("base_url", url))
            chrome_counts.update(set(chrome))
            page_images[url] = images
        except Exception as e:  # noqa: BLE001
            report["failed"].append({"url": url, "error": repr(e)[:200]})
            continue
        html_md[url], html_titles[url] = md, title or url
    html_md, common_lines = remove_sitewide_lines(html_md)
    min_pages = max(2, 0.3 * len(html_md))
    common_lines = list(dict.fromkeys([l for l, c in chrome_counts.items() if c >= min_pages] + common_lines))
    html_titles = strip_common_title_parts(html_titles)
    img_counts = Counter(i for imgs in page_images.values() for i in imgs)
    for url, md in html_md.items():
        content_imgs = [i for i in page_images[url] if img_counts[i] < min_pages and not re.search(r"logo|icon|favicon|backtotop|_bg\.", i, re.I)]
        if len(md) < 300 and content_imgs:
            report["image_only_pages"].append({"url": url, "images": content_imgs[:5]})

    records = []
    for url, md in html_md.items():
        rec = manifest[url]
        records.append({"url": url, "title": html_titles[url], "source_kind": "html", "text": md, "rec": rec, "extra": {}})

    office = MarkItDown()
    for url, rec in manifest.items():
        if rec.get("kind") != "doc" or not (ROOT / rec["path"]).exists():
            continue
        path = ROOT / rec["path"]
        try:
            if path.suffix == ".pdf":
                md, extra = parse_pdf(path)
            elif path.suffix in {".docx", ".pptx", ".xlsx"}:
                md, extra = parse_office(path, office)
            else:
                report["unparsed_formats"].append(url)
                continue
        except Exception as e:  # noqa: BLE001
            report["failed"].append({"url": url, "error": repr(e)[:200]})
            continue
        if extra.get("needs_ocr"):
            report["needs_ocr"].append(url)
        title = best_doc_title(rec, md, extra.pop("meta_title", ""))
        records.append({"url": url, "title": title, "source_kind": path.suffix.lstrip("."), "text": md, "rec": rec, "extra": extra})

    for f in sorted(MANUAL.glob("*.md")) if MANUAL.exists() else []:
        text = f.read_text(encoding="utf-8").strip()
        src = re.search(r"^Source:\s*(\S+)", text, re.M)
        title = re.search(r"^#\s+(.+)$", text, re.M)
        url = src.group(1) if src else f"manual://{f.name}"
        rec = dict(manifest.get(normalize_url(url), {}), section="manual")
        records.append({"url": url, "title": title.group(1).strip() if title else f.stem, "source_kind": "manual",
                        "text": tidy(text), "rec": rec, "extra": {}, "id_key": f"manual://{f.name}"})

    if common_lines:
        home = next((u for u in manifest if urlparse(u).path in {"", "/"}), next(iter(manifest)))
        records.append({
            "url": home + "#site-wide", "title": "Site-wide header and footer content", "source_kind": "html",
            "text": "\n".join(common_lines), "rec": {"section": "home", "fetched_at": manifest[home].get("fetched_at")},
            "extra": {},
        })

    with open(OUT / "corpus.jsonl", "w", encoding="utf-8") as out:
        for r in records:
            text, rec = r["text"], r["rec"]
            if len(text) < MIN_CHARS:
                report["too_short_skipped"].append(r["url"])
                continue
            dates, latest_year = dates_in(text)
            doc = {
                "id": doc_id(r.get("id_key", r["url"])),
                "url": r["url"],
                "title": r["title"],
                "doc_type": classify(r["url"], r["title"]),
                "section": rec.get("section"),
                "source_kind": r["source_kind"],
                "linked_from": list(dict.fromkeys(lt["from"] for lt in rec.get("link_texts", [])))[:5],
                "link_context": list(dict.fromkeys(lt["text"] for lt in rec.get("link_texts", []) if lt.get("text")))[:5],
                "dates_mentioned": dates,
                "latest_year": latest_year,
                "last_modified": rec.get("last_modified"),
                "fetched_at": rec.get("fetched_at"),
                "n_chars": len(text),
                **r["extra"],
                "text": text,
            }
            out.write(json.dumps(doc, ensure_ascii=False) + "\n")
            header = {k: v for k, v in doc.items() if k != "text" and v not in (None, [], "")}
            (MD_DIR / f"{doc['id']}.md").write_text(
                "---\n" + "\n".join(f"{k}: {json.dumps(v, ensure_ascii=False)}" for k, v in header.items())
                + "\n---\n\n" + ("" if text.startswith("# ") else f"# {doc['title']}\n\n") + f"{text}\n",
                encoding="utf-8",
            )
            report["by_doc_type"].append(doc["doc_type"])
            if r["source_kind"] == "html" and len(text) < 300:
                report["thin_pages"].append(r["url"])

    summary = {
        "documents": len(report["by_doc_type"]),
        "by_doc_type": dict(Counter(report.pop("by_doc_type")).most_common()),
        "sitewide_lines_removed": common_lines,
        **report,
    }
    (OUT / "parse_report.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: (v if not isinstance(v, list) else len(v)) for k, v in summary.items()}, indent=1))


if __name__ == "__main__":
    main()
