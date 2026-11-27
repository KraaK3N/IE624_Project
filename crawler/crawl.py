"""Crawl sc.iitb.ac.in and store raw pages and documents.

Raw HTML is kept untouched so the parser (python -m crawler.parse) can be re-tuned
without re-crawling. Re-running resumes and only re-downloads changed files.

    python -m crawler.crawl --max-pages 50    # quick test
    python -m crawler.crawl                   # full crawl / incremental update
    python -m crawler.crawl --fresh           # ignore previous manifest

Outputs (data/raw/):
    html/*.html            raw pages
    docs/*                 PDFs and office documents
    manifest.json          one record per URL (status, hashes, link context, ...)
    external_links.csv     links leaving the site, for manual review
    crawl_report.json      summary: counts, errors, robots-blocked, thin pages
"""

import argparse
import csv
import hashlib
import json
import re
import sys
import time
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import parse_qsl, unquote, urlencode, urljoin, urlparse, urlunparse
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from crawler.paths import RAW, ROOT, raw_file

START_URLS = ["https://www.sc.iitb.ac.in/"]
CANONICAL_HOST = "www.sc.iitb.ac.in"
SITE_HOSTS = {CANONICAL_HOST, "sc.iitb.ac.in"}
DOC_HOST_SUFFIX = ".iitb.ac.in"  # docs linked from SysCon pages on other IITB hosts are fetched (not crawled)
USER_AGENT = "IE624-SysCon-Chatbot-Crawler/2.0 (IIT Bombay course project)"

DOC_EXT = {".pdf", ".doc", ".docx", ".ppt", ".pptx", ".xls", ".xlsx", ".odt", ".ods", ".odp", ".rtf"}
SKIP_EXT = {
    ".jpg", ".jpeg", ".png", ".gif", ".svg", ".webp", ".ico", ".bmp", ".tif", ".tiff",
    ".mp4", ".mp3", ".avi", ".mov", ".wmv", ".webm", ".wav", ".zip", ".rar", ".gz", ".tar", ".7z",
    ".css", ".js", ".json", ".xml", ".rss", ".woff", ".woff2", ".ttf", ".eot", ".exe", ".dmg", ".iso",
}
DOC_CTYPES = {
    "application/pdf": ".pdf",
    "application/msword": ".doc",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
    "application/vnd.ms-powerpoint": ".ppt",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": ".pptx",
    "application/vnd.ms-excel": ".xls",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": ".xlsx",
}
TRAP_RE = re.compile(
    r"(login|logout|signin|signup|wp-admin|wp-login|/feed/?$|[?&](print|share|replytocom|sort|order|"
    r"orderby|lang|format)=|/calendar|ical|/tag/|/author/|mailto:|javascript:|tel:)",
    re.I,
)
TRACKING_PARAMS = re.compile(r"^(utm_\w+|fbclid|gclid|sessionid|phpsessid|sid|_ga)$", re.I)
GENERIC_ANCHOR = re.compile(r"^(click\s*here|here|link|download|more|read\s*more|details?|view|pdf|this)\W*$", re.I)

MAX_DEPTH = 10
MAX_QUERY_VARIANTS = 30
MAX_DOC_BYTES = 60 * 1024 * 1024
SAVE_EVERY = 25

HTML_DIR, DOC_DIR = RAW / "html", RAW / "docs"
MANIFEST, EXTERNAL, REPORT = RAW / "manifest.json", RAW / "external_links.csv", RAW / "crawl_report.json"


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def normalize(url: str) -> str | None:
    try:
        p = urlparse(url.strip())
    except ValueError:
        return None
    if p.scheme not in {"http", "https"} or not p.netloc:
        return None
    host = p.netloc.lower().split("@")[-1]
    if host.endswith(":80") or host.endswith(":443"):
        host = host.rsplit(":", 1)[0]
    scheme = p.scheme
    if host in SITE_HOSTS:
        host, scheme = CANONICAL_HOST, "https"
    path = re.sub(r"/{2,}", "/", unquote(p.path) or "/")
    if path != "/" and path.endswith("/"):
        path = path[:-1]
    query = urlencode(sorted((k, v) for k, v in parse_qsl(p.query, keep_blank_values=True) if not TRACKING_PARAMS.match(k)))
    return urlunparse((scheme, host, path, "", query, ""))


def is_site(url: str) -> bool:
    return urlparse(url).netloc in SITE_HOSTS


def ext_of(url: str) -> str:
    return Path(urlparse(url).path).suffix.lower()


def slug(url: str) -> str:
    p = urlparse(url)
    base = re.sub(r"[^a-zA-Z0-9]+", "_", f"{p.path}_{p.query}").strip("_")[:90] or "index"
    return f"{base}_{hashlib.sha1(url.encode()).hexdigest()[:10]}"


def section(url: str) -> str:
    parts = [s for s in urlparse(url).path.split("/") if s]
    if parts and "." in parts[-1]:
        parts = parts[:-1]
    return parts[0].lower() if parts else "home"


def make_session() -> requests.Session:
    s = requests.Session()
    retry = Retry(
        total=4, connect=4, read=3, backoff_factor=1.5,
        status_forcelist=(429, 500, 502, 503, 504), allowed_methods=("GET", "HEAD"),
        respect_retry_after_header=True,
    )
    s.mount("http://", HTTPAdapter(max_retries=retry))
    s.mount("https://", HTTPAdapter(max_retries=retry))
    s.headers.update({"User-Agent": USER_AGENT, "Accept-Language": "en"})
    return s


def load_robots(session: requests.Session, base: str) -> RobotFileParser:
    rp = RobotFileParser()
    try:
        r = session.get(urljoin(base, "/robots.txt"), timeout=20)
        rp.parse(r.text.splitlines() if r.status_code == 200 else [])
    except requests.RequestException as e:
        print(f"[WARN] robots.txt unreachable ({e}); proceeding as allowed")
        rp.parse([])
    return rp


def sitemap_urls(session: requests.Session, rp: RobotFileParser, base: str, limit: int = 20) -> set[str]:
    todo = deque(set(rp.site_maps() or []) | {urljoin(base, "/sitemap.xml")})
    seen, found = set(), set()
    while todo and len(seen) < limit:
        sm = todo.popleft()
        if sm in seen:
            continue
        seen.add(sm)
        try:
            r = session.get(sm, timeout=20)
            if r.status_code != 200 or b"<" not in r.content[:200]:
                continue
            root = ET.fromstring(r.content)
        except (requests.RequestException, ET.ParseError):
            continue
        for loc in root.iter():
            if loc.tag.endswith("loc") and loc.text:
                (todo.append if root.tag.endswith("sitemapindex") else found.add)(loc.text.strip())
    return {u for u in map(normalize, found) if u and is_site(u)}


def link_context(a) -> str:
    text = " ".join(a.get_text(" ", strip=True).split())
    if text and not GENERIC_ANCHOR.match(text) and len(text) > 3:
        return text[:300]
    for parent in a.parents:
        if parent.name in {"li", "p", "tr", "td", "dd", "h1", "h2", "h3", "h4", "div", "section"}:
            ctx = " ".join(parent.get_text(" ", strip=True).split())
            if len(ctx) > len(text) + 5:
                return ctx[:300]
    return text or (a.get("title") or "")[:300]


class Crawler:
    def __init__(self, max_pages: int, delay: float, ignore_robots: bool, fresh: bool):
        self.max_pages, self.delay, self.ignore_robots = max_pages, delay, ignore_robots
        self.session = make_session()
        self.manifest: dict[str, dict] = {} if fresh or not MANIFEST.exists() else json.loads(MANIFEST.read_text(encoding="utf-8"))
        self.queue: deque[tuple[str, int, str | None]] = deque()
        self.seen: set[str] = set()
        self.content_hashes: dict[str, str] = {}
        self.doc_hashes: dict[str, str] = {}
        self.query_variants: Counter = Counter()
        self.links_to: dict[str, list[dict]] = defaultdict(list)
        self.external: dict[str, dict] = {}
        self.robots_blocked: list[str] = []
        self.errors: list[dict] = []
        self.fetched = self.saved_at = 0
        self.last_request = 0.0

    # ---------- queue ----------
    def enqueue(self, url: str, depth: int, parent: str | None) -> None:
        if url in self.seen or depth > MAX_DEPTH or TRAP_RE.search(url):
            return
        p = urlparse(url)
        if p.query:
            key = f"{p.netloc}{p.path}"
            if self.query_variants[key] >= MAX_QUERY_VARIANTS:
                return
            self.query_variants[key] += 1
        self.seen.add(url)
        self.queue.append((url, depth, parent))

    def handle_link(self, a, base_url: str, depth: int) -> None:
        href = (a.get("href") or "").strip()
        if not href or href.startswith(("#", "mailto:", "tel:", "javascript:")):
            return
        url, page_url = normalize(urljoin(base_url, href)), normalize(base_url)
        if not url or url == page_url:
            return
        ext, ctx = ext_of(url), link_context(a)
        if ext in SKIP_EXT:
            return
        if len(self.links_to[url]) < 8:
            self.links_to[url].append({"from": page_url, "text": ctx})
        host = urlparse(url).netloc
        if is_site(url):
            self.enqueue(url, depth + 1, page_url)
        elif ext in DOC_EXT and host.endswith(DOC_HOST_SUFFIX.lstrip(".")):
            self.enqueue(url, depth + 1, page_url)
        elif url not in self.external:
            self.external[url] = {"url": url, "host": host, "from": page_url, "text": ctx}

    # ---------- fetching ----------
    def polite_get(self, url: str, prev: dict | None, stream: bool):
        wait = self.delay - (time.monotonic() - self.last_request)
        if wait > 0:
            time.sleep(wait)
        headers = {}
        if prev and prev.get("status") == 200 and prev.get("path") and raw_file(prev["path"]).exists():
            if prev.get("etag"):
                headers["If-None-Match"] = prev["etag"]
            if prev.get("last_modified"):
                headers["If-Modified-Since"] = prev["last_modified"]
        try:
            return self.session.get(url, headers=headers, timeout=(15, 60), stream=stream, allow_redirects=True)
        finally:
            self.last_request = time.monotonic()

    def save_doc(self, url: str, r: requests.Response, ext: str) -> tuple[Path, str] | None:
        size = int(r.headers.get("Content-Length") or 0)
        if size > MAX_DOC_BYTES:
            self.errors.append({"url": url, "error": f"too large ({size} bytes)"})
            return None
        cd = r.headers.get("Content-Disposition", "")
        m = re.search(r'filename\*?=(?:UTF-8\'\')?"?([^";]+)', cd)
        if m and Path(m.group(1)).suffix.lower() in DOC_EXT:
            ext = Path(m.group(1)).suffix.lower()
        path, h, total = DOC_DIR / f"{slug(url)}{ext}", hashlib.sha256(), 0
        with open(path, "wb") as f:
            for chunk in r.iter_content(64 * 1024):
                total += len(chunk)
                if total > MAX_DOC_BYTES:
                    f.close()
                    path.unlink(missing_ok=True)
                    self.errors.append({"url": url, "error": "too large (streamed)"})
                    return None
                h.update(chunk)
                f.write(chunk)
        return path, h.hexdigest()

    def process(self, url: str, depth: int, parent: str | None) -> None:
        if not self.ignore_robots and not self.robots.can_fetch(USER_AGENT, url):
            self.robots_blocked.append(url)
            return
        prev = self.manifest.get(url)
        ext = ext_of(url)
        likely_doc = ext in DOC_EXT
        try:
            r = self.polite_get(url, prev, stream=True)
        except requests.RequestException as e:
            self.errors.append({"url": url, "error": str(e)[:200]})
            print(f"[ERR] {url}: {e.__class__.__name__}")
            return
        self.fetched += 1

        rec = {
            "url": url, "parent": parent, "depth": depth, "host": urlparse(url).netloc,
            "section": section(url) if is_site(url) else "external_doc",
            "fetched_at": now(), "status": r.status_code,
        }

        if r.status_code == 304 and prev:
            r.close()
            prev.update(fetched_at=rec["fetched_at"], depth=depth, unchanged=True)
            (self.content_hashes if prev["kind"] == "html" else self.doc_hashes).setdefault(prev.get("sha256", ""), url)
            if prev["kind"] == "html":
                self.extract_links(raw_file(prev["path"]).read_bytes(), prev.get("base_url", url), depth, prev.get("encoding"))
            print(f"[304] {url}")
            return

        final = normalize(r.url) or url
        if final != url:
            rec["final_url"] = final
            if final in self.seen:
                rec.update(kind="redirect", duplicate_of=final)
                self.manifest[url] = rec
                r.close()
                self.enqueue(final, depth, url)
                return
            self.seen.add(final)

        if r.status_code != 200:
            r.close()
            rec["kind"] = "error"
            self.manifest[url] = rec
            self.errors.append({"url": url, "error": f"HTTP {r.status_code}", "parent": parent})
            print(f"[{r.status_code}] {url}")
            return

        ctype = r.headers.get("Content-Type", "").split(";")[0].strip().lower()
        rec.update(
            content_type=ctype, etag=r.headers.get("ETag"), last_modified=r.headers.get("Last-Modified"),
        )

        doc_ext = DOC_CTYPES.get(ctype) or (ext if likely_doc and ctype in {"application/octet-stream", "binary/octet-stream", ""} else None)
        if doc_ext:
            saved = self.save_doc(url, r, doc_ext if not likely_doc or ext == doc_ext else ext)
            if saved:
                path, digest = saved
                if digest in self.doc_hashes and self.doc_hashes[digest] != url:
                    path.unlink(missing_ok=True)
                    rec.update(kind="duplicate", duplicate_of=self.doc_hashes[digest])
                    self.manifest[url] = rec
                    return
                self.doc_hashes[digest] = url
                rec.update(kind="doc", path=path.relative_to(ROOT).as_posix(), sha256=digest, ext=path.suffix, bytes=path.stat().st_size)
                self.manifest[url] = rec
                print(f"[DOC] {url}")
            return

        if "html" not in ctype:
            r.close()
            return
        if not is_site(final):
            r.close()
            return

        body = r.content
        encoding = r.encoding if r.encoding and r.encoding.lower() != "iso-8859-1" else r.apparent_encoding
        soup = BeautifulSoup(body, "lxml", from_encoding=encoding)
        base_tag = soup.find("base", href=True)
        base_url = urljoin(r.url, base_tag["href"]) if base_tag else r.url
        rec["base_url"] = base_url

        refresh = soup.find("meta", attrs={"http-equiv": re.compile(r"^refresh$", re.I)})
        m = re.search(r"url\s*=\s*['\"]?([^'\";]+)", refresh.get("content", ""), re.I) if refresh else None
        if m:
            target = normalize(urljoin(base_url, m.group(1).strip()))
            if target and target != final and (is_site(target) or ext_of(target) in DOC_EXT):
                rec["meta_refresh"] = target
                self.enqueue(target, depth, url)

        canon = soup.find("link", rel="canonical")
        if canon and canon.get("href"):
            c = normalize(urljoin(base_url, canon["href"]))
            if c and c != final and is_site(c):
                rec["canonical"] = c
                self.enqueue(c, depth, url)

        text = " ".join(soup.get_text(" ", strip=True).split())
        digest = hashlib.sha256(text.encode()).hexdigest()
        if digest in self.content_hashes and self.content_hashes[digest] != url:
            rec.update(kind="duplicate", duplicate_of=self.content_hashes[digest])
            self.manifest[url] = rec
            self.extract_links(body, base_url, depth, encoding, soup)
            return
        self.content_hashes[digest] = url

        path = HTML_DIR / f"{slug(url)}.html"
        path.write_bytes(body)
        title = soup.title.get_text(" ", strip=True) if soup.title else ""
        rec.update(
            kind="html", path=path.relative_to(ROOT).as_posix(), sha256=digest, encoding=encoding,
            title=title, text_chars=len(text),
        )
        self.manifest[url] = rec
        self.extract_links(body, base_url, depth, encoding, soup)
        print(f"[HTML d{depth}] {url}")

    def extract_links(self, body: bytes, page_url: str, depth: int, encoding: str | None, soup=None) -> None:
        soup = soup or BeautifulSoup(body, "lxml", from_encoding=encoding)
        for tag in soup.find_all(["a", "area"], href=True):
            self.handle_link(tag, page_url, depth)
        for tag in soup.find_all(["iframe", "embed", "object"]):
            src = tag.get("src") or tag.get("data")
            if src:
                tag["href"] = src
                self.handle_link(tag, page_url, depth)

    # ---------- run ----------
    def save(self) -> None:
        for url, rec in self.manifest.items():
            if url in self.links_to:
                rec["link_texts"] = self.links_to[url]
        tmp = MANIFEST.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.manifest, indent=1, ensure_ascii=False), encoding="utf-8")
        tmp.replace(MANIFEST)

    def run(self) -> None:
        for d in (HTML_DIR, DOC_DIR):
            d.mkdir(parents=True, exist_ok=True)
        base = START_URLS[0]
        self.robots = load_robots(self.session, base)
        crawl_delay = self.robots.crawl_delay(USER_AGENT)
        if crawl_delay and not self.ignore_robots:
            self.delay = max(self.delay, float(crawl_delay))
        if not self.ignore_robots and not self.robots.can_fetch(USER_AGENT, base):
            sys.exit(
                "robots.txt disallows crawling the homepage for this user agent.\n"
                "Ask the SysCon webteam for permission, then re-run with --ignore-robots."
            )

        for u in START_URLS:
            self.enqueue(normalize(u), 0, None)
        for u in sorted(sitemap_urls(self.session, self.robots, base)):
            self.enqueue(u, 1, "sitemap")
        print(f"Seeds: {len(self.queue)} | delay {self.delay}s | robots {'IGNORED' if self.ignore_robots else 'respected'}")

        try:
            while self.queue and self.fetched < self.max_pages:
                self.process(*self.queue.popleft())
                if self.fetched - self.saved_at >= SAVE_EVERY:
                    self.save()
                    self.saved_at = self.fetched
        except KeyboardInterrupt:
            print("\nInterrupted - saving progress (re-run to resume).")
        finally:
            self.save()
            self.write_outputs()

    def write_outputs(self) -> None:
        with open(EXTERNAL, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["url", "host", "from", "text"])
            w.writeheader()
            w.writerows(sorted(self.external.values(), key=lambda x: (x["host"], x["url"])))

        recs = list(self.manifest.values())
        kinds = Counter(r.get("kind") for r in recs)
        report = {
            "finished_at": now(),
            "fetched_this_run": self.fetched,
            "queue_remaining": len(self.queue),
            "by_kind": dict(kinds),
            "html_by_section": dict(Counter(r["section"] for r in recs if r.get("kind") == "html").most_common()),
            "docs_by_ext": dict(Counter(r.get("ext") for r in recs if r.get("kind") == "doc").most_common()),
            "thin_html_pages": sorted(r["url"] for r in recs if r.get("kind") == "html" and r.get("text_chars", 0) < 300),
            "robots_blocked": self.robots_blocked,
            "errors": self.errors,
            "external_hosts": dict(Counter(e["host"] for e in self.external.values()).most_common()),
        }
        REPORT.write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
        print(
            f"\nDone. html={kinds.get('html', 0)} docs={kinds.get('doc', 0)} errors={len(self.errors)} "
            f"robots_blocked={len(self.robots_blocked)} external_links={len(self.external)} "
            f"queue_left={len(self.queue)}\nSee {REPORT.relative_to(ROOT)}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--max-pages", type=int, default=5000, help="max HTTP fetches this run")
    parser.add_argument("--delay", type=float, default=1.0, help="seconds between requests")
    parser.add_argument("--fresh", action="store_true", help="ignore existing manifest")
    parser.add_argument("--ignore-robots", action="store_true", help="only with the SysCon webteam's permission")
    args = parser.parse_args()
    Crawler(args.max_pages, args.delay, args.ignore_robots, args.fresh).run()


if __name__ == "__main__":
    main()
