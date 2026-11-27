"""Extract native PDF text as Markdown for both ingestion tools."""

import re
from pathlib import Path


TABLE_SEPARATOR = re.compile(r"\|(\s*:?-{3,}:?\s*\|)+")


def tidy_markdown(markdown: str) -> str:
    source = markdown.replace("\xa0", " ").replace("\u200b", "").splitlines()
    lines = []
    for index, line in enumerate(source):
        line = line.rstrip()
        empty_row = re.fullmatch(r"\|(\s*\|)+\s*", line)
        if empty_row and index + 2 < len(source) and TABLE_SEPARATOR.fullmatch(source[index + 1].strip()):
            source[index + 1], source[index + 2] = source[index + 2], source[index + 1]
            continue
        if empty_row or re.fullmatch(r"[-*_=\s]{3,}", line):
            continue
        lines.append(line if line.lstrip().startswith("|") else re.sub(r"[ \t]{2,}", " ", line))
    markdown = re.sub(r"\[\s*\]\([^)]*\)", "", "\n".join(lines))
    return re.sub(r"\n{3,}", "\n\n", markdown).strip()


def parse_pdf(path: Path) -> tuple[str, dict]:
    import pymupdf
    import pymupdf4llm

    text_pages = []
    image_only_pages = []
    with pymupdf.open(path) as document:
        page_count = document.page_count
        title = (document.metadata or {}).get("title", "")
        for page in document:
            native_text = page.get_text().strip()
            images = page.get_image_info() if len(native_text) < 80 else ()
            image_dominates = any(
                (page.rect & pymupdf.Rect(image["bbox"])).get_area() > page.rect.get_area() / 2
                for image in images
            )
            if images and (not native_text or image_dominates):
                image_only_pages.append(page.number + 1)
            elif native_text:
                text_pages.append(page.number)
        markdown = ""
        if text_pages:
            markdown = tidy_markdown(
                pymupdf4llm.to_markdown(document, pages=text_pages, show_progress=False, use_ocr=False)
            )

    return markdown, {
        "pages": page_count,
        "meta_title": title,
        "needs_ocr": not markdown or bool(image_only_pages),
        "image_only_pages": image_only_pages,
    }
