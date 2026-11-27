"""Paths shared by the site crawler and corpus parser."""

from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RAW = DATA / "raw"


def raw_file(stored_path: str, raw_dir: Path = RAW) -> Path:
    # Older manifests contain Windows separators in repository-relative paths.
    path = Path(stored_path.replace("\\", "/"))
    if path.parts[:2] == ("data", "raw"):
        return raw_dir.joinpath(*path.parts[2:])
    return ROOT / path
