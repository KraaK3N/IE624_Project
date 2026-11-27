import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from src.knowledge import load_documents


class KnowledgeTests(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.baseline = self.root / "corpus.jsonl"
        self.sources = self.root / "sources.json"
        self.page = {
            "id": "stable-id",
            "url": "https://www.sc.iitb.ac.in/courses/",
            "title": "Coursework",
            "doc_type": "course",
            "source_kind": "html",
            "text": "The baseline page.",
        }

    def set_baseline(self, *docs):
        self.baseline.write_text("\n".join(json.dumps(doc) for doc in docs), encoding="utf-8")

    def test_curated_source_replaces_baseline_by_normalized_url(self):
        self.set_baseline(self.page)
        (self.root / "course.md").write_text("## Credit requirements\n\n24 credits.", encoding="utf-8")
        citation = "http://sc.iitb.ac.in/courses#credits"
        self.sources.write_text(json.dumps([{
            "file": "course.md", "url": citation, "title": "Updated Coursework",
            "doc_type": "course", "tier": "core",
        }]), encoding="utf-8")

        [doc] = load_documents(self.baseline, self.sources)
        self.assertEqual(doc["id"], self.page["id"])
        self.assertEqual(doc["url"], citation)
        self.assertEqual(doc["tier"], "core")
        self.assertIn("24 credits.", doc["text"])

    def test_legacy_transcription_keeps_identity_of_crawled_page(self):
        manual = {**self.page, "id": "manual-id", "source_kind": "manual", "text": "Transcribed timetable."}
        for order in ((self.page, manual), (manual, self.page)):
            with self.subTest(first=order[0]["source_kind"]):
                self.set_baseline(*order)
                [doc] = load_documents(self.baseline, None)
                self.assertEqual(doc["id"], self.page["id"])
                self.assertEqual(doc["text"], manual["text"])


if __name__ == "__main__":
    unittest.main()
