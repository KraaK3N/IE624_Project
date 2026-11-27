import re
import unittest

from src.chunking import Chunker, build_chunks, tier_of


class Tokenizer:
    def encode(self, text, *, add_special_tokens, verbose=False):
        tokens = re.findall(r"\w+|[^\w\s]", text)
        return ["[CLS]", *tokens, "[SEP]"] if add_special_tokens else tokens


class ChunkerTests(unittest.TestCase):
    def setUp(self):
        self.chunker = Chunker(Tokenizer())

    def build(self, text, **overrides):
        doc = {
            "id": "handbook",
            "url": "https://example.org/handbook",
            "title": "Student Handbook",
            "source_kind": "markdown",
            "doc_type": "rules",
            "tier": "core",
            "text": text,
            **overrides,
        }
        chunks, _ = build_chunks([doc], self.chunker, {"core", "research"})
        return chunks

    def test_short_faq_answers_stay_with_their_own_headings(self):
        chunks = self.build(
            "# Registration\n\n## Is approval required?\n\nYes.\n\n"
            "## Is there a fee?\n\nYes.",
            source_kind="pdf",
        )
        self.assertEqual([chunk["section_path"] for chunk in chunks], [
            "Registration > Is approval required?",
            "Registration > Is there a fee?",
        ])
        self.assertIn("## Is approval required?\n\nYes.", chunks[0]["text"])
        self.assertIn("## Is there a fee?\n\nYes.", chunks[1]["text"])
        self.assertEqual(chunks[1]["doc_id"], "handbook")
        self.assertEqual(chunks[1]["url"], "https://example.org/handbook")
        self.assertEqual(chunks[1]["tier"], "core")

    def test_long_prose_carries_at_most_fifty_tokens(self):
        sentences = [
            f"Course {index} lists its prerequisites and deadlines on the departmental website."
            for index in range(80)
        ]
        chunks = self.build("## Registration\n\n" + " ".join(sentences))
        self.assertGreater(len(chunks), 2)
        for previous, next_chunk in zip(chunks, chunks[1:]):
            overlap, fresh = next_chunk["text"].split("\n\n", 1)
            self.assertTrue(previous["text"].endswith(overlap))
            self.assertTrue(fresh.startswith("Course "))
            self.assertLessEqual(self.chunker.n(overlap), 50)
        self.assertTrue(all(chunk["n_tokens"] <= 350 for chunk in chunks))
        for sentence in sentences:
            self.assertTrue(any(sentence in chunk["text"] for chunk in chunks))

    def test_short_answer_is_valid_even_when_heading_adds_punctuation(self):
        chunks = self.build("## Q?\n\nNo.", source_kind="pdf")
        self.assertEqual([chunk["text"] for chunk in chunks], ["## Q?\n\nNo."])

    def test_short_curated_pdf_notice_is_searchable(self):
        chunks = self.build("Registration closes on Monday.", source_kind="pdf")
        self.assertEqual([chunk["text"] for chunk in chunks], ["Registration closes on Monday."])

    def test_split_table_repeats_the_header_and_keeps_whole_rows(self):
        header = "Course | Term | Deadline\n--- | --- | ---"
        rows = [f"Course {index} | Autumn | October {index}" for index in range(60)]
        chunks = self.build("## Registration\n\n" + header + "\n" + "\n".join(rows))
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(header in chunk["text"] for chunk in chunks))
        self.assertEqual(
            [line for chunk in chunks for line in chunk["text"].splitlines() if line in rows],
            rows,
        )

    def test_embedding_budget_includes_metadata_and_special_tokens(self):
        section = " ".join(["Eligibility"] * 600)
        chunks = self.build(f"## {section}\n\n" + "Check the registration guidelines. " * 80)
        self.assertTrue(chunks)
        for chunk in chunks:
            self.assertEqual(chunk["section_path"], section)
            self.assertTrue(chunk["embed_text"].startswith("Student Handbook > Eligibility"))
            self.assertEqual(
                chunk["embedding_tokens"],
                self.chunker.n(chunk["embed_text"], special_tokens=True),
            )
            self.assertLessEqual(chunk["embedding_tokens"], 512)

    def test_legacy_tier_inference_and_explicit_override(self):
        doc = {"url": "https://example.org/~faculty/course.pdf", "source_kind": "pdf"}
        self.assertEqual(tier_of(doc), "research")
        self.assertEqual(tier_of({**doc, "tier": "core"}), "core")

    def test_pdf_form_fields_do_not_hide_the_form_text(self):
        body = (
            "I authorise the person named in this form to receive the amount recorded below "
            "on behalf of the department and its students. Name: " + "_" * 190
        )
        chunks = self.build(body, source_kind="pdf")
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["text"], body)


if __name__ == "__main__":
    unittest.main()
