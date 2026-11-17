"""CPU-only contract checks for A→B approved-row conversion."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from convert_approved import Approval, convert_row


def fixture(image: str, kind: str = "multiple-choice", answer: str = "B") -> dict:
    instruction = (
        'Options:\nA. 1\nB. 2\nC. 3\nD. 4\nThink step by step, then give your final answer on the last line in the form "Answer: X", where X is the option letter.'
        if kind == "multiple-choice"
        else 'Think step by step, then give your final answer on the last line in the form "Answer: <answer>".'
    )
    return {
        "id": "item-1", "source": "geo170k(qa)", "status": "ok", "flags": {},
        "question_type": kind,
        "images": [image],
        "messages": [
            {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
            {"role": "user", "content": [
                {"type": "text", "text": "Question: "},
                {"type": "image", "image": image},
                {"type": "text", "text": "\nHow many red squares?\n" + instruction},
            ]},
            {"role": "assistant", "content": [{"type": "text", "text": "I count two.\nAnswer: " + answer}]},
        ],
    }


class AdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "example.png").write_bytes(b"placeholder")

    def test_mc_structural_parity_and_provenance(self) -> None:
        rec = fixture("example.png")
        row, reason, images = convert_row(rec, Approval(rec["source"], rec["id"], "train"), self.root, "shard_1")
        self.assertIsNone(reason)
        self.assertEqual(row["answer"], "B")
        self.assertEqual(row["answer_type"], "mc")
        self.assertEqual(row["prompt"][1]["content"][1], {"type": "image"})
        self.assertEqual(row["choices"], {"A": "1", "B": "2", "C": "3", "D": "4"})
        self.assertEqual(row["dataset_revision"], "shard_1")
        self.assertEqual(images, [self.root / "example.png"])

    def test_open_answer(self) -> None:
        rec = fixture("example.png", "open", "8.0")
        row, reason, _ = convert_row(rec, Approval(None, rec["id"], "heldout"), self.root, "shard_1")
        self.assertIsNone(reason)
        self.assertEqual(row["answer_type"], "open")
        self.assertEqual(row["answer"], "8.0")
        self.assertEqual(row["split"], "heldout")

    def test_short_ecd_style_open_answers(self) -> None:
        for answer in ("Between 1550 and 1600.", "Urban Schools; 2020", "8 or 9"):
            rec = fixture("example.png", "open", answer)
            row, reason, _ = convert_row(rec, Approval(None, rec["id"], "train"), self.root, "shard_1")
            self.assertIsNone(reason)
            self.assertEqual(row["answer"], answer)

    def test_reject_long_open(self) -> None:
        rec = fixture("example.png", "open", "A sentence longer than the forty character limit for A data")
        row, reason, _ = convert_row(rec, Approval(None, rec["id"], "train"), self.root, "shard_1")
        self.assertIsNone(row)
        self.assertEqual(reason, "long_open_gold")

    def test_reject_non_ok_status(self) -> None:
        rec = fixture("example.png")
        rec["status"] = "short_no_reasoning"
        row, reason, _ = convert_row(rec, Approval(None, rec["id"], "train"), self.root, "shard_1")
        self.assertIsNone(row)
        self.assertEqual(reason, "status_short_no_reasoning")

    def test_reject_changed_image_order(self) -> None:
        rec = fixture("example.png")
        rec["messages"][1]["content"][1]["image"] = "other.png"
        row, reason, _ = convert_row(rec, Approval(None, rec["id"], "train"), self.root, "shard_1")
        self.assertIsNone(row)
        self.assertEqual(reason, "image_block_payload_mismatch")


if __name__ == "__main__":
    unittest.main()
