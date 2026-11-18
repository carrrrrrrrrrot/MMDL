"""Small integration check against A's actual handoff; no model or GPU use."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import unittest
from pathlib import Path

from convert_mammoth_handoff import convert, read_approvals


IDS = {
    "mammoth_6155522",  # four-choice MC
    "mammoth_1308788",  # five-choice MC
    "mammoth_2827742",  # numeric
    "mammoth_3028318",  # short word
}
BAD_IDS = {
    "mammoth_8768873": "residual_image_marker",
    "mammoth_6572627": "embedded_answer_line",
}
COMMIT = "a61adb899d3c2274ba98cba7a360c5b4f333979f"


def _rows(path: Path) -> dict[str, dict]:
    found = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row["id"] in IDS | set(BAD_IDS):
                found[row["id"]] = row
    if set(found) != IDS | set(BAD_IDS):
        raise AssertionError("Required A sample IDs are absent")
    return found


def _review(path: Path, rows: list[dict], leakage: str = "pass") -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps({
                "source": row["source"], "id": row["id"],
                "approval": "approved", "mmmu_leakage": leakage,
                "quality": "pass", "split": "train", "reviewer": "mock-test-only",
            }) + "\n")


class HandoffIntegration(unittest.TestCase):
    list_path: Path
    image_root: Path
    b_grpo_main: Path

    @classmethod
    def setUpClass(cls) -> None:
        cls.samples = _rows(cls.list_path)

    def test_real_samples_and_b_loader(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            review = root / "mock_c_review.jsonl"
            _review(review, [self.samples[item] for item in sorted(IDS)])
            manifest = root / "manifest.jsonl"
            report = convert(self.list_path, review, self.image_root, manifest, "test-A-shard1", COMMIT)
            self.assertEqual(report["converted_count"], 4)
            self.assertEqual(report["source_count"], 3000)
            self.assertEqual(report["answer_type_distribution"], {"mc": 2, "numeric": 1, "word": 1})
            output = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(output), 4)
            five = next(row for row in output if row["original_id"] == "mammoth_1308788")
            self.assertEqual(list(five["choices"]), list("ABCDE"))
            self.assertTrue(all(row["c_approved"] for row in output))
            self.assertTrue(all(row["c_reviewer"] == "mock-test-only" for row in output))
            self.assertTrue(all("<image" not in str(row["prompt"]) for row in output))
            sys.path.insert(0, str(self.b_grpo_main))
            from data import load_manifest
            dataset, summary = load_manifest(manifest)
            self.assertEqual(len(dataset), 4)
            self.assertEqual(summary["answer_types"], {"mc": 2, "open": 2})

    def test_missing_leakage_pass_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            review = Path(directory) / "bad_review.jsonl"
            _review(review, [self.samples["mammoth_6155522"]], leakage="unchecked")
            with self.assertRaisesRegex(ValueError, "leakage audit must be pass"):
                read_approvals(review)

    def test_question_quality_issues_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for original_id, reason in BAD_IDS.items():
                with self.subTest(original_id=original_id):
                    review = root / f"{original_id}.jsonl"
                    _review(review, [self.samples[original_id]])
                    manifest = root / f"{original_id}.manifest.jsonl"
                    with self.assertRaisesRegex(ValueError, reason):
                        convert(self.list_path, review, self.image_root, manifest, "test-A-shard1", COMMIT)
                    self.assertFalse(manifest.exists())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--list-jsonl", required=True, type=Path)
    parser.add_argument("--image-root", required=True, type=Path)
    parser.add_argument(
        "--b-grpo-main", type=Path,
        default=Path(__file__).resolve().parents[1] / "b_grpo_main",
    )
    args, remaining = parser.parse_known_args()
    HandoffIntegration.list_path = args.list_jsonl
    HandoffIntegration.image_root = args.image_root
    HandoffIntegration.b_grpo_main = args.b_grpo_main
    unittest.main(argv=[sys.argv[0], *remaining])
