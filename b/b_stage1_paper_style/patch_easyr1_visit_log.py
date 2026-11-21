"""Add opt-in attempted/used prompt-ID logging to pinned EasyR1 trainer."""

from __future__ import annotations

import argparse
import hashlib
import subprocess
from pathlib import Path


HEAD = "a73346d028ba0335aa42da104972c86e6cd0fe69"
SOURCE_SHA = "cb69c6457729f4b9afcd47ef013e3e5affaa126381ee2f8777172df39fba9e2c"


def replace_once(text: str, old: str, new: str) -> str:
    count = text.count(old)
    if count != 1:
        raise ValueError(f"Expected one source match, found {count}: {old[:60]!r}")
    return text.replace(old, new)


def patch(root: Path) -> Path:
    head = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if head != HEAD:
        raise ValueError(f"EasyR1 HEAD differs: {head}")
    path = root / "verl/trainer/ray_trainer.py"
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != SOURCE_SHA:
        raise ValueError("EasyR1 ray_trainer.py hash differs from pinned source")
    source = raw.decode("utf-8")
    source = replace_once(
        source,
        '            new_batch: DataProto = DataProto.from_single_dict(batch_dict, meta_info=meta_info)\n',
        '            new_batch: DataProto = DataProto.from_single_dict(batch_dict, meta_info=meta_info)\n'
        '            visit_log = os.environ.get("MMDL_VISIT_LOG_PATH")\n'
        '            if visit_log:\n'
        '                if "sample_id" not in new_batch.non_tensor_batch:\n'
        '                    raise RuntimeError("MMDL visit log requires sample_id column")\n'
        '                with open(visit_log, "a", encoding="utf-8") as stream:\n'
        '                    for sample_id in new_batch.non_tensor_batch["sample_id"]:\n'
        '                        stream.write(json.dumps({"event": "attempt", "step": self.global_step,\n'
        '                                                 "try": num_try_make_batch, "sample_id": str(sample_id)}) + "\\n")\n',
    )
    source = replace_once(
        source,
        '                return batch[: self.config.data.rollout_batch_size * self.config.worker.rollout.n]\n',
        '                selected = batch[: self.config.data.rollout_batch_size * self.config.worker.rollout.n]\n'
        '                visit_log = os.environ.get("MMDL_VISIT_LOG_PATH")\n'
        '                if visit_log:\n'
        '                    uid_to_sample = {}\n'
        '                    for uid, sample_id in zip(selected.non_tensor_batch["uid"],\n'
        '                                              selected.non_tensor_batch["sample_id"]):\n'
        '                        uid_to_sample.setdefault(str(uid), str(sample_id))\n'
        '                    with open(visit_log, "a", encoding="utf-8") as stream:\n'
        '                        for sample_id in uid_to_sample.values():\n'
        '                            stream.write(json.dumps({"event": "used", "step": self.global_step,\n'
        '                                                     "sample_id": sample_id}) + "\\n")\n'
        '                return selected\n',
    )
    path.write_text(source, encoding="utf-8", newline="")
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("easyr1_root", type=Path)
    print(patch(parser.parse_args().easyr1_root.resolve(strict=True)))
