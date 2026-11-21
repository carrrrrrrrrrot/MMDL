"""Summarize six EasyR1 Stage1 smoke runs from logs and saved artifacts.

The five-row holdout validation is a pipeline smoke check, not a performance
estimate. DAPO's ``reward.accuracy`` is measured over attempted rollouts,
whereas ``critic.rewards.mean`` describes the selected optimizer batch.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import re
import struct
import tempfile
from pathlib import Path


ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
RUNNER = re.compile(r"\(Runner pid=\d+\)\s?")
TOP = re.compile(r"^([A-Za-z][A-Za-z0-9_]*):$")
VALUE = re.compile(r"^  ([A-Za-z][A-Za-z0-9_]*):\s+([^\s]+)$")
ADAPTER_NAME = re.compile(
    r"base_model\.model\.model\.language_model\.layers\.\d+\."
    r"self_attn\.(?:q|k|v|o)_proj\.lora_[AB]\.weight"
)
RUNS = [(method, cap) for cap in (512, 1024, 2048) for method in ("grpo", "dapo_core")]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def runner_lines(path: Path) -> list[str]:
    """Ignore Ray worker/progress output, preserving the Runner metric indent."""
    lines = []
    for raw in ANSI.sub("", path.read_text(encoding="utf-8", errors="replace")).replace("\r", "\n").splitlines():
        match = RUNNER.search(raw)
        if match:
            lines.append(raw[match.end():].rstrip())
    return lines


def metric_block(lines: list[str], index: int) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    section = None
    for line in lines[index + 1:]:
        if line in {"Step 1", "Start validation...", "Final validation metrics:", "Finish validation."}:
            break
        top = TOP.fullmatch(line)
        if top:
            section = top.group(1)
            result.setdefault(section, {})
            continue
        value = VALUE.fullmatch(line)
        if value and section:
            try:
                number = float(value.group(2))
            except ValueError:
                continue
            if math.isfinite(number):
                result[section][value.group(1)] = number
    return result


def parse_console(path: Path) -> dict:
    lines = runner_lines(path)
    blocks = [metric_block(lines, index) for index, line in enumerate(lines) if line == "Step 1"]
    train = next((block for block in blocks if "actor" in block and "reward" in block), {})
    validation = next((block for block in blocks if "val" in block), {})
    result = {
        "training_step_observed": bool(train),
        "final_validation_observed": "Final validation metrics:" in lines,
        "train_raw_accuracy_attempted": train.get("reward", {}).get("accuracy"),
        "train_answer_present_attempted": train.get("reward", {}).get("answer_present"),
        "train_at_cap_attempted": train.get("reward", {}).get("at_cap_proxy"),
        "train_overlong_reward": train.get("reward", {}).get("overlong"),
        "train_selected_reward_mean": selected_reward_mean(lines),
        "train_response_mean_tokens": train.get("response_length", {}).get("mean"),
        "train_response_max_tokens": train.get("response_length", {}).get("max"),
        "grad_norm": train.get("actor", {}).get("grad_norm"),
        "step_seconds": train.get("perf", {}).get("time_per_step"),
        "validation5_accuracy": validation.get("val", {}).get("accuracy_reward"),
        "validation5_answer_present": validation.get("val", {}).get("answer_present_reward"),
    }
    return result


def selected_reward_mean(lines: list[str]) -> float | None:
    """Find ``critic.rewards.mean`` in the first training Step 1 block."""
    in_step = False
    in_critic = False
    in_rewards = False
    for line in lines:
        if line == "Step 1":
            if in_step:
                break
            in_step = True
            continue
        if not in_step:
            continue
        if line == "Start validation...":
            break
        top = TOP.fullmatch(line)
        if top:
            in_critic = top.group(1) == "critic"
            in_rewards = False
        elif in_critic and line == "  rewards:":
            in_rewards = True
        elif in_rewards and line.startswith("  ") and not line.startswith("    "):
            in_rewards = False
        elif in_rewards:
            match = re.fullmatch(r"    mean:\s+([^\s]+)", line)
            if match:
                return float(match.group(1))
    return None


def load_train_ids(path: Path) -> set[str]:
    ids = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                ids.append(str(json.loads(line)["sample_id"]))
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate sample_id in {path}")
    return set(ids)


def parse_visits(path: Path, train_ids: set[str], rollout_count: int = 5) -> dict:
    counts = {"attempt": collections.Counter(), "used": collections.Counter()}
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            event = json.loads(line)
            kind = event.get("event")
            if kind not in counts:
                raise ValueError(f"Unknown visit event at {path}:{number}")
            sample_id = str(event["sample_id"])
            if sample_id not in train_ids:
                raise ValueError(f"Unknown train ID at {path}:{number}: {sample_id}")
            counts[kind][sample_id] += 1
    if any(visits > counts["attempt"][sample_id] for sample_id, visits in counts["used"].items()):
        raise ValueError(f"Used visits exceed attempts in {path}")
    attempted = sum(counts["attempt"].values())
    used = sum(counts["used"].values())
    return {
        "attempted_prompt_visits": attempted,
        "used_prompt_visits": used,
        "attempted_unique_ids": len(counts["attempt"]),
        "used_unique_ids": len(counts["used"]),
        "attempted_rollouts_if_complete": attempted * rollout_count,
        "used_rollouts_if_complete": used * rollout_count,
        "attempt_visit_histogram": dict(sorted(collections.Counter(counts["attempt"].values()).items())),
        "used_visit_histogram": dict(sorted(collections.Counter(counts["used"].values()).items())),
    }


def parse_trainable(path: Path) -> dict:
    report = json.loads(path.read_text(encoding="utf-8"))
    entries = report["names_and_numel"]
    names = [item[0] for item in entries]
    count = sum(int(item[1]) for item in entries)
    valid = (
        len(names) == 288 and count == 11796480
        and all("lora_" in name and not any(x in name for x in ("visual", "merger", "projector", "connector")) for name in names)
    )
    return {"trainable_tensors": len(names), "trainable_parameters": count, "trainable_scope_ok": valid}


def parse_adapter(path: Path) -> dict:
    with path.open("rb") as stream:
        header_size_raw = stream.read(8)
        if len(header_size_raw) != 8:
            raise ValueError(f"Invalid safetensors header: {path}")
        header_size = struct.unpack("<Q", header_size_raw)[0]
        if header_size > 16 * 1024 * 1024:
            raise ValueError(f"Unreasonable safetensors header: {path}")
        header = json.loads(stream.read(header_size))
    tensors = {name: meta for name, meta in header.items() if name != "__metadata__"}
    count = len(tensors)
    parameters = sum(math.prod(meta["shape"]) for meta in tensors.values())
    valid = count == 288 and parameters == 11796480 and all(ADAPTER_NAME.fullmatch(name) for name in tensors)
    return {
        "adapter_bytes": path.stat().st_size,
        "adapter_sha256": sha256(path),
        "adapter_tensors": count,
        "adapter_parameters": parameters,
        "adapter_scope_ok": valid,
    }


def inspect_run(run_dir: Path, method: str, cap: int, train_ids: set[str]) -> dict:
    row = {"run": run_dir.name, "method": method, "cap_tokens": cap,
           "run_dir": str(run_dir), "rollouts_per_question": 5, "validation_scope": "five heldout questions; pipeline smoke only"}
    files = {
        "console": run_dir / "console.log",
        "visits": run_dir / "visits.jsonl",
        "trainable": run_dir / "trainable_params.json",
        "adapter": run_dir / "checkpoints/global_step_1/actor/lora_adapter/adapter_model.safetensors",
    }
    missing = [name for name, path in files.items() if not path.is_file()]
    row["missing_artifacts"] = missing
    if files["console"].is_file():
        row.update(parse_console(files["console"]))
    if files["visits"].is_file():
        row.update(parse_visits(files["visits"], train_ids))
    if files["trainable"].is_file():
        row.update(parse_trainable(files["trainable"]))
    if files["adapter"].is_file():
        row.update(parse_adapter(files["adapter"]))
    reload_report = run_dir / "adapter_reload.json"
    if reload_report.is_file():
        reload_data = json.loads(reload_report.read_text(encoding="utf-8"))
        row["adapter_reload_ok"] = (
            reload_data.get("peft_reload") == "exact"
            and reload_data.get("adapter_sha256") == row.get("adapter_sha256")
            and reload_data.get("tensor_count") == 288
            and reload_data.get("parameter_count") == 11796480
        )
    else:
        row["adapter_reload_ok"] = None
    accuracy = row.get("train_raw_accuracy_attempted")
    gradient = row.get("grad_norm")
    row["learning_signal_ok"] = (
        accuracy is not None and 0 < accuracy < 1
        and gradient is not None and gradient > 0
    )
    row["status"] = (
        "observed_complete" if not missing and row.get("training_step_observed")
        and row.get("final_validation_observed") and row.get("trainable_scope_ok")
        and row.get("adapter_scope_ok") and row["learning_signal_ok"]
        and row["adapter_reload_ok"] is not False else "partial_or_invalid"
    )
    return row


def fmt(value: object, digits: int = 2) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def render_md(rows: list[dict], train_ids: set[str]) -> str:
    out = [
        "# Stage 1 GRPO/DAPO core smoke summary",
        "",
        f"Training subset: {len(train_ids)} unique ECD questions. Each row is one optimizer-step smoke on five selected questions, with five fresh rollouts per question.",
        "Validation used five heldout questions solely to test the pipeline. These accuracies are not benchmark results.",
        "DAPO core used asymmetric clipping, token-level loss, and raw v1.2 accuracy filtering. Soft overlong punishment was disabled, so this is not full DAPO.",
        "Adapter header, tensor count, parameter count, and scope checks are static. Actual PEFT reload is checked separately and recorded when `adapter_reload.json` is available.",
        "",
        "| Method | Cap | Attempted / used questions | Raw accuracy (attempted) | Selected reward mean | Answer present | At cap | Mean tokens | Grad norm | Signal | Step s | Holdout5 | Adapter MiB | PEFT reload | SHA256 | Status |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- | --- | --- |",
    ]
    for row in rows:
        method = "GRPO" if row["method"] == "grpo" else "DAPO core"
        attempted = fmt(row.get("attempted_prompt_visits"), 0)
        used = fmt(row.get("used_prompt_visits"), 0)
        adapter_mib = row.get("adapter_bytes")
        adapter_mib = adapter_mib / 1024**2 if adapter_mib is not None else None
        sha = row.get("adapter_sha256")
        out.append("| " + " | ".join([
            method, str(row["cap_tokens"]), f"{attempted} / {used}",
            fmt(row.get("train_raw_accuracy_attempted"), 3),
            fmt(row.get("train_selected_reward_mean"), 3),
            fmt(row.get("train_answer_present_attempted"), 3),
            fmt(row.get("train_at_cap_attempted"), 3),
            fmt(row.get("train_response_mean_tokens"), 1),
            fmt(row.get("grad_norm"), 3), "yes" if row["learning_signal_ok"] else "no",
            fmt(row.get("step_seconds"), 1),
            fmt(row.get("validation5_accuracy"), 3), fmt(adapter_mib, 1),
            "yes" if row.get("adapter_reload_ok") is True else ("no" if row.get("adapter_reload_ok") is False else "—"),
            sha[:12] if sha else "—", row["status"],
        ]) + " |")
    out += ["", "Attempted visits are logged before generation. For completed runs, attempted visits × 5 gives generated responses. DAPO may attempt more questions than it uses for an optimizer step.", ""]
    return "\n".join(out)


def self_test() -> None:
    example = (
        "\x1b[36m(Runner pid=1)\x1b[0m Step 1\n"
        "\x1b[36m(Runner pid=1)\x1b[0m actor:\n"
        "\x1b[36m(Runner pid=1)\x1b[0m   grad_norm: 0.079\n"
        "\x1b[36m(Runner pid=1)\x1b[0m critic:\n"
        "\x1b[36m(Runner pid=1)\x1b[0m   rewards:\n"
        "\x1b[36m(Runner pid=1)\x1b[0m     mean: 0.56\n"
        "\x1b[36m(Runner pid=1)\x1b[0m perf:\n"
        "\x1b[36m(Runner pid=1)\x1b[0m   time_per_step: 31.991\n"
        "\x1b[36m(Runner pid=1)\x1b[0m response_length:\n"
        "\x1b[36m(Runner pid=1)\x1b[0m   mean: 300.64\n"
        "\x1b[36m(Runner pid=1)\x1b[0m reward:\n"
        "\x1b[36m(Runner pid=1)\x1b[0m   accuracy: 0.56\n"
        "\x1b[36m(Runner pid=1)\x1b[0m   answer_present: 0.84\n"
        "\x1b[36m(Runner pid=1)\x1b[0m   at_cap_proxy: 0.16\n"
        "\x1b[36m(Runner pid=1)\x1b[0m Start validation...\n"
        "\x1b[36m(Runner pid=1)\x1b[0m Step 1\n"
        "\x1b[36m(Runner pid=1)\x1b[0m val:\n"
        "\x1b[36m(Runner pid=1)\x1b[0m   accuracy_reward: 0.6\n"
        "\x1b[36m(Runner pid=1)\x1b[0m Final validation metrics:\n"
    )
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        log = root / "console.log"
        log.write_text(example, encoding="utf-8")
        parsed = parse_console(log)
        assert parsed["train_raw_accuracy_attempted"] == 0.56
        assert parsed["train_selected_reward_mean"] == 0.56
        assert parsed["train_response_mean_tokens"] == 300.64
        assert parsed["grad_norm"] == 0.079
        assert parsed["validation5_accuracy"] == 0.6
        visits = root / "visits.jsonl"
        visits.write_text('\n'.join(json.dumps(event) for event in (
            {"event": "attempt", "sample_id": "one", "step": 1},
            {"event": "attempt", "sample_id": "two", "step": 1},
            {"event": "used", "sample_id": "one", "step": 1},
        )) + '\n', encoding="utf-8")
        checked = parse_visits(visits, {"one", "two"})
        assert checked["attempted_prompt_visits"] == 2
        assert checked["used_prompt_visits"] == 1
        adapter = root / "adapter_model.safetensors"
        header = json.dumps({"small.weight": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]}}).encode()
        adapter.write_bytes(struct.pack("<Q", len(header)) + header + bytes(8))
        adapter_report = parse_adapter(adapter)
        assert adapter_report["adapter_tensors"] == 1
        assert adapter_report["adapter_parameters"] == 2
    print("self-test passed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path)
    parser.add_argument("--train-jsonl", type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-md", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if not all((args.runs_root, args.train_jsonl, args.output_json, args.output_md)):
        parser.error("--runs-root, --train-jsonl, --output-json, and --output-md are required")
    train_ids = load_train_ids(args.train_jsonl)
    rows = [inspect_run(args.runs_root / f"ecd20_signal_{method}_cap{cap}_20261120", method, cap, train_ids)
            for method, cap in RUNS]
    result = {
        "training_subset_unique_ids": len(train_ids),
        "train_jsonl": str(args.train_jsonl),
        "train_jsonl_sha256": sha256(args.train_jsonl),
        "validation_scope": "five heldout questions; smoke only, not performance evaluation",
        "dapo_algorithm": "asymmetric clip + token-level loss + raw-accuracy dynamic sampling; no overlong penalty",
        "runs": rows,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.output_md.write_text(render_md(rows, train_ids), encoding="utf-8")
    print(json.dumps({"runs": len(rows), "observed_complete": sum(row["status"] == "observed_complete" for row in rows),
                      "output_json": str(args.output_json), "output_md": str(args.output_md)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
