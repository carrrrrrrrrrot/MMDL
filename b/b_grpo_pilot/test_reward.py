"""Dependency-free checks for the proposed strict multiple-choice reward."""

from reward import parse_final_mc_answer, strict_mc_reward


CASES = [
    ("R01", "B", "Answer: B", "B", 1.0),
    ("R02", "B", "Answer: C", "C", 0.0),
    ("R03", "B", "B may work.\nAnswer: C", "C", 0.0),
    ("R04", "B", "B is the answer.", None, 0.0),
    ("R05", "B", "Answer: C\nAnswer: B", "B", 1.0),
    ("R06", "B", "Answer: B\nAnswer:", None, 0.0),
    ("R07", "B", "Answer: B or C", None, 0.0),
    ("R07-C", "C", "Answer: B or C", None, 0.0),
    ("R08", "B", "Answer: E", None, 0.0),
    ("R09", "B", "Answer: b", None, 0.0),
    ("truncated", "B", "Answer: B\nAnswer: ", None, 0.0),
]


def main() -> None:
    for case_id, gold, output, parsed, expected in CASES:
        actual_parsed = parse_final_mc_answer(output)
        actual_reward = strict_mc_reward([[{"role": "assistant", "content": output}]], [gold])[0]
        assert actual_parsed == parsed, (case_id, actual_parsed, parsed)
        assert actual_reward == expected, (case_id, actual_reward, expected)
    try:
        strict_mc_reward([[{"role": "assistant", "content": "no answer"}]], [None])
    except ValueError:
        pass
    else:
        raise AssertionError("Missing gold answer must fail validation")
    print(f"PASS: {len(CASES)} strict multiple-choice cases")


if __name__ == "__main__":
    main()
