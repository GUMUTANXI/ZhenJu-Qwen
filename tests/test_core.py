"""Unit tests for edit replay, response parsing and scoring. Run: python -I tests/test_core.py"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.stdout.reconfigure(encoding="utf-8")

from zhenju.data import apply_reference, describe_edit, references_from_operations, strip_punct  # noqa: E402
from zhenju.metrics import best_reference_counts, edit_spans, is_exact, mcnemar_exact  # noqa: E402
from zhenju.prompts import format_target, parse_response  # noqa: E402


def test_readme_examples():
    # The four worked examples from the FCGEC data README.
    assert apply_reference("ABCDE", {"Switch": [0, 2, 1, 3, 4]}) == ["ACBDE"]
    assert apply_reference("ABCDE", {"Delete": [3]}) == ["ABCE"]
    assert apply_reference("ABCDE", {"Insert": [{"pos": 1, "tag": "INS_1", "label": ["F"]}]}) == ["ABFCDE"]
    assert apply_reference("ABCDE", {"Modify": [{"pos": 2, "tag": "MOD_1", "label": ["F"]}]}) == ["ABFDE"]


def test_span_alternatives_and_prepend():
    op = {"Modify": [{"pos": 1, "tag": "MOD_2", "label": ["X", "Y"]}]}
    assert apply_reference("ABCDE", op) == ["AXDE", "AYDE"]
    assert apply_reference("AB", {"Insert": [{"pos": -1, "tag": "INS_1", "label": "Z"}]}) == ["ZAB"]


def test_mixed_ops_keep_every_edit():
    # The official converter's unpack step drops Delete here; both edits must survive.
    op = {"Delete": [0], "Modify": [{"pos": 2, "tag": "MOD_1", "label": ["X"]}]}
    assert apply_reference("ABCD", op) == ["BXD"]


def test_positions_follow_switch():
    op = {"Switch": [1, 0, 2], "Delete": [0]}
    assert apply_reference("ABC", op) == ["AC"]


def test_references_are_independent():
    ops = [{"Switch": [1, 0, 2]}, {"Delete": [0]}]
    assert references_from_operations("ABC", ops) == ["BAC", "BC"]


def test_describe_edit():
    assert describe_edit("他很高兴", "他很开心") == "把“高兴”改为“开心”"
    assert describe_edit("他的的书", "他的书").startswith("删去")
    assert describe_edit("同句", "同句") == "无"


def test_parser_roundtrip():
    sentence = "他提高了水平。"
    text = format_target(1, ["IWC", "CM"], "把“a”改为“b”", "改好的句子。")
    parsed = parse_response(text, sentence)
    assert parsed.flag == 1 and parsed.types == ["IWC", "CM"]
    assert parsed.correction == "改好的句子。" and parsed.well_formed


def test_parser_tolerates_markdown_and_think():
    text = "<think>想一想</think>\n**判断**：正确\n**类型**：无\n**修改**：无\n**改正**：原句。"
    parsed = parse_response(text, "原句。")
    assert parsed.flag == 0 and parsed.types == [] and parsed.correction == "原句。"


def test_parser_fallback_marks_bad_format():
    parsed = parse_response("这句话有误，主语残缺。", "源句。")
    assert parsed.flag == 1 and not parsed.well_formed and parsed.correction == "源句。"


def test_metrics():
    assert strip_punct("你好，世界！") == "你好世界"
    assert edit_spans("abc", "abc") == set()
    assert best_reference_counts("abc", "axc", ["abc", "axc"]) == (1, 0, 0)
    assert best_reference_counts("abc", "abc", ["axc"]) == (0, 0, 1)
    assert best_reference_counts("abc", "ayc", ["axc"]) == (0, 1, 1)
    assert is_exact("a，b", ["ab"]) == 1
    result = mcnemar_exact([1, 1, 1, 0], [0, 0, 1, 0])
    assert result["only_a"] == 2 and result["only_b"] == 0


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
        print("ok", test.__name__)
    print(f"{len(tests)} tests passed")
