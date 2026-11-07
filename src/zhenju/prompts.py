"""Instruction template and parser shared by every Qwen arm, so only the weights differ."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .data import ERROR_TYPES, TYPE_CODES, ZH_TO_CODE

SYSTEM_PROMPT = "你是一名中文语言学与语文教学专家，擅长诊断句子的语法与逻辑问题。"

_TYPE_LIST = "、".join(ERROR_TYPES.values())

USER_TEMPLATE = (
    "请诊断下面的句子是否为病句，并严格按四行格式回答：\n"
    "判断：病句 或 正确\n"
    f"类型：从{_TYPE_LIST}中选择，可多选并用分号分隔；句子正确时填“无”\n"
    "修改：简要说明修改位置与方法；句子正确时填“无”\n"
    "改正：写出改正后的完整句子；句子正确时照抄原句\n\n"
    "句子：{sentence}"
)


def format_target(flag: int, types: list[str], note: str, correction: str) -> str:
    """Canonical four-line answer the instruction-tuned model is trained to produce."""
    if flag == 0:
        return f"判断：正确\n类型：无\n修改：无\n改正：{correction}"
    names = ";".join(ERROR_TYPES[c] for c in types)
    return f"判断：病句\n类型：{names}\n修改：{note}\n改正：{correction}"


def build_messages(sentence: str, shots: list[dict] | None = None) -> list[dict]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for shot in shots or []:
        messages.append({"role": "user", "content": USER_TEMPLATE.format(sentence=shot["sentence"])})
        messages.append({"role": "assistant", "content": format_target(
            shot["error_flag"], shot["types"], shot["edit_note"], shot["references"][0])})
    messages.append({"role": "user", "content": USER_TEMPLATE.format(sentence=sentence)})
    return messages


_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)
_FIELD = re.compile(r"^\s*[*#\s]*(判断|类型|修改|改正)[*\s]*[：:]\s*(.*)$", re.MULTILINE)


@dataclass
class Diagnosis:
    flag: int
    types: list[str]
    note: str
    correction: str
    well_formed: bool


def parse_response(text: str, source: str) -> Diagnosis:
    """Parse a reply leniently, so a chatty pretrained model is not punished for layout alone.

    Unparseable fields fall back to the source sentence and no type; ``well_formed``
    records that so the format-failure rate can be reported separately.
    """
    body = _THINK.sub("", text).strip()
    if "</think>" in body:  # truncated or unmatched thinking block
        body = body.split("</think>")[-1].strip()
    fields: dict[str, str] = {}
    for match in _FIELD.finditer(body):
        fields.setdefault(match.group(1), match.group(2).strip())

    verdict = fields.get("判断", "")
    if verdict:
        if "病句" in verdict or "错误" in verdict or "有误" in verdict:
            flag = 1
        elif "正确" in verdict or "无" in verdict:
            flag = 0
        else:
            flag = 0
    else:
        head = body[:120]
        flag = int(any(word in head for word in ("病句", "有误", "错误", "不当")))

    raw_types = fields.get("类型", "")
    types = [code for name, code in ZH_TO_CODE.items() if name in raw_types]
    types.sort(key=TYPE_CODES.index)

    correction = fields.get("改正", "").strip().strip("“”\"'")
    well_formed = all(key in fields for key in ("判断", "类型", "修改", "改正")) and bool(correction)
    if not correction:
        correction = source
    if flag == 0:
        types = []
    return Diagnosis(flag, types, fields.get("修改", ""), correction, well_formed)
