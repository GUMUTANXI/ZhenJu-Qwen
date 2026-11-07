"""ZhenJu demo: diagnose a Chinese sentence and compare the tuned model with the pretrained one.

Both columns run on one set of weights: the comparison simply disables the LoRA adapter, so
any difference comes from fine-tuning alone.

    python -I scripts/app.py --base models/Qwen3-1.7B --adapter adapters/zhenju-sft
"""

from __future__ import annotations

import argparse
import html
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.stdout.reconfigure(encoding="utf-8")

import gradio as gr  # noqa: E402
import torch  # noqa: E402
from rapidfuzz.distance import Levenshtein  # noqa: E402

from zhenju.data import ERROR_TYPES  # noqa: E402
from zhenju.prompts import parse_response  # noqa: E402
from zhenju.qwen import encode_prompt, generate_texts, load_policy, load_tokenizer  # noqa: E402

EXAMPLES = [
    "通过这次社会实践活动，使我们开阔了眼界，增长了见识。",
    "我们要不断提高学生的学习水平和良好的学习习惯。",
    "能否保持一颗平常心，是考试取得好成绩的关键。",
    "他那篇文章的观点，对我们很有启发。",
    "这家工厂的产品质量，近年来有了很大的提高，受到了消费者的广泛好评。",
    "校长、老师和同学们纷纷走出教室，欢迎到校视察的上级领导。",
]

TYPE_NOTES = {
    "语序不当": "词语或分句的先后次序违反了汉语的语法或逻辑顺序。",
    "搭配不当": "主谓、动宾或修饰语与中心语之间在语义上不能相配。",
    "成分残缺": "句子缺少必要的主语、谓语、宾语或其他成分。",
    "成分赘余": "句中出现了多余的词语，语义重复或累赘。",
    "结构混乱": "两种句式杂糅，或前后结构中途改变。",
    "不合逻辑": "表述违反事理，或前后概念、判断之间存在矛盾。",
    "歧义": "同一表达存在两种以上的合理理解。",
}

CSS = """
:root { --zj-ink:#17202b; --zj-muted:#5d6876; --zj-line:#dfe3e8; --zj-surface:#ffffff;
        --zj-canvas:#f5f6f8; --zj-accent:#1c5cab; --zj-ok:#14704f; --zj-bad:#b4372c; }
.gradio-container { background: var(--zj-canvas) !important; max-width: 1180px !important; }
.zj-head { padding: 22px 2px 10px; }
.zj-head h1 { font-size: 26px; margin: 0; color: var(--zj-ink); letter-spacing: .02em; font-weight: 700; }
.zj-head p { margin: 6px 0 0; color: var(--zj-muted); font-size: 14px; }
.zj-card { background: var(--zj-surface); border: 1px solid var(--zj-line); border-radius: 10px;
           padding: 18px 20px; min-height: 280px; }
.zj-tag { font-size: 12px; color: var(--zj-muted); letter-spacing: .08em; }
.zj-tuned { border-top: 3px solid var(--zj-accent); }
.zj-base { border-top: 3px solid #b4b8bf; }
.zj-verdict { font-size: 19px; font-weight: 700; margin: 8px 0 12px; }
.zj-verdict.bad { color: var(--zj-bad); }
.zj-verdict.ok { color: var(--zj-ok); }
.zj-chip { display: inline-block; padding: 2px 10px; margin: 0 6px 6px 0; border-radius: 999px; font-size: 13px;
           background: #e9f0f9; color: var(--zj-accent); }
.zj-label { font-size: 12px; color: var(--zj-muted); margin-top: 12px; }
.zj-text { font-size: 15.5px; line-height: 1.85; color: var(--zj-ink); }
.zj-del { background: #fbe4e1; color: var(--zj-bad); text-decoration: line-through; border-radius: 3px; padding: 0 1px; }
.zj-ins { background: #dff3ea; color: var(--zj-ok); border-radius: 3px; padding: 0 1px; font-weight: 600; }
.zj-hint { font-size: 13px; color: var(--zj-muted); line-height: 1.7; margin-top: 8px; }
.zj-warn { font-size: 13px; color: #8a6100; margin-top: 10px; }
"""


def diff_html(source: str, target: str) -> str:
    pieces: list[str] = []
    for op in Levenshtein.opcodes(source, target):
        old, new = html.escape(source[op.src_start:op.src_end]), html.escape(target[op.dest_start:op.dest_end])
        if op.tag == "equal":
            pieces.append(old)
            continue
        if old:
            pieces.append(f'<span class="zj-del">{old}</span>')
        if new:
            pieces.append(f'<span class="zj-ins">{new}</span>')
    return "".join(pieces)


def render_card(title: str, css_class: str, sentence: str, raw: str) -> str:
    parsed = parse_response(raw, sentence)
    verdict = "病句" if parsed.flag else "未发现语病"
    tone = "bad" if parsed.flag else "ok"
    names = [ERROR_TYPES[c] for c in parsed.types]
    chips = "".join(f'<span class="zj-chip">{n}</span>' for n in names)
    hint = "".join(f'<div class="zj-hint"><b>{n}</b>：{TYPE_NOTES[n]}</div>' for n in names)
    body = [f'<div class="zj-card {css_class}"><div class="zj-tag">{title}</div>',
            f'<div class="zj-verdict {tone}">{verdict}</div>']
    if chips:
        body.append(chips)
    if parsed.flag and parsed.note and parsed.note != "无":
        body.append(f'<div class="zj-label">修改说明</div><div class="zj-text">{html.escape(parsed.note)}</div>')
    if parsed.flag:
        body.append(f'<div class="zj-label">改正（红删绿增）</div>'
                    f'<div class="zj-text">{diff_html(sentence, parsed.correction)}</div>')
    body.append(hint)
    if not parsed.well_formed:
        body.append('<div class="zj-warn">该输出未严格遵守四行格式。</div>')
    body.append("</div>")
    return "".join(body)


def build_app(model, tokenizer) -> gr.Blocks:
    def generate(sentence: str) -> str:
        return generate_texts(model, tokenizer, [encode_prompt(tokenizer, sentence)], batch_size=1)[0]

    def diagnose(sentence: str):
        sentence = sentence.strip()
        if not sentence:
            empty = '<div class="zj-card"><div class="zj-hint">请输入一个中文句子。</div></div>'
            return empty, empty
        tuned_raw = generate(sentence)
        with model.disable_adapter(), torch.inference_mode():
            base_raw = generate(sentence)
        return (render_card("微调后 · Qwen3-1.7B + LoRA", "zj-tuned", sentence, tuned_raw),
                render_card("预训练 · Qwen3-1.7B", "zj-base", sentence, base_raw))

    with gr.Blocks(title="诊句 ZhenJu") as demo:
        gr.HTML('<div class="zj-head"><h1>诊句 ZhenJu</h1>'
                "<p>中文病句诊断：判断、归类、说明、改正。两栏使用同一份权重，差别只来自微调。</p></div>")
        box = gr.Textbox(label="待诊断的句子", lines=2, value=EXAMPLES[0])
        button = gr.Button("诊断", variant="primary")
        with gr.Row(equal_height=True):
            tuned = gr.HTML()
            base = gr.HTML()
        gr.Examples(EXAMPLES, inputs=box, label="示例")
        button.click(diagnose, inputs=box, outputs=[tuned, base])
        box.submit(diagnose, inputs=box, outputs=[tuned, base])
    return demo


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    tokenizer = load_tokenizer(args.base)
    model = load_policy(args.base, args.adapter)
    model.eval()
    build_app(model, tokenizer).launch(server_name="127.0.0.1", server_port=args.port, css=CSS,
                                       theme=gr.themes.Base(font=["Microsoft YaHei", "PingFang SC", "sans-serif"]))


if __name__ == "__main__":
    main()
