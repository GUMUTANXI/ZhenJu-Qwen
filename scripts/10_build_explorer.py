"""Build docs/index.html: a static browser over every test sentence and each model's real output."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.stdout.reconfigure(encoding="utf-8")

from rapidfuzz.distance import Levenshtein  # noqa: E402

from zhenju.data import ERROR_TYPES, read_jsonl  # noqa: E402

ARM_LABELS = {
    "pretrained_zero": "预训练（零样本）",
    "sft_s42": "LoRA 指令微调",
    "dpo": "SFT + DPO",
}


def segments(source: str, target: str) -> list:
    """Compact diff: a plain string for equal text, [deleted, inserted] for a change."""
    out: list = []
    for op in Levenshtein.opcodes(source, target):
        old, new = source[op.src_start:op.src_end], target[op.dest_start:op.dest_end]
        if op.tag == "equal":
            out.append(old)
        else:
            out.append([old, new])
    return out


def compact_prediction(sample: dict, prediction: dict) -> dict:
    correction = prediction["correction"]
    ok_flag = prediction["flag"] == sample["error_flag"]
    ok_type = (not sample["error_flag"]) or set(prediction["types"]) == set(sample["types"])
    return {
        "f": prediction["flag"],
        "t": [ERROR_TYPES[c] for c in prediction["types"]],
        "n": prediction.get("note", ""),
        "d": segments(sample["sentence"], correction),
        "v": int(ok_flag),
        "y": int(ok_type),
    }


PAGE = r"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>诊句结果浏览</title>
<style>
:root{--bg:#f6f7f9;--card:#fff;--ink:#17202b;--muted:#5d6876;--line:#dfe3e8;--accent:#1c5cab;--chip:#e9f0f9;
--good:#14704f;--bad:#b4372c;--del:#fbe4e1;--ins:#dff3ea}
@media (prefers-color-scheme:dark){:root{--bg:#14171b;--card:#1d2127;--ink:#eceef1;--muted:#9aa4b1;--line:#30363e;
--accent:#7fb2ee;--chip:#22303f;--good:#5fcf9f;--bad:#ee8a80;--del:#47231f;--ins:#163a2c}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.7 "PingFang SC","Microsoft YaHei","Noto Sans SC",sans-serif}
main{max-width:1120px;margin:0 auto;padding:32px 16px 72px}
h1{font-size:26px;margin:0;letter-spacing:.02em}
.lead{color:var(--muted);margin:6px 0 22px;max-width:760px}
.table-wrap{overflow-x:auto;background:var(--card);border:1px solid var(--line);border-radius:10px;margin-bottom:22px}
table{border-collapse:collapse;width:100%;font-size:13.5px;font-variant-numeric:tabular-nums}
th,td{padding:8px 14px;text-align:right;border-bottom:1px solid var(--line);white-space:nowrap}
th:first-child,td:first-child{text-align:left}
tr:last-child td{border-bottom:0}
th{color:var(--muted);font-weight:500}
.hero td{font-weight:700;color:var(--accent)}
.filters{display:flex;flex-wrap:wrap;gap:8px 18px;align-items:center;margin-bottom:14px}
.filters label{font-size:13px;color:var(--muted);display:flex;gap:6px;align-items:center}
select,input{font:inherit;font-size:14px;color:var(--ink);background:var(--card);border:1px solid var(--line);border-radius:7px;padding:6px 9px;min-height:36px}
input{min-width:220px}
.count{font-size:13px;color:var(--muted);margin-left:auto}
.item{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px 18px;margin-bottom:12px}
.src{font-size:16px;margin-bottom:6px}
.gold{font-size:13px;color:var(--muted);margin-bottom:12px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:12px}
.arm{border-top:2px solid var(--line);padding-top:8px;min-width:0}
.arm.hero-arm{border-top-color:var(--accent)}
.arm h3{margin:0 0 4px;font-size:12.5px;font-weight:500;color:var(--muted)}
.verdict{font-weight:700;font-size:14px}
.ok{color:var(--good)}.no{color:var(--bad)}
.chip{display:inline-block;background:var(--chip);color:var(--accent);border-radius:99px;padding:0 9px;font-size:12.5px;margin:2px 4px 2px 0}
.note{font-size:13px;color:var(--muted);margin:3px 0}
.fix{font-size:14.5px;overflow-wrap:anywhere}
del{background:var(--del);color:var(--bad);border-radius:3px;padding:0 1px}
ins{background:var(--ins);color:var(--good);text-decoration:none;border-radius:3px;padding:0 1px;font-weight:600}
button{font:inherit;color:#fff;background:var(--accent);border:0;border-radius:8px;padding:9px 18px;cursor:pointer;display:block;margin:18px auto 0;min-height:40px}
@media (prefers-color-scheme:dark){button{color:#10151b}}
</style>
</head>
<body>
<main>
<h1>诊句结果浏览</h1>
<p class="lead">FCGEC 留出测试集共 __N__ 句，下面是各模型未经挑选的真实输出。结果判断按语料标注计算，红色为删除，绿色为新增。</p>
<div class="table-wrap"><table id="metrics"></table></div>
<div class="filters">
<label>错误类型 <select id="type"><option value="">全部</option></select></label>
<label>查看 <select id="view">
<option value="all">全部句子</option>
<option value="win">微调判对、预训练判错</option>
<option value="lose">微调判错</option>
<option value="fixed">微调改正完全正确</option>
</select></label>
<label>搜索 <input id="q" type="search" placeholder="输入句中词语"></label>
<span class="count" id="count"></span>
</div>
<div id="list"></div>
<button id="more" type="button">再显示 40 句</button>
</main>
<script>
const DATA=__DATA__;
const METRICS=__METRICS__;
const ARMS=__ARMS__;
const $=id=>document.getElementById(id);
const esc=s=>String(s).replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const fmt=v=>v===null||v===undefined?"—":Number(v).toFixed(3);
(function(){
  const head="<tr><th>模型</th><th>检测 Acc</th><th>精确率</th><th>召回率</th><th>类型 Macro-F1</th><th>改正 F0.5</th><th>病句完全改对</th></tr>";
  $("metrics").innerHTML=head+METRICS.map(m=>`<tr class="${m.hero?"hero":""}"><td>${esc(m.name)}</td><td>${fmt(m.detection)}</td><td>${fmt(m.precision)}</td><td>${fmt(m.recall)}</td><td>${fmt(m.type)}</td><td>${fmt(m.edit)}</td><td>${fmt(m.exact)}</td></tr>`).join("");
  const types=[...new Set(DATA.flatMap(d=>d.t))];
  $("type").innerHTML+=types.map(t=>`<option>${esc(t)}</option>`).join("");
})();
const diff=parts=>parts.map(p=>typeof p==="string"?esc(p):(p[0]?`<del>${esc(p[0])}</del>`:"")+(p[1]?`<ins>${esc(p[1])}</ins>`:"")).join("");
function armHtml(key,a,flag){
  const label=ARMS[key];
  const right=a.v?'<span class="ok">判断正确</span>':'<span class="no">判断错误</span>';
  const chips=a.t.map(t=>`<span class="chip">${esc(t)}</span>`).join("");
  const note=a.f&&a.n&&a.n!=="无"?`<div class="note">${esc(a.n)}</div>`:"";
  return `<div class="arm ${key==="sft_s42"?"hero-arm":""}"><h3>${esc(label)}</h3><div class="verdict">${a.f?"病句":"正确"} · ${right}</div>${chips}${note}<div class="fix">${diff(a.d)}</div></div>`;
}
function itemHtml(d){
  const gold=d.f?`标注：病句（${d.t.map(esc).join("；")}）。参考改正：${esc(d.r)}`:"标注：正确句";
  return `<article class="item"><div class="src">${esc(d.s)}</div><div class="gold">${gold}</div><div class="grid">${Object.keys(ARMS).filter(k=>d.p[k]).map(k=>armHtml(k,d.p[k],d.f)).join("")}</div></article>`;
}
let shown=0,view=[];
function apply(){
  const t=$("type").value,v=$("view").value,q=$("q").value.trim();
  view=DATA.filter(d=>{
    if(t&&!d.t.includes(t))return false;
    if(q&&!d.s.includes(q))return false;
    const a=d.p.sft_s42,b=d.p.pretrained_zero;
    if(v==="win")return a&&b&&a.v&&!b.v;
    if(v==="lose")return a&&!a.v;
    if(v==="fixed")return a&&a.x;
    return true;
  });
  shown=0;$("list").innerHTML="";$("count").textContent=`共 ${view.length} 句`;more();
}
function more(){
  const next=view.slice(shown,shown+40);
  $("list").insertAdjacentHTML("beforeend",next.map(itemHtml).join(""));
  shown+=next.length;$("more").style.display=shown<view.length?"block":"none";
}
["type","view"].forEach(i=>$(i).addEventListener("change",apply));
$("q").addEventListener("input",apply);$("more").addEventListener("click",more);apply();
</script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--results", default=ROOT / "results", type=Path)
    parser.add_argument("--dpo-arm", default=None, help="selected DPO arm, chosen on the dev set")
    parser.add_argument("--out", default=ROOT / "docs" / "index.html", type=Path)
    args = parser.parse_args()

    samples = read_jsonl(args.data / "test.jsonl")
    predictions = {}
    for key, arm in {"pretrained_zero": "pretrained_zero", "sft_s42": "sft_s42", "dpo": args.dpo_arm}.items():
        path = args.results / "predictions" / f"{arm}.test.jsonl"
        if arm and path.exists():
            predictions[key] = read_jsonl(path)

    from zhenju.metrics import is_exact

    rows = []
    for i, sample in enumerate(samples):
        entry = {"s": sample["sentence"], "f": sample["error_flag"], "t": [ERROR_TYPES[c] for c in sample["types"]],
                 "r": sample["references"][0], "p": {}}
        for key, preds in predictions.items():
            compact = compact_prediction(sample, preds[i])
            compact["x"] = int(is_exact(preds[i]["correction"], sample["references"]) and sample["error_flag"] == 1
                               and preds[i]["flag"] == 1)
            entry["p"][key] = compact
        rows.append(entry)

    with open(args.results / "summary.csv", encoding="utf-8") as fh:
        summary = {r["arm"]: r for r in csv.DictReader(fh)}

    def metric_row(arm: str, name: str, hero: bool = False) -> dict | None:
        row = summary.get(arm)
        if not row:
            return None

        def num(key: str):
            return float(row[key]) if row.get(key) else None

        return {"name": name, "hero": hero, "detection": num("detection"), "precision": num("precision"),
                "recall": num("recall"), "type": num("type_macro_f1"), "edit": num("edit_f05"), "exact": num("exact_faulty")}

    table = [metric_row("pretrained_zero", "预训练 · 零样本"), metric_row("pretrained_few", "预训练 · 少样本"),
             metric_row("bert_full_s42", "BERT 全量微调"), metric_row("sft_s42", "LoRA 指令微调", True)]
    if args.dpo_arm:
        table.append(metric_row(args.dpo_arm, "SFT + DPO"))
    table = [t for t in table if t]

    arms = {k: ARM_LABELS[k] for k in ("pretrained_zero", "sft_s42", "dpo") if k in predictions}
    page = (PAGE.replace("__DATA__", json.dumps(rows, ensure_ascii=False, separators=(",", ":")))
            .replace("__METRICS__", json.dumps(table, ensure_ascii=False))
            .replace("__ARMS__", json.dumps(arms, ensure_ascii=False))
            .replace("__N__", f"{len(rows):,}"))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(page, encoding="utf-8")
    print("wrote", args.out.name, f"{args.out.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
