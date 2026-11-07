"""Render the report figures from results/. Each figure defends one claim:

fig1  Instruction tuning turns a general model into a usable diagnoser.
fig2  Fine-tuning reshapes the representation, and how much depends on the unfreezing strategy.
fig3  PPO and DPO each fail in their own way, and those failures are measurable.
fig4  End-to-end pipeline, annotated with this project's measured numbers.
short logo.png, hero banners, and the one-row scores and alignment figures for the short report.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.stdout.reconfigure(encoding="utf-8")

import matplotlib as mpl  # noqa: E402

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

from zhenju.data import ERROR_TYPES, TYPE_CODES, read_jsonl  # noqa: E402

INK = "#17202b"
MUTED = "#5d6876"
GRID = "#e6e8ec"
PRETRAINED = "#b4b8bf"
FEWSHOT = "#8a8f98"
BERT_C = "#3f4650"
SFT_C = "#2a78d6"
DPO_C = "#eb6834"
PPO_C = "#1baf7a"
FAULT_C = "#d64541"
CLEAN_C = "#9aa1ab"

ARMS = {
    "sft_s42": ("LoRA 指令微调", SFT_C),
    "bert_full_s42": ("BERT 全量微调", BERT_C),
    "pretrained_few": ("预训练 · 少样本", FEWSHOT),
    "pretrained_zero": ("预训练 · 零样本", PRETRAINED),
}

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Microsoft YaHei", "Arial", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": 7,
    "axes.labelsize": 7,
    "axes.titlesize": 7.5,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.6,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "legend.frameon": False,
    "lines.solid_capstyle": "round",
})


def load_csv(path: Path) -> dict[str, dict]:
    with open(path, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    key = "arm" if rows and "arm" in rows[0] else None
    out = {}
    for i, row in enumerate(rows):
        parsed = {}
        for k, v in row.items():
            try:
                parsed[k] = float(v)
            except (TypeError, ValueError):
                parsed[k] = v
        out[row[key] if key else i] = parsed
    return out


def load_log(path: Path) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = {}
    if not path.exists():
        return groups
    for row in read_jsonl(path):
        groups.setdefault(row["event"], []).append(row)
    return groups


def train_seconds_per_step(path: Path, key: str) -> float:
    """Wall-clock per training step, leaving out the gaps where dev evaluation ran."""
    total, covered, previous = 0.0, 0, None
    for row in read_jsonl(path):
        if row["event"] == "train":
            if previous is not None:
                total += row["elapsed"] - previous["elapsed"]
                covered += row[key] - previous[key]
            previous = row
        elif row["event"] == "dev":
            previous = None
    return total / covered


def has(row: dict, key: str) -> bool:
    return isinstance(row.get(key), float) and not np.isnan(row[key])


def panel_label(ax, letter: str, x: float = -0.02, y: float = 1.09) -> None:
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=9, fontweight="bold", color=INK, ha="right", va="bottom")


def soft_grid(ax, axis: str = "x") -> None:
    ax.grid(axis=axis, color=GRID, linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)


def smooth(values, window: int = 7) -> np.ndarray:
    """Centered rolling mean. An EMA seeded with the first value would drag an outlier across the curve."""
    values = np.asarray(values, dtype=float)
    half = window // 2
    return np.array([values[max(0, i - half):i + half + 1].mean() for i in range(len(values))])


def save(fig, out: Path, name: str, alignment: bool = True) -> None:
    out.mkdir(parents=True, exist_ok=True)
    stem = out / name
    if alignment:
        try:
            from audit_panel_alignment import require_matplotlib_panel_alignment
        except ImportError:
            print("alignment auditor not on sys.path; skipping alignment gate")
        else:
            require_matplotlib_panel_alignment(
                fig, json_out=f"{stem}.alignment.json", overlay_svg=None,
                tolerance_pt=1.5, gutter_tolerance_pt=1.5, strict=False)
    fig.savefig(f"{stem}.pdf", bbox_inches="tight", pad_inches=0.03)
    fig.savefig(f"{stem}.svg", bbox_inches="tight", pad_inches=0.03)
    fig.savefig(f"{stem}.png", dpi=400, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print("saved", name)


# ---------------------------------------------------------------- figure 1
def tier_panel(ax, summary, arms, metric, low, high, title, xlim, na_text=None) -> None:
    y = np.arange(len(arms))[::-1]
    for row_y, arm in zip(y, arms):
        row = summary.get(arm)
        label, color = ARMS[arm]
        if row is None or not has(row, metric):
            if na_text:
                ax.text(xlim[0] + 0.015 * (xlim[1] - xlim[0]), row_y, na_text, va="center", ha="left",
                        fontsize=6, color=MUTED)
            continue
        value = row[metric]
        ax.plot([row[low], row[high]], [row_y, row_y], color=color, linewidth=1.3, zorder=2, solid_capstyle="butt")
        ax.scatter([value], [row_y], s=26 if arm == "sft_s42" else 18, color=color, zorder=3,
                   edgecolor="white", linewidth=0.5)
        ax.text(row[high] + 0.018 * (xlim[1] - xlim[0]), row_y, f"{value:.3f}", va="center", ha="left",
                fontsize=6.3, color=INK, fontweight="bold" if arm == "sft_s42" else "normal")
    ax.set_xlim(*xlim)
    ax.set_ylim(-0.6, len(arms) - 0.4)
    ax.set_title(title, loc="left", color=INK, fontweight="bold", pad=4)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)


def figure1(results: Path, out: Path) -> None:
    summary = load_csv(results / "summary.csv")
    per_type = load_csv(results / "per_type_f1.csv")
    arms = [a for a in ARMS if a in summary]

    fig = plt.figure(figsize=(6.0, 4.0))
    grid = GridSpec(2, 3, figure=fig, height_ratios=[0.86, 1.1], hspace=0.5, wspace=0.42,
                    left=0.135, right=0.985, top=0.94, bottom=0.105)
    axes_top = [fig.add_subplot(grid[0, i]) for i in range(3)]
    specs = [
        ("detection", "detection_low", "detection_high", "病句检测 Accuracy", (0.45, 0.92), None),
        ("type_macro_f1", "type_low", "type_high", "错误类型 Macro-F1", (0.0, 0.75), None),
        ("edit_f05", "edit_low", "edit_high", "字符级改正 F0.5", (0.0, 0.62), "判别模型，不能生成改正句"),
    ]
    for ax, (metric, low, high, title, xlim, na) in zip(axes_top, specs):
        tier_panel(ax, summary, arms, metric, low, high, title, xlim, na)
    y = np.arange(len(arms))[::-1]
    axes_top[0].set_yticks(y, [ARMS[a][0] for a in arms], color=INK, fontsize=6.6)
    for ax in axes_top[1:]:
        ax.set_yticks(y, [""] * len(arms))
    panel_label(axes_top[0], "a", x=-0.62, y=1.05)

    # b: how the detector's behaviour changed
    ax = fig.add_subplot(grid[1, 0])
    prevalence = 1100 / 1998
    ax.axhline(prevalence, color=MUTED, linewidth=0.6, linestyle=(0, (3, 2)), zorder=1)
    ax.text(0.02, prevalence + 0.006, "全判病句的精确率", fontsize=5.8, color=MUTED, va="bottom", ha="left")
    zero, tuned = summary.get("pretrained_zero"), summary.get("sft_s42")
    if zero and tuned:
        ax.add_patch(FancyArrowPatch((zero["recall"], zero["precision"]), (tuned["recall"], tuned["precision"]),
                                     arrowstyle="-|>", mutation_scale=7, color=SFT_C, linewidth=0.9,
                                     shrinkA=4, shrinkB=4, connectionstyle="arc3,rad=-0.12", zorder=2))
    label_offset = {"sft_s42": (0.0, 0.035, "center"), "bert_full_s42": (-0.035, 0.0, "right"),
                    "pretrained_zero": (-0.03, -0.03, "right"), "pretrained_few": (-0.03, 0.018, "right")}
    for arm in arms:
        row = summary[arm]
        if not has(row, "precision"):
            continue
        ax.scatter(row["recall"], row["precision"], s=26 if arm == "sft_s42" else 18, color=ARMS[arm][1],
                   zorder=3, edgecolor="white", linewidth=0.5)
        dx, dy, ha = label_offset[arm]
        ax.text(row["recall"] + dx, row["precision"] + dy, ARMS[arm][0].replace("预训练 · ", ""), fontsize=5.8,
                color=INK, ha=ha, va="center")
    ax.set_xlim(0, 1.04)
    ax.set_ylim(0.5, 0.95)
    ax.set_xlabel("召回率（找出多少病句）")
    ax.set_ylabel("精确率（报警有多准）")
    panel_label(ax, "b", x=-0.24, y=1.03)

    # c: per-type F1
    ax = fig.add_subplot(grid[1, 1])
    heat_arms = [a for a in ("sft_s42", "bert_full_s42", "pretrained_few", "pretrained_zero") if a in per_type]
    matrix = np.array([[per_type[a][c] for c in TYPE_CODES] for a in heat_arms])
    cmap = mpl.colors.LinearSegmentedColormap.from_list("zj", ["#f7f9fc", "#b7d3f6", "#3987e5", "#104281"])
    ax.imshow(matrix, cmap=cmap, vmin=0, vmax=0.9, aspect="auto")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:.2f}"[1:] if matrix[i, j] < 1 else "1", ha="center", va="center", fontsize=5.5,
                    color="white" if matrix[i, j] > 0.5 else INK)
    ax.set_xticks(range(len(TYPE_CODES)),
                  [ERROR_TYPES[c].replace("不当", "\n不当").replace("成分", "成分\n").replace("结构", "结构\n").replace("不合", "不合\n")
                   for c in TYPE_CODES], fontsize=5.3, linespacing=0.95)
    names = {"sft_s42": "LoRA", "bert_full_s42": "BERT", "pretrained_few": "少样本", "pretrained_zero": "零样本"}
    ax.set_yticks(range(len(heat_arms)), [names[a] for a in heat_arms], fontsize=6.2, color=INK)
    ax.tick_params(length=0)
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.set_xlabel("分类型 F1")
    panel_label(ax, "c", x=-0.15, y=1.03)

    # d: data efficiency
    ax = fig.add_subplot(grid[1, 2])
    points = []
    for frac, name in [(0.01, "sft_frac0.01_s42"), (0.05, "sft_frac0.05_s42"), (0.2, "sft_frac0.20_s42"),
                       (0.5, "sft_frac0.50_s42"), (1.0, "sft_s42")]:
        if name in summary:
            log = load_log(results / "logs" / f"{name}.jsonl")
            n = log["summary"][0]["n_train"] if log.get("summary") else int(25603 * frac)
            points.append((n, summary[name]))
    series = [("detection", "检测", "o"), ("type_macro_f1", "类型", "s"), ("edit_f05", "改正", "^")]
    anchor = 90
    for metric, label, marker in series:
        if not points:
            break
        xs = [p[0] for p in points]
        ys = [p[1][metric] for p in points]
        if zero:
            ax.plot([anchor, xs[0]], [zero[metric], ys[0]], color=SFT_C, linewidth=0.8, linestyle=(0, (2, 2)), zorder=2)
            ax.scatter([anchor], [zero[metric]], s=13, facecolor="white", edgecolor=SFT_C, linewidth=0.8,
                       marker=marker, zorder=3)
        ax.plot(xs, ys, color=SFT_C, linewidth=1.1, zorder=2)
        ax.scatter(xs, ys, s=13, color=SFT_C, marker=marker, zorder=3, edgecolor="white", linewidth=0.4)
        ax.text(xs[-1] * 1.25, ys[-1], label, fontsize=6.2, color=INK, va="center", ha="left")
    ax.set_xscale("log")
    ax.set_xlim(60, 90000)
    ax.set_ylim(0, 0.92)
    ax.set_xticks([90, 1e3, 1e4], ["0", "$10^3$", "$10^4$"])
    ax.xaxis.set_minor_locator(mpl.ticker.NullLocator())
    ax.set_xlabel("训练句子数（对数轴，0 为预训练）")
    ax.set_ylabel("测试集得分")
    soft_grid(ax, "y")
    panel_label(ax, "d", x=-0.24, y=1.03)

    save(fig, out, "fig1_instruction_tuning")


# ---------------------------------------------------------------- figure 2
def figure2(results: Path, embeddings: Path, data: Path, out: Path) -> None:
    from sklearn.metrics import silhouette_score

    try:
        import umap
    except ImportError:
        umap = None

    samples = read_jsonl(data / "test.jsonl")
    labels = np.array([s["error_flag"] for s in samples])
    before = np.load(embeddings / "bert_full_detect_s42_before.npy")
    after = np.load(embeddings / "bert_full_detect_s42_after.npy")
    summary = load_csv(results / "summary.csv")

    fig = plt.figure(figsize=(6.0, 2.15))
    outer = GridSpec(1, 2, figure=fig, wspace=0.14, left=0.02, right=0.985, top=0.88, bottom=0.2,
                     width_ratios=[1.45, 2.0])
    scatter_grid = outer[0, 0].subgridspec(1, 2, wspace=0.08)
    right_grid = outer[0, 1].subgridspec(1, 2, width_ratios=[1.0, 1.0], wspace=0.75)
    scatter_axes = [fig.add_subplot(scatter_grid[0, 0]), fig.add_subplot(scatter_grid[0, 1])]
    report = {}
    for index, (vectors, title) in enumerate([(before, "微调前"), (after, "微调后")]):
        ax = scatter_axes[index]
        sil = float(silhouette_score(vectors, labels, metric="cosine"))
        report[title] = sil
        if umap is not None:
            reducer = umap.UMAP(n_neighbors=30, min_dist=0.25, metric="cosine", random_state=7, n_jobs=1)
            xy = reducer.fit_transform(vectors)
        else:
            from sklearn.decomposition import PCA
            xy = PCA(n_components=2, random_state=7).fit_transform(vectors)
        for flag, color, name, size in [(0, CLEAN_C, "正确句", 3.2), (1, FAULT_C, "病句", 3.2)]:
            pick = labels == flag
            ax.scatter(xy[pick, 0], xy[pick, 1], s=size, c=color, linewidths=0, alpha=0.75, rasterized=True, label=name)
        ax.set_xticks([])
        ax.set_yticks([])
        for side in ("left", "bottom"):
            ax.spines[side].set_visible(False)
        ax.set_title(title, loc="left", color=INK, fontweight="bold", pad=3, x=0.08 if index == 0 else 0.0)
        ax.text(0.0, -0.05, f"轮廓系数 {sil:.3f}", transform=ax.transAxes, fontsize=6.2, color=INK, va="top")
        if index == 0:
            ax.legend(loc="upper left", bbox_to_anchor=(-0.04, -0.13), ncol=2, handletextpad=0.1, columnspacing=0.6,
                      markerscale=2.6, fontsize=6, borderpad=0)
            panel_label(ax, "a", x=0.05, y=1.0)

    # b: minimal pairs. Faulty sentence vs its own correction, before and after tuning.
    ax = fig.add_subplot(right_grid[0, 0])
    cos_before = 1 - np.load(embeddings / "bert_full_detect_s42_before_paircos.npy")
    cos_after = 1 - np.load(embeddings / "bert_full_detect_s42_after_paircos.npy")
    for values, color, name in [(cos_before, CLEAN_C, "微调前"), (cos_after, FAULT_C, "微调后")]:
        ordered = np.sort(np.clip(values, 1e-5, None))
        ax.plot(ordered, np.linspace(0, 1, len(ordered)), color=color, linewidth=1.3)
        median = float(np.median(ordered))
        report[f"pair_distance_median_{name}"] = median
    ax.set_xscale("log")
    ax.set_xlabel("病句与其改正句的余弦距离")
    ax.set_ylabel("累计比例")
    ax.set_ylim(0, 1)
    ax.text(0.03, 0.97, f"微调前\n中位 {report['pair_distance_median_微调前']:.4f}", transform=ax.transAxes, fontsize=5.8,
            color=MUTED, va="top", linespacing=1.15)
    ax.text(0.98, 0.04, f"微调后\n中位 {report['pair_distance_median_微调后']:.4f}", transform=ax.transAxes, fontsize=5.8,
            color=FAULT_C, va="bottom", ha="right", linespacing=1.15)
    panel_label(ax, "b", x=-0.28, y=1.0)

    # c: unfreezing strategies from the paper
    ax = fig.add_subplot(right_grid[0, 1])
    strategies = [("bert_head_s42", "仅训分类头"), ("bert_top2_s42", "解冻顶 2 层"),
                  ("bert_layerwise_s42", "逐层解冻"), ("bert_full_s42", "全量微调")]
    y = np.arange(len(strategies))[::-1]
    for row_y, (arm, name) in zip(y, strategies):
        row = summary.get(arm)
        if not row:
            continue
        color = BERT_C if arm != "bert_head_s42" else PRETRAINED
        ax.plot([row["detection_low"], row["detection_high"]], [row_y, row_y], color=color, linewidth=1.3,
                solid_capstyle="butt")
        ax.scatter([row["detection"]], [row_y], s=20, color=color, zorder=3, edgecolor="white", linewidth=0.5)
        ax.text(row["detection_high"] + 0.006, row_y, f"{row['detection']:.3f}", va="center", fontsize=6, color=INK)
    ax.set_yticks(y, [s[1] for s in strategies], fontsize=6.2, color=INK)
    ax.set_ylim(-0.6, len(strategies) - 0.4)
    ax.set_xlabel("病句检测 Accuracy")
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    panel_label(ax, "c", x=-0.56, y=1.0)
    lows = [summary[s[0]]["detection_low"] for s in strategies if s[0] in summary]
    highs = [summary[s[0]]["detection_high"] for s in strategies if s[0] in summary]
    if lows:
        ax.set_xlim(min(lows) - 0.01, max(highs) + 0.05)

    (results / "representation_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    save(fig, out, "fig2_representation", alignment=False)


# ---------------------------------------------------------------- figure 3
def figure3(results: Path, out: Path) -> None:
    summary = load_csv(results / "summary.csv")
    tests = load_csv(results / "paired_tests.csv")
    tests_by_arm = {row["a"]: row for row in tests.values() if row["b"] == "sft_s42"}

    fig = plt.figure(figsize=(6.0, 4.0))
    grid = GridSpec(2, 3, figure=fig, height_ratios=[1, 1.05], hspace=0.62, wspace=0.42,
                    left=0.075, right=0.985, top=0.93, bottom=0.105)

    # a: DPO only rewards the gap, so watch both log-ratios
    ax = fig.add_subplot(grid[0, 0])
    dpo_log = load_log(results / "logs" / "dpo_beta0.1.jsonl")
    train = dpo_log.get("train", [])
    if train:
        steps = [r["step"] for r in train]
        chosen = smooth([r["chosen_logratio"] for r in train])
        rejected = smooth([r["rejected_logratio"] for r in train])
        ax.axhline(0, color=MUTED, linewidth=0.6)
        ax.plot(steps, chosen, color=SFT_C, linewidth=1.3)
        ax.plot(steps, rejected, color=DPO_C, linewidth=1.3)
        ax.text(0.97, 0.97, "好回答（chosen）", transform=ax.transAxes, fontsize=6, color=SFT_C, ha="right", va="top")
        ax.text(0.04, 0.05, "坏回答（rejected）", transform=ax.transAxes, fontsize=6, color=DPO_C, ha="left", va="bottom")
    ax.set_xlabel("DPO 训练步（β=0.1）")
    ax.set_ylabel("对数概率比（相对参考模型）")
    panel_label(ax, "a", x=-0.17, y=1.03)

    # b: does a better preference margin mean a better diagnoser?
    ax = fig.add_subplot(grid[0, 1])
    dev = dpo_log.get("dev", [])
    if dev:
        steps = [r["step"] for r in dev]
        placements = {"detection": (0, 6), "type_macro_f1": (0, -9), "edit_f05": (0, 6)}
        for key, label, marker in [("detection", "检测", "o"), ("type_macro_f1", "类型", "s"), ("edit_f05", "改正", "^")]:
            values = [r[key] for r in dev]
            ax.plot(steps, values, color=DPO_C, linewidth=1.1, marker=marker, markersize=3,
                    markeredgecolor="white", markeredgewidth=0.4)
            ax.annotate(label, (steps[3], values[3]), xytext=placements[key], textcoords="offset points", fontsize=6,
                        color=INK, ha="center")
    ax.set_xlabel("DPO 训练步（β=0.1）")
    ax.set_ylabel("开发集得分")
    ax.set_ylim(0, 0.9)
    panel_label(ax, "b", x=-0.17, y=1.03)

    # c: change against the SFT model on the test set
    ax = fig.add_subplot(grid[0, 2])
    arms = [("dpo_beta0.1", "DPO", DPO_C), ("ppo_rubric", "PPO\n量规", PPO_C), ("ppo_naive", "PPO\n相似度", PPO_C)]
    rows_y = np.arange(len(arms))[::-1]
    ax.axvline(0, color=MUTED, linewidth=0.7)
    metrics = [("detection", "检测", "o", 0.24), ("type", "类型", "s", 0.0), ("edit", "改正", "^", -0.24)]
    for row_y, (arm, name, color) in zip(rows_y, arms):
        row = tests_by_arm.get(arm)
        if not row:
            continue
        for key, _, marker, offset in metrics:
            y = row_y + offset
            ax.plot([row[f"{key}_ci_low"], row[f"{key}_ci_high"]], [y, y], color=color, linewidth=1.0, solid_capstyle="butt")
            ax.scatter([row[f"{key}_diff"]], [y], s=13, color=color, marker=marker, edgecolor="white", linewidth=0.4, zorder=3)
    for _, label, marker, _ in metrics:
        ax.scatter([], [], s=13, color=MUTED, marker=marker, label=label)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=3, fontsize=5.8, handletextpad=0.05, columnspacing=0.7, borderpad=0)
    ax.set_yticks(rows_y, [a[1] for a in arms], fontsize=6, color=INK)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_ylim(-0.6, len(arms) - 0.4)
    ax.set_xlabel("测试集相对 SFT 的变化")
    soft_grid(ax, "x")
    panel_label(ax, "c", x=-0.3, y=1.03)

    # d: PPO reward with its per-batch spread
    ax = fig.add_subplot(grid[1, 0])
    ppo = {name: load_log(results / "logs" / f"ppo_{name}.jsonl") for name in ("rubric", "naive")}
    train = ppo["rubric"].get("train", [])
    if train:
        x = [r["update"] for r in train]
        mean = np.array([r["reward_mean"] for r in train])
        std = np.array([r["reward_std"] for r in train])
        ax.fill_between(x, smooth(mean - std), smooth(mean + std), color=PPO_C, alpha=0.18, linewidth=0)
        ax.plot(x, mean, color=PPO_C, linewidth=0.5, alpha=0.55)
        ax.plot(x, smooth(mean), color=PPO_C, linewidth=1.4)
    ax.set_xlabel("PPO 更新次数")
    ax.set_ylabel("评分奖励（均值 ±1 SD）")
    soft_grid(ax, "y")
    panel_label(ax, "d", x=-0.17, y=1.03)

    # e: reward design decides behaviour
    ax = fig.add_subplot(grid[1, 1])
    styles = {"rubric": ("评分量规奖励", PPO_C, "-"), "naive": ("字面相似度奖励", DPO_C, (0, (3, 1.5)))}
    for name, (label, color, style) in styles.items():
        rows = ppo[name].get("train", [])
        if rows:
            ax.plot([r["update"] for r in rows], smooth([r["copy_rate"] for r in rows]), color=color,
                    linewidth=1.3, linestyle=style, label=label)
    ax.set_xlabel("PPO 更新次数")
    ax.set_ylabel("病句未改动比例（采样输出）")
    ax.set_ylim(0.1, 0.72)
    ax.legend(loc="upper left", fontsize=5.8, handlelength=1.8, borderaxespad=0.1)
    panel_label(ax, "e", x=-0.17, y=1.03)

    # f: what each method costs
    ax = fig.add_subplot(grid[1, 2])
    costs = []
    sft = load_log(results / "logs" / "sft_s42.jsonl").get("summary", [{}])[0]
    if sft:
        per_step = train_seconds_per_step(results / "logs" / "sft_s42.jsonl", "step")
        costs.append(("SFT", per_step / (sft["n_train"] * sft["epochs"] / sft["steps"]) * 1000, SFT_C))
    dpo = load_log(results / "logs" / "dpo_beta0.1.jsonl").get("summary", [{}])[0]
    if dpo:
        per_step = train_seconds_per_step(results / "logs" / "dpo_beta0.1.jsonl", "step")
        costs.append(("DPO", per_step / dpo["sequences_per_step"] * 1000, DPO_C))
    rubric = ppo["rubric"].get("summary", [{}])[0]
    if rubric:
        per_update = train_seconds_per_step(results / "logs" / "ppo_rubric.jsonl", "update")
        costs.append(("PPO", per_update / (rubric["episodes"] / rubric["updates"]) * 1000, PPO_C))
    (results / "cost_report.json").write_text(
        json.dumps({name: seconds for name, seconds, _ in costs}, ensure_ascii=False, indent=2), encoding="utf-8")
    rows_y = np.arange(len(costs))[::-1]
    for row_y, (name, seconds, color) in zip(rows_y, costs):
        ax.barh(row_y, seconds, color=color, height=0.46, zorder=2)
        ax.text(seconds, row_y, f" {seconds:,.0f} 秒", va="center", ha="left", fontsize=6.2, color=INK)
    ax.set_yticks(rows_y, [c[0] for c in costs], color=INK)
    ax.set_xlabel("每处理 1000 条序列的耗时（秒）")
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    if costs:
        ax.set_xlim(0, max(c[1] for c in costs) * 1.35)
    panel_label(ax, "f", x=-0.3, y=1.03)

    save(fig, out, "fig3_ppo_dpo")


# ---------------------------------------------------------------- figure 4
def box(ax, x, y, w, h, title, lines, color, fill="#ffffff") -> None:
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.004,rounding_size=0.012", linewidth=0.9,
                                edgecolor=color, facecolor=fill, transform=ax.transAxes))
    ax.text(x + 0.012, y + h - 0.05, title, transform=ax.transAxes, fontsize=7.2, fontweight="bold", color=INK, va="top")
    ax.text(x + 0.012, y + h - 0.21, "\n".join(lines), transform=ax.transAxes, fontsize=5.9, color=MUTED, va="top",
            linespacing=1.32)


def arrow(ax, start, end, color=MUTED) -> None:
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=7, color=color, linewidth=0.9,
                                 transform=ax.transAxes, shrinkA=0, shrinkB=0))


def figure4(results: Path, out: Path) -> None:
    summary = load_csv(results / "summary.csv")
    data = json.loads((results / "data_report.json").read_text(encoding="utf-8"))
    sft_log = load_log(results / "logs" / "sft_s42.jsonl").get("summary", [{}])[0]
    splits = data["splits"]
    tuned = summary.get("sft_s42", {})

    pairs = json.loads((results / "dpo_pairs_report.json").read_text(encoding="utf-8"))
    fig = plt.figure(figsize=(6.0, 1.28))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    width, height, gap = 0.178, 0.94, 0.0245
    top = 0.03
    xs = [0.006 + i * (width + gap) for i in range(5)]
    params = sft_log.get("params", {})
    minutes = sft_log.get("seconds", 0) / 60
    box(ax, xs[0], top, width, height, "① 语料与审计",
        ["FCGEC 中文病句语料", f"训练 {sft_log.get('n_train', splits['main']['n']):,} 句", f"开发 {splits['dev']['n']:,} 句",
         f"测试 {splits['test']['n']:,} 句",
         f"同源剔除 {data['homology_audit']['removed_from_training']:,} 句"], MUTED)
    box(ax, xs[1], top, width, height, "② 指令微调",
        ["Qwen3-1.7B + LoRA", f"可训练 {params.get('ratio', 0) * 100:.1f}% 参数", "判断/类型/修改/改正",
         "四行结构化输出", f"{sft_log.get('steps', 0):,} 步 · {minutes:.0f} 分钟"], SFT_C)
    box(ax, xs[2], top, width, height, "③ 对齐微调",
        ["DPO：模型真实错误", f"挖出 {pairs['pairs']:,} 对偏好", "PPO：在线采样 +", "量规 / 相似度奖励"], DPO_C)
    box(ax, xs[3], top, width, height, "④ 对比评测",
        ["预训练两组 + BERT", "检测 / 类型 / 改正", "自助法 95% CI",
         f"检测 {tuned.get('detection', 0):.3f}"], BERT_C)
    box(ax, xs[4], top, width, height, "⑤ 应用",
        ["LoRA 适配器", "Gradio 对比界面", "结果浏览网页", "语文批改与", "二语写作反馈"], PPO_C)
    y_mid = top + height / 2
    for i in range(4):
        arrow(ax, (xs[i] + width + 0.002, y_mid), (xs[i + 1] - 0.002, y_mid))
    save(fig, out, "fig4_pipeline", alignment=False)


# ---------------------------------------------------------------- logo, banner and short-report figures
TILE_BG = "#13303b"


def draw_logo(ax, *, outline: str = "none") -> None:
    """Draw the project tile into a square axes: a marked-up page above the name, three diagnosis steps below."""
    fig = ax.figure
    scale = ax.get_position().width * fig.get_figwidth() * 72
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.add_patch(FancyBboxPatch((0.03, 0.03), 0.94, 0.94, boxstyle="round,pad=0,rounding_size=0.17",
                                facecolor=TILE_BG, edgecolor=outline, linewidth=0.5))
    ax.text(0.14, 0.775, "ZHENJU", fontsize=scale * 0.062, fontweight="bold", color="#c9dbe3", va="center")
    ax.add_patch(mpl.patches.Polygon([(0.70, 0.67), (0.70, 0.88), (0.80, 0.88), (0.87, 0.81), (0.87, 0.67)],
                                     closed=True, facecolor="#2f6fe0", edgecolor="none"))
    ax.add_patch(mpl.patches.Polygon([(0.80, 0.88), (0.87, 0.81), (0.80, 0.81)], closed=True,
                                     facecolor="#8db8f5", edgecolor="none"))
    ax.plot([0.73, 0.84], [0.775, 0.775], color="#ff8a80", linewidth=scale * 0.008, solid_capstyle="round")
    ax.plot([0.73, 0.82], [0.725, 0.725], color="#5fe0b5", linewidth=scale * 0.008, solid_capstyle="round")
    ax.text(0.125, 0.46, "诊句", fontsize=scale * 0.215, fontweight="bold", color="white", va="center")
    ax.plot([0.15, 0.85], [0.2, 0.2], color="#5f8a99", linewidth=scale * 0.005, zorder=1)
    for x, color in [(0.15, "#5b9df0"), (0.5, "#3fc1a8"), (0.85, "white")]:
        ax.scatter([x], [0.2], s=(scale * 0.034) ** 2, color=color, zorder=2, linewidths=0)


def figure_logo(out: Path) -> None:
    fig = plt.figure(figsize=(2, 2))
    draw_logo(fig.add_axes([0, 0, 1, 1]))
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "logo.png", dpi=256, transparent=True)
    plt.close(fig)
    print("saved logo")


def figure_hero(results: Path, out: Path, with_logo: bool, name: str, language: str = "zh", *,
                dark: bool = False) -> None:
    """One real test sentence before and after tuning, above the whole-test-set change."""
    summary = load_csv(results / "summary.csv")
    zero, tuned = summary["pretrained_zero"], summary["sft_s42"]
    width, height = 6.0, 1.5
    fig = plt.figure(figsize=(width, height))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_axis_off()
    left = 0.02
    if with_logo:
        logo = fig.add_axes([0.0, 0.03 / height, (height - 0.06) / width, (height - 0.06) / height])
        draw_logo(logo, outline="#365865" if dark else "none")
        left = height + 0.08

    if language == "en":
        heading = "ZhenJu-Qwen · Chinese grammar diagnosis"
        source = "Input  从这篇文章中，给了我很大的启发和鼓励。"
        before = ("Before · base model", "Predicts a word-order error.", "Rewords the verb. The subject is still missing.")
        after = ("After · LoRA", "Identifies the missing subject.", "这篇文章，给了我很大的启发和鼓励。")
        metric_labels = ("Detection accuracy", "Error-type macro F1", "Edit F0.5")
    else:
        heading = "诊句：微调一个小模型，让它学会给句子“看病”"
        source = "原句：从这篇文章中，给了我很大的启发和鼓励。"
        before = ("微调前（预训练模型）", "判成“语序不当”；把“给了我”改成“给予”", "语病还在。")
        after = ("微调后", "判成“成分残缺”；删去“从”“中”：", "这篇文章，给了我很大的启发和鼓励。")
        metric_labels = ("判断准确率", "错误归类 F1", "改正得分 F0.5")

    if dark:
        ink, muted, accent = "#e6edf3", "#abb8c7", "#7bbaff"
        before_palette = ("#1b2530", "#374554", "#ff988f")
        after_palette = ("#182c3e", "#32536e", "#75dfbb")
    else:
        ink, muted, accent = INK, MUTED, SFT_C
        before_palette = ("#f3f4f6", "none", "#b4372c")
        after_palette = ("#e9f0f9", "none", "#14704f")

    ax.text(left, 1.39, heading, fontsize=8.8, fontweight="bold", color=ink, va="center")
    ax.text(left, 1.195, source, fontsize=6.8, color=ink, va="center")

    gap = 0.1
    card_w = (width - left - 0.02 - gap) / 2
    cards = [
        (left, "×", before_palette, before),
        (left + card_w + gap, "√", after_palette, after),
    ]
    for x, mark, (fill, outline, mark_color), (title, line1, line2) in cards:
        ax.add_patch(FancyBboxPatch((x, 0.47), card_w, 0.61, boxstyle="round,pad=0,rounding_size=0.05",
                                    facecolor=fill, edgecolor=outline, linewidth=0.5))
        ax.text(x + 0.08, 0.955, mark, fontsize=8.2, fontweight="bold", color=mark_color, va="center")
        ax.text(x + 0.22, 0.955, title, fontsize=6.8, fontweight="bold", color=ink, va="center")
        ax.text(x + 0.08, 0.785, line1, fontsize=6.3, color=ink, va="center")
        ax.text(x + 0.08, 0.61, line2, fontsize=6.3, color=ink, va="center")

    metrics = zip(metric_labels, ("detection", "type_macro_f1", "edit_f05"))
    kpi_w = (width - left - 0.02) / 3
    for index, (label, key) in enumerate(metrics):
        x = left + index * kpi_w
        ax.text(x + 0.02, 0.33, label, fontsize=6.1, color=muted, va="center")
        ax.text(x + 0.02, 0.13, f"{zero[key]:.3f} → {tuned[key]:.3f}", fontsize=9.2, fontweight="bold",
                color=accent, va="center")
    fig.savefig(out / f"{name}.png", dpi=400, transparent=dark, facecolor="none" if dark else "white")
    plt.close(fig)
    print("saved", name)


def figure_scores(results: Path, out: Path) -> None:
    summary = load_csv(results / "summary.csv")
    arms = [a for a in ARMS if a in summary]
    fig = plt.figure(figsize=(6.0, 1.32))
    grid = GridSpec(1, 3, figure=fig, wspace=0.14, left=0.135, right=0.985, top=0.84, bottom=0.17)
    specs = [
        ("detection", "detection_low", "detection_high", "判断准确率", (0.45, 0.92), None),
        ("type_macro_f1", "type_low", "type_high", "错误归类 F1", (0.0, 0.75), None),
        ("edit_f05", "edit_low", "edit_high", "改正得分 F0.5", (0.0, 0.62), "只分类，不生成改正句"),
    ]
    axes = [fig.add_subplot(grid[0, i]) for i in range(3)]
    for ax, (metric, low, high, title, xlim, na) in zip(axes, specs):
        tier_panel(ax, summary, arms, metric, low, high, title, xlim, na)
    y = np.arange(len(arms))[::-1]
    axes[0].set_yticks(y, [ARMS[a][0] for a in arms], color=INK, fontsize=6.4)
    for ax in axes[1:]:
        ax.set_yticks(y, [""] * len(arms))
    save(fig, out, "scores", alignment=False)


def figure_alignment(results: Path, out: Path) -> None:
    dpo = load_log(results / "logs" / "dpo_beta0.1.jsonl")
    ppo = load_log(results / "logs" / "ppo_rubric.jsonl")
    fig = plt.figure(figsize=(6.0, 1.5))
    grid = GridSpec(1, 3, figure=fig, wspace=0.34, left=0.06, right=0.985, top=0.86, bottom=0.235)
    axes = [fig.add_subplot(grid[0, i]) for i in range(3)]

    ax = axes[0]
    train = dpo["train"]
    steps = [r["step"] for r in train]
    ax.axhline(0, color=MUTED, linewidth=0.6)
    ax.plot(steps, smooth([r["chosen_logratio"] for r in train]), color=SFT_C, linewidth=1.2)
    ax.plot(steps, smooth([r["rejected_logratio"] for r in train]), color=DPO_C, linewidth=1.2)
    ax.text(0.98, 0.95, "好回答", transform=ax.transAxes, fontsize=6, color=SFT_C, ha="right", va="top")
    ax.text(0.04, 0.05, "坏回答", transform=ax.transAxes, fontsize=6, color=DPO_C, ha="left", va="bottom")
    ax.set_title("DPO：好坏回答的差距被拉大", loc="left", color=INK, fontweight="bold", fontsize=6.8, pad=3)
    ax.set_xlabel("DPO 训练步")
    ax.set_ylabel("相对对数概率")

    ax = axes[1]
    dev = dpo["dev"]
    steps = [r["step"] for r in dev]
    for key, label, marker, offset in [("detection", "判断", "o", 5), ("edit_f05", "改正", "^", 5), ("type_macro_f1", "归类", "s", -8)]:
        values = [r[key] for r in dev]
        ax.plot(steps, values, color=DPO_C, linewidth=1.0, marker=marker, markersize=2.6,
                markeredgecolor="white", markeredgewidth=0.3)
        ax.annotate(label, (steps[3], values[3]), xytext=(0, offset), textcoords="offset points", fontsize=5.8,
                    color=INK, ha="center")
    ax.set_ylim(0.1, 0.9)
    ax.set_title("DPO：开发集得分却在下降", loc="left", color=INK, fontweight="bold", fontsize=6.8, pad=3)
    ax.set_xlabel("DPO 训练步")
    ax.set_ylabel("开发集得分")

    ax = axes[2]
    train = ppo["train"]
    x = [r["update"] for r in train]
    mean = np.array([r["reward_mean"] for r in train])
    std = np.array([r["reward_std"] for r in train])
    ax.fill_between(x, smooth(mean - std), smooth(mean + std), color=PPO_C, alpha=0.18, linewidth=0)
    ax.plot(x, smooth(mean), color=PPO_C, linewidth=1.3)
    ax.set_title("PPO：奖励波动远大于提升", loc="left", color=INK, fontweight="bold", fontsize=6.8, pad=3)
    ax.set_xlabel("PPO 更新次数")
    ax.set_ylabel("奖励（均值 ±1 SD）")
    save(fig, out, "alignment", alignment=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--results", default=ROOT / "results", type=Path)
    parser.add_argument("--out", default=ROOT / "figures", type=Path)
    parser.add_argument("--auditor-dir", default=None, help="directory holding audit_panel_alignment.py")
    parser.add_argument("--only", default="1,2,3,4,short")
    args = parser.parse_args()

    if args.auditor_dir:
        sys.path.insert(0, args.auditor_dir)
    wanted = args.only.split(",")
    if "1" in wanted:
        figure1(args.results, args.out)
    if "2" in wanted:
        figure2(args.results, args.embeddings, args.data, args.out)
    if "3" in wanted:
        figure3(args.results, args.out)
    if "4" in wanted:
        figure4(args.results, args.out)
    if "short" in wanted:
        figure_logo(args.out)
        figure_hero(args.results, args.out, with_logo=True, name="hero")
        figure_hero(args.results, args.out, with_logo=True, name="hero-dark", dark=True)
        figure_hero(args.results, args.out, with_logo=True, name="hero_en", language="en")
        figure_hero(args.results, args.out, with_logo=True, name="hero_en-dark", language="en", dark=True)
        figure_hero(args.results, args.out, with_logo=False, name="hero_example")
        figure_scores(args.results, args.out)
        figure_alignment(args.results, args.out)


if __name__ == "__main__":
    main()
