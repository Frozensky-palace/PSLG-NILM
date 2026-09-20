"""Evaluate state exchangeability and export example waveforms (Phase A2).

Reads one or more train-only state libraries, computes per-state distribution,
distance, replacement-boundary and multimodality diagnostics, exports example
waveform figures and a combined Markdown report. Validation/test data are never
read.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Windows CJK-capable fonts first; minus sign must render as ASCII hyphen.
matplotlib.rcParams["font.sans-serif"] = [
    "Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.validation.state_exchangeability import (  # noqa: E402
    EXAMPLE_CATEGORIES,
    StateLibrary,
    exchangeability_verdict,
    load_state_library,
    select_example_blocks,
    state_metrics,
    transition_context_stats,
)

# Validated categorical palette (fixed slot order), light-mode surface.
SERIES_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                 "#e87ba4", "#008300", "#4a3aa7", "#e34948")
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID_COLOR = "#e1e0d9"
SURFACE = "#fcfcfb"

CATEGORY_LABELS_ZH = {
    "typical": "典型（特征中位）",
    "shape_typical": "形状最典型",
    "shape_outlier": "形状最异常",
    "shortest": "最短",
    "longest": "最长",
    "highest_power": "最高平均功率",
    "lowest_power": "最低平均功率",
}


def _style_axis(ax: plt.Axes) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID_COLOR, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#c3c2b7")
    ax.tick_params(colors=INK_MUTED, labelsize=8)
    ax.xaxis.label.set_color(INK_SECONDARY)
    ax.yaxis.label.set_color(INK_SECONDARY)
    ax.title.set_color(INK_PRIMARY)


def _plot_waveform(ax: plt.Axes, library: StateLibrary, row_index: int,
                   title: str, color: str) -> None:
    power = library.waveform(row_index)
    seconds = np.arange(len(power)) * library.sample_seconds
    row = library.inventory.iloc[row_index]
    ax.plot(seconds / 60.0, power, color=color, linewidth=1.2)
    _style_axis(ax)
    ax.set_title(
        f"{title} · 块{int(row['state_block_id'])} · "
        f"{int(row['duration_seconds'])}s · {row['mean_power_w']:.0f}W",
        fontsize=9)
    ax.set_xlabel("时间（分钟）", fontsize=8)
    ax.set_ylabel("功率 (W)", fontsize=8)


def _state_color(library: StateLibrary, state_label: int) -> str:
    order = sorted(library.inventory["state_label"].unique())
    return SERIES_COLORS[order.index(state_label) % len(SERIES_COLORS)]


def export_waveform_examples(library: StateLibrary, output_dir: Path) -> dict:
    selections = select_example_blocks(library)
    written = {}
    for state_label, categories in selections.items():
        color = _state_color(library, state_label)
        fig, axes = plt.subplots(4, 2, figsize=(10, 12), dpi=150)
        fig.patch.set_facecolor(SURFACE)
        for ax in axes.flat[len(categories):]:
            ax.axis("off")
        for ax, category in zip(axes.flat, categories):
            _plot_waveform(ax, library, categories[category],
                           CATEGORY_LABELS_ZH[category], color)
        fig.suptitle(
            f"{library.tag} 状态 {state_label}：典型/边缘/异常波形示例",
            fontsize=12, color=INK_PRIMARY)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        path = output_dir / f"waveform_examples_{library.tag}_state{state_label}.png"
        fig.savefig(path, facecolor=SURFACE)
        plt.close(fig)
        written[state_label] = path.name
    return written


def export_state_distributions(library: StateLibrary, metrics: pd.DataFrame,
                               output_dir: Path) -> str:
    fields = (
        ("duration_seconds", "时长 (s)"),
        ("mean_power_w", "平均功率 (W)"),
        ("energy_wh", "能量 (Wh)"),
        ("start_power_w", "起点功率 (W)"),
        ("end_power_w", "终点功率 (W)"),
        ("mean_abs_slope_w_per_sample", "平均绝对斜率 (W/样本)"),
    )
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.5), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    inventory = library.inventory
    for ax, (field, label) in zip(axes.flat, fields):
        for state_label in sorted(inventory["state_label"].unique()):
            values = inventory.loc[
                inventory["state_label"] == state_label, field
            ].to_numpy(dtype=float)
            sorted_values = np.sort(values)
            ecdf = np.arange(1, len(sorted_values) + 1) / len(sorted_values)
            ax.plot(sorted_values, ecdf,
                    color=_state_color(library, state_label), linewidth=1.6,
                    label=f"状态 {state_label}")
        _style_axis(ax)
        ax.set_xlabel(label, fontsize=9)
        ax.set_ylabel("ECDF", fontsize=8)
    handles, names = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, names, loc="lower center", ncol=5, frameon=False,
               fontsize=9)
    fig.suptitle(f"{library.tag}：各状态分布对比", fontsize=12,
                 color=INK_PRIMARY)
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    path = output_dir / f"state_distributions_{library.tag}.png"
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    return path.name


def export_nn_summary(all_metrics: pd.DataFrame, output_dir: Path) -> str:
    """Grouped bars over (tag, state) pairs; states missing in a tag are skipped."""
    pairs = list(all_metrics[["tag", "state_label"]]
                 .drop_duplicates()
                 .itertuples(index=False, name=None))
    pair_labels = [f"{tag}·状态{s}" for tag, s in pairs]
    width = 1.1 * len(pairs)
    fig, axes = plt.subplots(1, 2, figsize=(max(8.0, width), 4.2), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    tag_order = list(dict.fromkeys(all_metrics["tag"]))
    for axis, column, label in (
            (axes[0], "nn_same_cycle_fraction", "最近邻同 cycle 比例（越低越好）"),
            (axes[1], "replaced_over_original", "替换后/原始边界跳变比（越接近 1 越好）")):
        values = []
        colors = []
        for tag, state_label in pairs:
            row = all_metrics[(all_metrics["tag"] == tag)
                              & (all_metrics["state_label"] == state_label)]
            values.append(float(row[column].iloc[0]))
            colors.append(SERIES_COLORS[tag_order.index(tag) % len(SERIES_COLORS)])
        axis.bar(np.arange(len(pairs)), values, width=0.72, color=colors)
        _style_axis(axis)
        axis.set_xticks(np.arange(len(pairs)))
        axis.set_xticklabels(pair_labels, rotation=45, ha="right", fontsize=8)
        axis.set_ylabel(label, fontsize=9)
    handles = [plt.Rectangle((0, 0), 1, 1, color=SERIES_COLORS[i])
               for i in range(len(tag_order))]
    fig.legend(handles, tag_order, loc="lower center", ncol=len(tag_order),
               frameon=False, fontsize=9)
    fig.suptitle("状态可交换性核心指标（跨状态库）", fontsize=12,
                 color=INK_PRIMARY)
    fig.tight_layout(rect=(0, 0.06, 1, 0.94))
    path = output_dir / "nn_exchangeability_summary.png"
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    return path.name


def _format_table(metrics: pd.DataFrame, columns: list[str],
                  headers: list[str]) -> str:
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in metrics.itertuples(index=False):
        values = []
        for column in columns:
            value = getattr(row, column)
            if isinstance(value, float):
                values.append(f"{value:,.3f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def build_report(libraries: list[StateLibrary], all_metrics: pd.DataFrame,
                 figure_files: dict[str, list[str]], output_dir: Path
                 ) -> Path:
    md = ["# k=3/4/5 状态可交换性分析（Phase A2，train-only）", ""]
    md.append("> 输入全部为 train-only pilot 状态库；未读取 validation/test。")
    md.append("> **产物可见性说明**：PNG 图与 CSV 是本地产物（Git 只追踪本 "
              "Markdown），在其他机器上图片链接不可用；本报告的全部结论"
              "由下列数字表格完整承载，波形示例的身份由示例索引表给出，"
              "可在对应状态库的 `state_inventory.csv` 中按 "
              "`state_block_id` 检索波形。")
    md.append("")
    metric_columns = [
        "tag", "state_label", "blocks", "cycles", "median_duration_s",
        "median_power_w", "within_over_between", "nn_same_cycle_fraction",
        "replaced_over_original", "multimodal_entropy",
    ]
    metric_headers = [
        "库", "状态", "状态块", "覆盖 cycle", "中位时长 (s)", "中位功率 (W)",
        "内/间距离比", "最近邻同cycle比例", "替换边界跳变比", "多模态熵",
    ]
    md.append("## 1. 核心指标总表")
    md.append("")
    md.append(_format_table(all_metrics, metric_columns, metric_headers))
    md.append("")
    extended_columns = [
        "tag", "state_label", "duration_p5_s", "duration_p95_s",
        "power_p5_w", "power_p95_w", "median_energy_wh", "median_start_w",
        "median_end_w", "multimodal_subcluster_sizes",
    ]
    extended_headers = [
        "库", "状态", "时长 p5 (s)", "时长 p95 (s)", "功率 p5 (W)",
        "功率 p95 (W)", "中位能量 (Wh)", "起点中位 (W)", "终点中位 (W)",
        "形状子类规模",
    ]
    md.append("## 2. 分布细节表（报告自足性用）")
    md.append("")
    md.append(_format_table(all_metrics, extended_columns, extended_headers))
    md.append("")
    md.append("## 3. 波形示例索引表（图件的本机文件名）")
    md.append("")
    md.append("波形内容随状态库 CSV 保存；下表给出每个状态被选中的示例块。")
    md.append("")
    for library in libraries:
        selections = select_example_blocks(library)
        md.append(f"### {library.tag}")
        md.append("")
        header = "| 状态 | " + " | ".join(CATEGORY_LABELS_ZH) + " |"
        align = "|" + "---|" * (len(CATEGORY_LABELS_ZH) + 1)
        md.append(header)
        md.append(align)
        for state_label, categories in selections.items():
            cells = " | ".join(
                str(categories[category]) for category in CATEGORY_LABELS_ZH)
            md.append(f"| {state_label} | {cells} |")
        md.append("")
    md.append("## 4. 逐状态结论")
    md.append("")
    for library in libraries:
        metrics = all_metrics[all_metrics["tag"] == library.tag]
        md.append(f"### {library.tag}")
        md.append("")
        for row in metrics.itertuples(index=False):
            md.append(f"- **状态 {row.state_label}**（{row.blocks} 块，"
                      f"{row.cycles} cycle，中位 {row.median_duration_s:,.0f} s"
                      f"/{row.median_power_w:,.0f} W）"
                      f"→ {exchangeability_verdict(pd.Series(row._asdict()))}")
        md.append("")
    md.append("## 5. 图件（仅本机可看）")
    md.append("")
    md.append(f"- 跨库汇总：nn_exchangeability_summary.png")
    for library in libraries:
        for name in figure_files[library.tag]:
            md.append(f"- {name}")
    md.append("")
    md.append("## 6. 使用说明")
    md.append("")
    md.append("- 直接交换候选：可作为 B2/B4 基元重组与生成的优先对象；")
    md.append("- 需要条件化：donor/生成需匹配端点功率、前驱后继或时长；")
    md.append("- 需要继续细分：多模态熵高的长状态（k=4/5 的拆分动机）。")
    path = output_dir / "k3_k4_k5_exchangeability.md"
    path.write_text("\n".join(md), encoding="utf-8")
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--state-library", action="append", required=True,
                    help="tag=directory, repeatable, train-only libraries")
    ap.add_argument("--output-dir", default="reports/state_quality")
    ap.add_argument("--sample-seconds", type=int, default=6)
    ap.add_argument("--seed", type=int, default=17)
    args = ap.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    libraries = []
    for item in args.state_library:
        tag, _, directory = item.partition("=")
        if not directory:
            raise SystemExit(f"--state-library must be tag=directory: {item}")
        library = load_state_library(tag.strip(), Path(directory))
        library.sample_seconds = args.sample_seconds
        libraries.append(library)

    all_metrics_parts = []
    figure_files: dict[str, list[str]] = {}
    context_frames = []
    for library in libraries:
        metrics = state_metrics(library, rng_seed=args.seed)
        all_metrics_parts.append(metrics)
        files = []
        files.extend(export_waveform_examples(library, output_dir).values())
        files.append(export_state_distributions(library, metrics, output_dir))
        context = transition_context_stats(library.inventory)
        context.insert(0, "tag", library.tag)
        context_frames.append(context)
        figure_files[library.tag] = files
        print(f"[exchangeability] {library.tag}: "
              f"{len(metrics)} states analyzed")
    all_metrics = pd.concat(all_metrics_parts, ignore_index=True)
    all_metrics.to_csv(output_dir / "state_exchangeability_metrics.csv",
                       index=False)
    pd.concat(context_frames, ignore_index=True).to_csv(
        output_dir / "state_transition_context.csv", index=False)
    figure_files["cross_library"] = [export_nn_summary(all_metrics, output_dir)]
    report = build_report(libraries, all_metrics, figure_files, output_dir)
    print(f"[exchangeability] metrics csv -> {output_dir / 'state_exchangeability_metrics.csv'}")
    print(f"[exchangeability] report -> {report}")


if __name__ == "__main__":
    main()
