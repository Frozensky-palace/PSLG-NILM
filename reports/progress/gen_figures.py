"""Generate the progress-report figures from verified experiment numbers."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# dataviz palette: categorical slots 1-3, neutral ink
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"

OUT = "reports/progress/figures"


def style_ax(ax):
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.6, alpha=0.6)
    ax.set_axisbelow(True)


# ---- Fig 1: C2 k=3/4/5 状态块结构 ----
fig, ax = plt.subplots(figsize=(7.2, 3.6), dpi=150)
states = ["状态0", "状态1", "状态2", "状态3", "状态4"]
k3 = [555, 566, 441, 0, 0]
k4 = [467, 566, 422, 475, 0]
k5 = [475, 567, 467, 422, 2]
x = np.arange(5)
w = 0.26
for offset, vals, color, label in ((-w, k3, C1, "k=3"),
                                   (0, k4, C2, "k=4"),
                                   (w, k5, C3, "k=5")):
    bars = ax.bar(x + offset, vals, w * 0.92, color=color, label=label)
    for rect, v in zip(bars, vals):
        if v:
            ax.annotate(str(v), (rect.get_x() + rect.get_width() / 2, v),
                        ha="center", va="bottom", fontsize=7.5, color=INK2)
ax.annotate("退化态：仅 2 块\n（k=5 被排除的依据）",
            (4 + w, 2), xytext=(3.35, 220), fontsize=8, color=INK2,
            arrowprops=dict(arrowstyle="->", color=INK2, lw=0.8))
ax.set_xticks(x, states)
ax.set_ylabel("状态块数量", fontsize=9, color=INK)
ax.set_title("C2 正式状态发现：k=3/4/5 状态块结构（493 train 周期）",
             fontsize=10.5, color=INK, pad=10)
ax.legend(frameon=False, fontsize=8.5, ncol=3, loc="upper right")
style_ax(ax)
fig.tight_layout()
fig.savefig(f"{OUT}/fig1_c2_k345_structure.png", bbox_inches="tight")
plt.close(fig)

# ---- Fig 2: G-2 validation 选 k 复核 ----
fig, ax = plt.subplots(figsize=(7.2, 3.4), dpi=150)
names = ["C2-k3", "C2-k4\n(冻结 v1)", "C2-k5"]
maes = [24.305, 24.434, 24.334]
bars = ax.bar(names, maes, 0.45, color=C1)
for rect, v in zip(bars, maes):
    ax.annotate(f"{v:.3f}", (rect.get_x() + rect.get_width() / 2, v),
                ha="center", va="bottom", fontsize=8.5, color=INK)
ax.axhline(21.277, color=C2, linewidth=1.4, linestyle="--")
ax.annotate("B1 参考 21.277", (2.42, 21.277), ha="right", va="bottom",
            fontsize=8, color=C2)
ax.axhline(19.920, color=INK2, linewidth=1.2, linestyle=":")
ax.annotate("B0 参考 19.920", (2.42, 19.920), ha="right", va="top",
            fontsize=8, color=INK2)
ax.set_ylabel("B2-matched validation MAE (W)", fontsize=9, color=INK)
ax.set_ylim(0, 27)
ax.set_title("G-2 选 k 复核：三个 C2 库差异 ≤0.13W（噪声内），结构证据决定冻结 k=4",
             fontsize=10, color=INK, pad=10)
style_ax(ax)
fig.tight_layout()
fig.savefig(f"{OUT}/fig2_g2_k_check.png", bbox_inches="tight")
plt.close(fig)

# ---- Fig 3: C4 比例-性能曲线 ----
fig, ax = plt.subplots(figsize=(7.2, 3.8), dpi=150)
ratios = [0.5, 1.0, 2.0]
b1_mean, b1_std = [12.04, 10.14, 11.99], [2.90, 0.79, 2.59]
b2_mean, b2_std = [11.84, 10.55, 10.38], [1.61, 0.76, 0.08]
ax.errorbar(ratios, b1_mean, yerr=b1_std, marker="o", markersize=6,
            linewidth=2, color=C2, label="B1 完整周期重放", capsize=3)
ax.errorbar(ratios, b2_mean, yerr=b2_std, marker="s", markersize=6,
            linewidth=2, color=C1, label="B2-matched 基元拼接", capsize=3)
ax.axhline(13.73, color=INK2, linewidth=1.2, linestyle=":")
ax.annotate("B0 无增强 13.73", (0.52, 13.73), ha="left", va="bottom",
            fontsize=8, color=INK2)
for x, y in zip(ratios, b2_mean):
    ax.annotate(f"{y:.2f}", (x, y - 0.55), ha="center", fontsize=8.5,
                color=C1)
for x, y in zip(ratios, b1_mean):
    ax.annotate(f"{y:.2f}", (x, y + 0.35), ha="center", fontsize=8.5,
                color=C2)
ax.annotate("B1 退化\n(重复放置冗余)", (2.0, 11.99), xytext=(1.72, 12.6),
            fontsize=8, color=INK2,
            arrowprops=dict(arrowstyle="->", color=INK2, lw=0.8))
ax.set_xticks(ratios, ["0.5", "1.0", "2.0"])
ax.set_xlabel("合成数据比例（真实:合成）", fontsize=9, color=INK)
ax.set_ylabel("validation MAE (W)\n3 seeds mean±std", fontsize=9, color=INK)
ax.set_title("C4 比例实验：基元重组单调改善，完整周期重放不可扩展",
             fontsize=10.5, color=INK, pad=10)
ax.legend(frameon=False, fontsize=8.5, loc="upper right")
style_ax(ax)
fig.tight_layout()
fig.savefig(f"{OUT}/fig3_c4_ratio_curve.png", bbox_inches="tight")
plt.close(fig)

# ---- Fig 4: D/E 生成五路线双门 ----
fig, ax = plt.subplots(figsize=(7.2, 3.4), dpi=150)
routes = ["B3-T 变换", "B3-V CVAE", "B3-G WGAN", "B3-D 扩散", "B4 基元拼接"]
replication = [0.9919, 0.0, 0.0, 0.0, 0.0041]
colors = [C2 if r > 0.5 else C1 for r in replication]
bars = ax.barh(routes, replication, 0.5, color=colors)
for rect, v in zip(bars, replication):
    label = f"{v:.4f}" if v > 0 else "0.000"
    ax.annotate(label, (v + 0.012, rect.get_y() + rect.get_height() / 2),
                va="center", fontsize=8.5, color=INK)
ax.axvline(0.01, color=INK2, linewidth=1.0, linestyle=":")
ax.annotate("神经路线复制率上限 1%", (0.01, 4.45), ha="left", va="top",
            fontsize=8, color=INK2)
ax.annotate("B3-T 定义属性：\n形变真实周期本就近邻\n（exact 复制 = 0）",
            (0.9919, 0), xytext=(0.62, 1.15), fontsize=8, color=INK2,
            arrowprops=dict(arrowstyle="->", color=INK2, lw=0.8))
ax.set_xlabel("对真实 train 库的形状复制率（exact 复制全部为 0）",
              fontsize=9, color=INK)
ax.set_xlim(0, 1.12)
ax.set_title("D/E 生成五路线：质量门 + 记忆审计双门全部通过（5/5）",
             fontsize=10.5, color=INK, pad=10)
ax.grid(axis="x", color=GRID, linewidth=0.6, alpha=0.6)
ax.set_axisbelow(True)
for spine in ("top", "right"):
    ax.spines[spine].set_visible(False)
ax.tick_params(colors=INK2, labelsize=9)
fig.tight_layout()
fig.savefig(f"{OUT}/fig4_de_gates.png", bbox_inches="tight")
plt.close(fig)

print("figures ->", OUT)
