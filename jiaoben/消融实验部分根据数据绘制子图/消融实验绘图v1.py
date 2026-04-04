import matplotlib.pyplot as plt
from matplotlib.patches import Wedge
import numpy as np

# =========================
# Real-world domain data
# =========================
data = {
    "FADE": {
        "w/o CMEA": 0.5865,
        "w/o ReMix": 0.8772,
        "w/o CDA": 0.9677,
        "Full Model": 0.3256,
    },
    "BRISQUE": {
        "w/o CMEA": 22.3201,
        "w/o ReMix": 25.2041,
        "w/o CDA": 26.0199,
        "Full Model": 20.0917,
    },
    "PM2.5": {
        "w/o CMEA": 71.0032,
        "w/o ReMix": 82.1211,
        "w/o CDA": 72.4881,
        "Full Model": 59.1232,
    },
}

# 手动设置刻度范围，便于和论文图风格一致
tick_dict = {
    "FADE":    [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    "BRISQUE": [20, 21, 22, 23, 24, 25, 26, 27],
    "PM2.5":   [55, 60, 65, 70, 75, 80, 85],
}

settings = ["w/o CMEA", "w/o ReMix", "w/o CDA", "Full Model"]

# 使用 matplotlib 默认颜色
colors = plt.rcParams['axes.prop_cycle'].by_key()['color'][:4]


def value_to_angle(value, vmin, vmax, start_deg=90, total_span=300):
    """
    将数值映射到角度。
    这里 lower is better，所以值越小，弧越短，更靠近起点。
    """
    frac = (value - vmin) / (vmax - vmin)
    frac = np.clip(frac, 0, 1)
    return start_deg - frac * total_span


def draw_ring_subplot(ax, metric_name, metric_values, tick_values):
    ax.set_aspect("equal")
    ax.axis("off")

    start_deg = 90
    total_span = 300
    vmin, vmax = min(tick_values), max(tick_values)

    # 参考圆弧
    theta = np.linspace(np.deg2rad(start_deg), np.deg2rad(start_deg - total_span), 500)
    ref_r = 1.18
    ax.plot(ref_r * np.cos(theta), ref_r * np.sin(theta), linewidth=0.8, color="black")

    # 刻度文字
    for tv in tick_values:
        ang_deg = value_to_angle(tv, vmin, vmax, start_deg, total_span)
        ang = np.deg2rad(ang_deg)
        x = 1.28 * np.cos(ang)
        y = 1.28 * np.sin(ang)
        rot = ang_deg - 90

        if metric_name == "FADE":
            txt = f"{tv:.1f}"
        elif metric_name == "BRISQUE":
            txt = f"{int(tv)}"
        else:
            txt = f"{int(tv)}"

        ax.text(x, y, txt, ha="center", va="center", rotation=rot, fontsize=10)

    # 标题
    ax.text(-1.12, 1.08, metric_name, ha="left", va="top",
            fontsize=24, fontweight="bold")

    # 4 个同心环
    radii = [0.98, 0.78, 0.58, 0.38]
    width = 0.14

    for r, setting, color in zip(radii, settings, colors):
        value = metric_values[setting]
        end_deg = value_to_angle(value, vmin, vmax, start_deg, total_span)

        ring = Wedge(
            center=(0, 0),
            r=r,
            theta1=end_deg,
            theta2=start_deg,
            width=width,
            facecolor=color,
            edgecolor="none"
        )
        ax.add_patch(ring)

    ax.set_xlim(-1.38, 1.38)
    ax.set_ylim(-1.28, 1.18)


# =========================
# Draw figure
# =========================
fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6))

for ax, metric in zip(axes, ["FADE", "BRISQUE", "PM2.5"]):
    draw_ring_subplot(ax, metric, data[metric], tick_dict[metric])

# 图例
handles = [
    plt.Line2D([0], [0], color=c, lw=10, label=s)
    for c, s in zip(colors, settings)
]
fig.legend(
    handles=handles,
    loc="lower center",
    ncol=4,
    frameon=False,
    fontsize=12,
    bbox_to_anchor=(0.5, -0.02)
)

plt.subplots_adjust(wspace=0.28, bottom=0.18)
plt.savefig("realworld_ablation_ring_3subplots.png", dpi=300, bbox_inches="tight", pad_inches=0.03)
plt.savefig("realworld_ablation_ring_3subplots.pdf", bbox_inches="tight", pad_inches=0.03)
plt.show()