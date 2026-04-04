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

# 【修改2】扩宽刻度范围（尤其是拉低起点），保证 Full Model 有足够的展示弧长
tick_dict = {
    "FADE":    [0.1, 0.3, 0.5, 0.7, 0.9, 1.1],
    "BRISQUE": [15, 18, 21, 24, 27],
    "PM2.5":   [45, 55, 65, 75, 85],
}

settings = ["w/o CMEA", "w/o ReMix", "w/o CDA", "Full Model"]

# 【修改1】使用高级淡雅配色（蓝、粉、绿、紫），替代默认的刺眼颜色
colors = ["#7BB4E3", "#EDA1A4", "#81C7B3", "#C7A8D6"]

def value_to_angle(value, vmin, vmax, start_deg=90, total_span=300):
    """
    将数值映射到角度。Lower is better。
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

    # 参考圆弧（外圈黑线）
    theta = np.linspace(np.deg2rad(start_deg), np.deg2rad(start_deg - total_span), 500)
    ref_r = 1.18
    ax.plot(ref_r * np.cos(theta), ref_r * np.sin(theta), linewidth=0.6, color="#555555")

    # 刻度文字
    for tv in tick_values:
        ang_deg = value_to_angle(tv, vmin, vmax, start_deg, total_span)
        ang = np.deg2rad(ang_deg)
        x = 1.28 * np.cos(ang)
        y = 1.28 * np.sin(ang)
        rot = ang_deg - 90

        if metric_name == "FADE":
            txt = f"{tv:.1f}"
        else:
            txt = f"{int(tv)}"

        ax.text(x, y, txt, ha="center", va="center", rotation=rot, fontsize=9, family='serif')

    # 【修改3】缩小标题字号，向左上角微调位置，并设置衬线字体
    ax.text(-1.25, 1.20, metric_name, ha="left", va="center",
            fontsize=16, fontweight="bold", family='serif')

    # 4 个同心环（由外向内）
    radii = [0.98, 0.78, 0.58, 0.38]
    width = 0.14

    for r, setting, color in zip(radii, settings, colors):
        value = metric_values[setting]
        end_deg = value_to_angle(value, vmin, vmax, start_deg, total_span)

        ring = Wedge(
            center=(0, 0),
            r=r,
            theta1=end_deg,  # 起始弧度（较小的角度）
            theta2=start_deg, # 结束弧度（90度）
            width=width,
            facecolor=color,
            edgecolor="white", # 加上一点白色描边会让各环之间更清晰
            linewidth=0.5
        )
        ax.add_patch(ring)

    ax.set_xlim(-1.45, 1.45)
    ax.set_ylim(-1.35, 1.35)


# =========================
# Draw figure
# =========================
fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6))

for ax, metric in zip(axes, ["FADE", "BRISQUE", "PM2.5"]):
    draw_ring_subplot(ax, metric, data[metric], tick_dict[metric])

# 图例绘制
handles = [
    plt.Line2D([0], [0], color=c, lw=10, label=s)
    for c, s in zip(colors, settings)
]

# 图例也换成了更学术的衬线字体
fig.legend(
    handles=handles,
    loc="lower center",
    ncol=4,
    frameon=False,
    fontsize=13,
    bbox_to_anchor=(0.5, -0.05),
    prop={'family': 'serif', 'size': 13}
)

plt.subplots_adjust(wspace=0.15, bottom=0.15)
plt.savefig("realworld_ablation_ring_3subplots.png", dpi=300, bbox_inches="tight", pad_inches=0.03)
plt.savefig("realworld_ablation_ring_3subplots.pdf", bbox_inches="tight", pad_inches=0.03)
plt.show()