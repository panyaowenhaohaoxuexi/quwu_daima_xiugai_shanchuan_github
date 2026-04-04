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

tick_dict = {
    "FADE":    [0.1, 0.3, 0.5, 0.7, 0.9, 1.1],
    "BRISQUE": [15, 18, 21, 24, 27],
    "PM2.5":   [45, 55, 65, 75, 85],
}

settings = ["w/o CMEA", "w/o ReMix", "w/o CDA", "Full Model"]
colors = ["#7BB4E3", "#EDA1A4", "#81C7B3", "#C7A8D6"]

def value_to_angle(value, vmin, vmax, start_deg=90, total_span=300):
    frac = (value - vmin) / (vmax - vmin)
    frac = np.clip(frac, 0, 1)
    return start_deg - frac * total_span

def draw_ring_subplot(ax, metric_name, metric_values, tick_values):
    ax.set_aspect("equal")
    ax.axis("off")

    start_deg = 90
    total_span = 300
    vmin, vmax = min(tick_values), max(tick_values)

    theta = np.linspace(np.deg2rad(start_deg), np.deg2rad(start_deg - total_span), 500)
    ref_r = 1.18
    ax.plot(ref_r * np.cos(theta), ref_r * np.sin(theta), linewidth=0.6, color="#555555")

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

        ax.text(x, y, txt, ha="center", va="center", rotation=rot, fontsize=12, family='serif')

    # 【关键修复】：根据名称长度，单独为每个子图定制坐标 (X, Y)
    # BRISQUE 比较长，所以把它往更左边推 (-1.75)，保证右侧间距和 FADE 视觉一致
    title_pos = {
        "FADE":    (-1.45, 1.35),
        "BRISQUE": (-1.75, 1.35),
        "PM2.5":   (-1.45, 1.35)
    }
    tx, ty = title_pos[metric_name]

    ax.text(tx, ty, metric_name, ha="left", va="center",
            fontsize=20, fontweight="bold", family='serif')

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
            edgecolor="white",
            linewidth=0.5
        )
        ax.add_patch(ring)

    # 【关键修复】：扩大坐标轴的左侧范围 (从 -1.5 改为 -1.8)
    # 这样 BRISQUE 往左挪的时候，最左边的字母才不会被画布边缘切掉
    ax.set_xlim(-1.8, 1.5)
    ax.set_ylim(-1.4, 1.4)

# =========================
# Draw figure
# =========================
# 稍微把画布拉宽拉高一点点，给文字留出从容的物理空间
fig, axes = plt.subplots(1, 3, figsize=(14, 5.2))

for ax, metric in zip(axes, ["FADE", "BRISQUE", "PM2.5"]):
    draw_ring_subplot(ax, metric, data[metric], tick_dict[metric])

handles = [
    plt.Line2D([0], [0], color=c, lw=10, label=s)
    for c, s in zip(colors, settings)
]

# 保持你调整好的完美图例高度
fig.legend(
    handles=handles,
    loc="lower center",
    ncol=4,
    frameon=False,
    fontsize=16,
    bbox_to_anchor=(0.5, 0.12),
    prop={'family': 'serif', 'size': 16}
)

# 保持底部安全距离
plt.subplots_adjust(wspace=0.15, bottom=0.2)

plt.savefig("realworld_ablation_ring_3subplots.png", dpi=300, bbox_inches="tight", pad_inches=0.03)
plt.savefig("realworld_ablation_ring_3subplots.pdf", bbox_inches="tight", pad_inches=0.03)
plt.show()