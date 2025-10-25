# -*- coding: utf-8 -*-
from graphviz import Digraph

# === 全局风格 ===
FONT = "Microsoft YaHei, Arial"
dot = Digraph("DualStreamTeacher_VisibleStream", format="png")
dot.attr(rankdir="LR", splines="ortho", nodesep="0.4", ranksep="0.6")
dot.attr(fontname=FONT)

# 统一节点风格
NODE_STYLE = {
    "shape": "box",
    "style": "rounded,filled",
    "penwidth": "1.5",
    "fontname": FONT,
}

# 统一辅助函数
def add_node(g, name, label, fillcolor):
    g.node(name, label=label, fillcolor=fillcolor, **NODE_STYLE)

def add_chain_edges(g, names):
    """按顺序串接：n0->n1->n2..."""
    g.edges([(names[i], names[i+1]) for i in range(len(names)-1)])

# =============== 顶层总览（主干流程占位） ===============
with dot.subgraph(name="cluster_overall") as overall:
    overall.attr(label="可见光流（Visible Stream）— Overall Flow", color="black", fontsize="16")

    add_node(overall, "input", "Input (x_vis)\n(B, 3, H, W)", "gold")
    add_node(overall, "encoder_blk", "Encoder (Res2Net)", "lightskyblue")
    add_node(overall, "cra_blk", "CRA 映射 (CRA1–CRA4)", "paleturquoise")
    add_node(overall, "dehaze_blk", "Dehaze 残差细化 (×18)", "palegreen")
    add_node(overall, "decoder_blk", "Decoder 逐级上采样与融合\n(Stage1→4)", "lightpink")
    add_node(overall, "vis_feat_out", "可见光特征输出 (B, 16, H, W)\n(后续与 IR 特征拼接，未在本图绘制)", "lightgray")

    add_chain_edges(overall, ["input", "encoder_blk", "cra_blk", "dehaze_blk", "decoder_blk", "vis_feat_out"])

# =============== Encoder 细节 ===============
with dot.subgraph(name="cluster_encoder") as enc:
    enc.attr(label="Encoder（Res2Net，输出多尺度特征）", color="deepskyblue", fontsize="14")
    # conv1 + BN + ReLU
    add_node(enc, "enc_conv1", "conv1: 3→64, s=2, k=3\nBN + ReLU", "aliceblue")
    # 这里单独放一个 maxpool，因为 CRA4 要从 maxpool 前取 x_layer0
    add_node(enc, "enc_maxpool", "MaxPool k=3, s=2, p=1", "aliceblue")
    # layer1-3
    add_node(enc, "enc_l1", "layer1: Bottle2neck×3\n64→256 @ 1/4", "aliceblue")
    add_node(enc, "enc_l2", "layer2: Bottle2neck×4\n128→512 @ 1/8", "aliceblue")
    add_node(enc, "enc_l3", "layer3: Bottle2neck×23\n256→1024 @ 1/16", "aliceblue")

    add_chain_edges(enc, ["enc_conv1", "enc_maxpool", "enc_l1", "enc_l2", "enc_l3"])

    # 和整体占位相连（让总览里的占位块有真实锚点）
    enc.edge("input", "enc_conv1", lhead="cluster_encoder")

# =============== CRA 细节 ===============
with dot.subgraph(name="cluster_cra") as cra:
    cra.attr(label="CRA（Cross-scale Reduction Adapter）\n将不同尺度的 encoder 特征压缩到统一通道", color="teal", fontsize="14")

    # 四个 1x1 conv 对应：layer3, layer2, layer1, conv1输出(即maxpool前的x_layer0)
    add_node(cra, "cra1", "CRA1: 1024→256 (1×1)\n来自 layer3 @1/16 → res16x", "azure")
    add_node(cra, "cra2", "CRA2: 512→128 (1×1)\n来自 layer2 @1/8 → res8x", "azure")
    add_node(cra, "cra3", "CRA3: 256→64 (1×1)\n来自 layer1 @1/4 → res4x", "azure")
    add_node(cra, "cra4", "CRA4: 64→32 (1×1)\n来自 conv1输出 @1/2 → res2x", "azure")

    # 从 Encoder 对应尺度接线
    cra.edge("enc_l3", "cra1")
    cra.edge("enc_l2", "cra2")
    cra.edge("enc_l1", "cra3")
    cra.edge("enc_conv1", "cra4")

    # CRA 块和占位块的衔接
    dot.edge("encoder_blk", "cra1", lhead="cluster_cra", color="gray50")
    dot.edge("encoder_blk", "cra2", color="gray50")
    dot.edge("encoder_blk", "cra3", color="gray50")
    dot.edge("encoder_blk", "cra4", color="gray50")

# =============== Dehaze 细节 ===============
with dot.subgraph(name="cluster_dehaze") as deh:
    deh.attr(label="Dehaze（深层特征残差细化）", color="darkgreen", fontsize="14")
    add_node(deh, "deh_in", "输入：res16x (B,256,H/16,W/16)\n= CRA1(c3)", "honeydew")
    # 用 3 段表示 18 个残差块（视觉更简洁）
    add_node(deh, "deh_rb_1", "ResidualBlock ×6", "honeydew")
    add_node(deh, "deh_rb_2", "ResidualBlock ×6", "honeydew")
    add_node(deh, "deh_rb_3", "ResidualBlock ×6", "honeydew")
    add_node(deh, "deh_out", "res16x_dehazed = RBs(res16x) + res16x", "honeydew")

    add_chain_edges(deh, ["deh_in", "deh_rb_1", "deh_rb_2", "deh_rb_3", "deh_out"])

    # CRA1 → Dehaze 入口
    deh.edge("cra1", "deh_in")

    # 和总览占位连接
    dot.edge("cra_blk", "deh_in", lhead="cluster_dehaze")

# =============== Decoder（四个 Stage） ===============
with dot.subgraph(name="cluster_decoder") as dec:
    dec.attr(label="Decoder（四级上采样与跨尺度融合，带记忆栈）", color="firebrick", fontsize="14")

    # ---------- Memory Stack 可视化（feature_mem_up） ----------
    with dec.subgraph(name="cluster_mem") as mem:
        mem.attr(label="Memory Stack（feature_mem_up）", color="gray40", fontsize="12")
        add_node(mem, "mem_res16x_1", "push: res16x_1 (B,128,1/16)\n来自 Split(res16x_dehazed)", "whitesmoke")
        add_node(mem, "mem_res8x_1", "push: res8x_1 (B,64,1/8)", "whitesmoke")
        add_node(mem, "mem_res4x_1", "push: res4x_1 (B,32,1/4)", "whitesmoke")
        # 不串接，按需虚线指向各 MDCBlock

    # ========== Stage 1: 16× → 8× ==========
    with dec.subgraph(name="cluster_stage1") as s1:
        s1.attr(label="Stage1: 1/16 → 1/8", color="indianred", fontsize="12")

        add_node(s1, "s1_up", "UpsampleConvLayer\n256→128, stride=2\n(最近邻+1×1)", "mistyrose")
        add_node(s1, "s1_align", "对齐到 res8x 尺度\nbilinear align", "mistyrose")
        add_node(s1, "s1_add", "Add: (res16x_up + res8x)", "mistyrose")
        add_node(s1, "s1_dense", "Dense: 3×ResidualBlock\n+ 残差回加", "mistyrose")
        add_node(s1, "s1_split", "Split: 128 → 64 + 64\n(res8x_1, res8x_2)", "mistyrose")
        add_node(s1, "s1_mdc", "MDCBlock1(iter2)\n输入: res8x_1\n记忆: [res16x_1]", "mistyrose")
        add_node(s1, "s1_rdb", "RDB(64, L=4, g=64)\nconv_4_vis", "mistyrose")
        add_node(s1, "s1_concat", "Concat: [res8x_1', res8x_2']\n= res8x_out (B,128,1/8)", "mistyrose")

        add_chain_edges(s1, ["s1_up", "s1_align", "s1_add", "s1_dense", "s1_split"])
        s1.edge("s1_split", "s1_mdc", label="64")
        s1.edge("s1_split", "s1_rdb", label="64")
        s1.edge("s1_mdc", "s1_concat")
        s1.edge("s1_rdb", "s1_concat")

        # 输入来源
        s1.edge("deh_out", "s1_up")
        s1.edge("cra2", "s1_add", xlabel="res8x", color="black")

        # Memory: 先由 res16x_dehazed split 得 res16x_1（128通道）
        add_node(s1, "s1_res16x_split", "Split: res16x_dehazed →\nres16x_1 + res16x_2", "lavenderblush")
        s1.edge("deh_out", "s1_res16x_split")
        # 推入内存栈
        dec.edge("s1_res16x_split", "mem_res16x_1", style="dashed", color="gray40", xlabel="push")

        # 记忆流向 MDC
        dec.edge("mem_res16x_1", "s1_mdc", style="dashed", color="gray40", xlabel="memory")

    # ========== Stage 2: 8× → 4× ==========
    with dec.subgraph(name="cluster_stage2") as s2:
        s2.attr(label="Stage2: 1/8 → 1/4", color="indianred", fontsize="12")

        add_node(s2, "s2_up", "UpsampleConvLayer\n128→64, stride=2", "mistyrose")
        add_node(s2, "s2_align", "对齐到 res4x 尺度", "mistyrose")
        add_node(s2, "s2_add", "Add: (res8x_up + res4x)", "mistyrose")
        add_node(s2, "s2_dense", "Dense: 3×ResidualBlock", "mistyrose")
        add_node(s2, "s2_split", "Split: 64 → 32 + 32", "mistyrose")
        add_node(s2, "s2_mdc", "MDCBlock1(iter2)\n记忆: [res16x_1, res8x_1]", "mistyrose")
        add_node(s2, "s2_rdb", "RDB(32, L=4, g=32)\nconv_3_vis", "mistyrose")
        add_node(s2, "s2_concat", "Concat → res4x_out (B,64,1/4)", "mistyrose")

        add_chain_edges(s2, ["s2_up", "s2_align", "s2_add", "s2_dense", "s2_split"])
        s2.edge("s2_split", "s2_mdc")
        s2.edge("s2_split", "s2_rdb")
        s2.edge("s2_mdc", "s2_concat")
        s2.edge("s2_rdb", "s2_concat")

        # 上游输入：来自 Stage1
        dec.edge("s1_concat", "s2_up")

        # CRA3 提供 res4x
        s2.edge("cra3", "s2_add", xlabel="res4x", color="black")

        # 推入内存栈 res8x_1
        add_node(s2, "s2_push_mem", "push: res8x_1", "lavenderblush")
        s2.edge("s2_split", "s2_push_mem")
        dec.edge("s2_push_mem", "mem_res8x_1", style="dashed", color="gray40")
        # 内存到 MDC
        dec.edge("mem_res16x_1", "s2_mdc", style="dashed", color="gray40")
        dec.edge("mem_res8x_1", "s2_mdc", style="dashed", color="gray40")

    # ========== Stage 3: 4× → 2× ==========
    with dec.subgraph(name="cluster_stage3") as s3:
        s3.attr(label="Stage3: 1/4 → 1/2", color="indianred", fontsize="12")

        add_node(s3, "s3_up", "UpsampleConvLayer\n64→32, stride=2", "mistyrose")
        add_node(s3, "s3_align", "对齐到 res2x 尺度", "mistyrose")
        add_node(s3, "s3_add", "Add: (res4x_up + res2x)", "mistyrose")
        add_node(s3, "s3_dense", "Dense: 3×ResidualBlock", "mistyrose")
        add_node(s3, "s3_split", "Split: 32 → 16 + 16", "mistyrose")
        add_node(s3, "s3_mdc", "MDCBlock1(iter2)\n记忆: [res16x_1, res8x_1, res4x_1]", "mistyrose")
        add_node(s3, "s3_rdb", "RDB(16, L=4, g=16)\nconv_2_vis", "mistyrose")
        add_node(s3, "s3_concat", "Concat → res2x_out (B,32,1/2)", "mistyrose")

        add_chain_edges(s3, ["s3_up", "s3_align", "s3_add", "s3_dense", "s3_split"])
        s3.edge("s3_split", "s3_mdc")
        s3.edge("s3_split", "s3_rdb")
        s3.edge("s3_mdc", "s3_concat")
        s3.edge("s3_rdb", "s3_concat")

        # 上游输入：来自 Stage2
        dec.edge("s2_concat", "s3_up")
        # CRA4 提供 res2x
        s3.edge("cra4", "s3_add", xlabel="res2x", color="black")

        # 推入内存栈 res4x_1
        add_node(s3, "s3_push_mem", "push: res4x_1", "lavenderblush")
        s3.edge("s3_split", "s3_push_mem")
        dec.edge("s3_push_mem", "mem_res4x_1", style="dashed", color="gray40")

        # Memory → MDC
        dec.edge("mem_res16x_1", "s3_mdc", style="dashed", color="gray40")
        dec.edge("mem_res8x_1", "s3_mdc", style="dashed", color="gray40")
        dec.edge("mem_res4x_1", "s3_mdc", style="dashed", color="gray40")

    # ========== Stage 4: 2× → 1× ==========
    with dec.subgraph(name="cluster_stage4") as s4:
        s4.attr(label="Stage4: 1/2 → 1/1", color="indianred", fontsize="12")

        add_node(s4, "s4_up", "UpsampleConvLayer\n32→16, stride=2", "mistyrose")
        add_node(s4, "s4_dense", "Dense: 3×ResidualBlock", "mistyrose")
        add_node(s4, "s4_split", "Split: 16 → 8 + 8", "mistyrose")
        add_node(s4, "s4_mdc", "MDCBlock1(iter2)\n记忆: [res16x_1, res8x_1, res4x_1]", "mistyrose")
        add_node(s4, "s4_rdb", "RDB(8, L=4, g=8)\nconv_1_vis", "mistyrose")
        add_node(s4, "s4_concat", "Concat → x_out_before_fusion\n(B,16,1/1)", "mistyrose")

        add_chain_edges(s4, ["s4_up", "s4_dense", "s4_split"])
        s4.edge("s4_split", "s4_mdc")
        s4.edge("s4_split", "s4_rdb")
        s4.edge("s4_mdc", "s4_concat")
        s4.edge("s4_rdb", "s4_concat")

        # 上游输入：来自 Stage3
        dec.edge("s3_concat", "s4_up")

        # Memory → MDC
        dec.edge("mem_res16x_1", "s4_mdc", style="dashed", color="gray40")
        dec.edge("mem_res8x_1", "s4_mdc", style="dashed", color="gray40")
        dec.edge("mem_res4x_1", "s4_mdc", style="dashed", color="gray40")

    # Decoder 输出连到总览“可见光特征输出”
    dec.edge("s4_concat", "vis_feat_out", ltail="cluster_decoder")

# =============== H 层蒸馏分支（从 CRA 特征出发） ===============
with dot.subgraph(name="cluster_H") as H:
    H.attr(label="H 层（蒸馏特征）", color="purple", fontsize="14")

    add_node(H, "H1", "H1_vis: 256→128 (1×1)\n输入: res16x", "lavender")
    add_node(H, "H2", "H2_vis: 128→64 (1×1)\n输入: res8x", "lavender")
    add_node(H, "H3", "H3_vis: 64→32 (1×1)\n输入: res4x", "lavender")
    add_node(H, "H4", "H4_vis: 32→16 (1×1)\n输入: res2x", "lavender")

    # 从 CRA 各尺度输入
    H.edge("cra1", "H1")
    H.edge("cra2", "H2")
    H.edge("cra3", "H3")
    H.edge("cra4", "H4")

    # 和总览占位的虚线联系（提示这是中间监督/蒸馏分支）
    dot.edge("decoder_blk", "H1", style="dashed", color="mediumpurple", label="distill (intermediate)")
    dot.edge("decoder_blk", "H2", style="dashed", color="mediumpurple")
    dot.edge("decoder_blk", "H3", style="dashed", color="mediumpurple")
    dot.edge("decoder_blk", "H4", style="dashed", color="mediumpurple")

# =============== 说明 / 图例 ===============
with dot.subgraph(name="cluster_legend") as leg:
    leg.attr(label="图例（Legend）", color="gray30", fontsize="12")
    add_node(leg, "lg_enc", "Encoder 子模块", "lightskyblue")
    add_node(leg, "lg_cra", "CRA 1×1 压缩映射", "paleturquoise")
    add_node(leg, "lg_deh", "Dehaze 残差细化", "palegreen")
    add_node(leg, "lg_dec", "Decoder 上采样融合", "lightpink")
    add_node(leg, "lg_mem", "虚线：记忆栈/蒸馏连线", "whitesmoke")
    leg.edges([
        ("lg_enc", "lg_cra"),
        ("lg_cra", "lg_deh"),
        ("lg_deh", "lg_dec"),
    ])

# === 输出图像 ===
out_path = dot.render("VisibleStream_Detailed", view=True)
print("✅ 流程图已生成：", out_path)
