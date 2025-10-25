
from graphviz import Digraph

# 创建 Graphviz 有向图
dot = Digraph(
    name="VisibleStreamFlowDetailed",
    format="png",
    graph_attr={
        "rankdir": "LR",  # 从左到右布局
        "splines": "polyline",  # 直线边
        "bgcolor": "white",  # 背景白色
        "fontfamily": "SimHei",  # 支持中文
        "fontsize": "12",
    },
    node_attr={
        "shape": "box",
        "style": "filled",
        "fontfamily": "SimHei",
        "fontsize": "12",
    },
    edge_attr={
        "color": "navy",
        "penwidth": "2",
        "arrowsize": "1.2",
        "fontfamily": "SimHei",
        "fontsize": "10",
    }
)

# 输入节点
dot.node("InputVis", "输入 (可见光)\n(1, 3, 256, 256)", fillcolor="lightgreen")

# 编码器子图 (Res2Net)
with dot.subgraph(name="cluster_encoder_vis") as s:
    s.attr(label="encoder_vis (Res2Net)", style="filled", fillcolor="lightblue", penwidth="2")
    s.node("Layer0Vis", "layer0\n(1, 64, 128, 128)")
    s.node("Layer1Vis", "layer1\n(1, 256, 64, 64)")
    s.node("Layer2Vis", "layer2\n(1, 512, 32, 32)")
    s.node("Layer3Vis", "layer3\n(1, 1024, 16, 16)")
    s.edge("Layer0Vis", "Layer1Vis")
    s.edge("Layer1Vis", "Layer2Vis")
    s.edge("Layer2Vis", "Layer3Vis")

# CRA 层
with dot.subgraph(name="cluster_cra_vis") as s:
    s.attr(label="CRA 层", style="filled", fillcolor="lightcyan", penwidth="2")
    s.node("CRA1Vis", "CRA1_vis\n(1024 → 256)")
    s.node("CRA2Vis", "CRA2_vis\n(512 → 128)")
    s.node("CRA3Vis", "CRA3_vis\n(256 → 64)")
    s.node("CRA4Vis", "CRA4_vis\n(64 → 32)")
    s.attr(rank="same")

# 去雾模块子图 (dehaze_vis)
with dot.subgraph(name="cluster_dehaze_vis") as s:
    s.attr(label="dehaze_vis (18 ResidualBlocks)", style="filled", fillcolor="orange", penwidth="2")
    s.node("ResBlock1", "ResidualBlock 1\n(256 → 256)")
    s.node("ResBlockDots", "...", shape="plaintext")
    s.node("ResBlock18", "ResidualBlock 18\n(256 → 256)")
    with s.subgraph(name="cluster_residual_block") as rb:
        rb.attr(label="ResidualBlock 结构", style="filled", fillcolor="lightyellow", penwidth="1")
        rb.node("Conv1RB", "ConvLayer\n(3x3, 256 → 256)")
        rb.node("PReLURB", "PReLU")
        rb.node("Conv2RB", "ConvLayer\n(3x3, 256 → 256)")
        rb.node("ResidualRB", "残差连接\n(*0.1 + input)")
        rb.edge("Conv1RB", "PReLURB")
        rb.edge("PReLURB", "Conv2RB")
        rb.edge("Conv2RB", "ResidualRB")
    s.edge("ResBlock1", "ResBlockDots")
    s.edge("ResBlockDots", "ResBlock18")

# 解码器子图 (16x)
with dot.subgraph(name="cluster_decoder16x") as s:
    s.attr(label="解码器 16x", style="filled", fillcolor="purple", penwidth="2")
    s.node("Convd16xVis", "convd16x_vis\n上采样")
    s.node("Dense4Vis", "dense_4_vis\n密集连接")
    s.node("Conv4Vis", "conv_4_vis\n卷积")
    s.node("Fusion4Vis", "fusion_4_vis\n特征融合")
    s.edge("Convd16xVis", "Dense4Vis")
    s.edge("Dense4Vis", "Conv4Vis")
    s.edge("Conv4Vis", "Fusion4Vis")

# 解码器子图 (8x)
with dot.subgraph(name="cluster_decoder8x") as s:
    s.attr(label="解码器 8x", style="filled", fillcolor="purple", penwidth="2")
    s.node("Convd8xVis", "convd8x_vis\n上采样")
    s.node("Dense3Vis", "dense_3_vis\n密集连接")
    s.node("Conv3Vis", "conv_3_vis\n卷积")
    s.node("Fusion3Vis", "fusion_3_vis\n特征融合")
    s.edge("Convd8xVis", "Dense3Vis")
    s.edge("Dense3Vis", "Conv3Vis")
    s.edge("Conv3Vis", "Fusion3Vis")

# 解码器子图 (4x)
with dot.subgraph(name="cluster_decoder4x") as s:
    s.attr(label="解码器 4x", style="filled", fillcolor="purple", penwidth="2")
    s.node("Convd4xVis", "convd4x_vis\n上采样")
    s.node("Dense2Vis", "dense_2_vis\n密集连接")
    s.node("Conv2Vis", "conv_2_vis\n卷积")
    s.node("Fusion2Vis", "fusion_2_vis\n特征融合")
    s.edge("Convd4xVis", "Dense2Vis")
    s.edge("Dense2Vis", "Conv2Vis")
    s.edge("Conv2Vis", "Fusion2Vis")

# 解码器子图 (2x)
with dot.subgraph(name="cluster_decoder2x") as s:
    s.attr(label="解码器 2x", style="filled", fillcolor="purple", penwidth="2")
    s.node("Convd2xVis", "convd2x_vis\n上采样")
    s.node("Dense1Vis", "dense_1_vis\n密集连接")
    s.node("Conv1Vis", "conv_1_vis\n卷积")
    s.node("Fusion1Vis", "fusion_1_vis\n特征融合")
    s.edge("Convd2xVis", "Dense1Vis")
    s.edge("Dense1Vis", "Conv1Vis")
    s.edge("Conv1Vis", "Fusion1Vis")

# H 层子图
with dot.subgraph(name="cluster_h_layers") as s:
    s.attr(label="H 层", style="filled", fillcolor="lightyellow", penwidth="2")
    s.node("H1Vis", "H1_vis\n(256 → 128)")
    s.node("H2Vis", "H2_vis\n(128 → 64)")
    s.node("H3Vis", "H3_vis\n(64 → 32)")
    s.node("H4Vis", "H4_vis\n(32 → 16)")
    s.attr(rank="same")

# 融合与输出
dot.node("IRFused", "ir_features\n(1, 16, 256, 256)", shape="box", fillcolor="lightcoral")
dot.node("Fusion", "融合\n(torch.cat)", shape="ellipse", fillcolor="pink")
dot.node("ConvOutput", "conv_output\n(32 → 3)", shape="box", fillcolor="lightblue")
dot.node("Output", "输出\n(1, 3, 256, 256)", shape="box", fillcolor="lightgreen")

# 数据流连接
dot.edge("InputVis", "Layer0Vis", label="(1, 3, 256, 256)")
dot.edge("Layer3Vis", "CRA1Vis", label="x_layer3\n(1, 1024, 16, 16)")
dot.edge("Layer2Vis", "CRA2Vis", label="x_layer2\n(1, 512, 32, 32)")
dot.edge("Layer1Vis", "CRA3Vis", label="x_layer1\n(1, 256, 64, 64)")
dot.edge("Layer0Vis", "CRA4Vis", label="x_layer0\n(1, 64, 128, 128)")
dot.edge("CRA1Vis", "ResBlock1", label="res16x\n(1, 256, 16, 16)")
dot.edge("ResBlock18", "Convd16xVis", label="res16x_dehazed\n(1, 256, 16, 16)")
dot.edge("CRA2Vis", "Fusion4Vis", label="res8x\n(1, 128, 32, 32)")
dot.edge("Fusion4Vis", "Convd8xVis", label="res8x_out\n(1, 128, 32, 32)")
dot.edge("CRA3Vis", "Fusion3Vis", label="res4x\n(1, 64, 64, 64)")
dot.edge("Fusion3Vis", "Convd4xVis", label="res4x_out\n(1, 64, 64, 64)")
dot.edge("CRA4Vis", "Fusion2Vis", label="res2x\n(1, 32, 128, 128)")
dot.edge("Fusion2Vis", "Convd2xVis", label="res2x_out\n(1, 32, 128, 128)")
dot.edge("Fusion1Vis", "Fusion", label="vis_features\n(1, 16, 256, 256)")
dot.edge("CRA1Vis", "H1Vis", label="res16x\n(1, 256, 16, 16)")
dot.edge("CRA2Vis", "H2Vis", label="res8x\n(1, 128, 32, 32)")
dot.edge("CRA3Vis", "H3Vis", label="res4x\n(1, 64, 64, 64)")
dot.edge("CRA4Vis", "H4Vis", label="res2x\n(1, 32, 128, 128)")
dot.edge("H1Vis", "HOutput", label="(1, 128, 16, 16)")
dot.edge("H2Vis", "HOutput", label="(1, 64, 32, 32)")
dot.edge("H3Vis", "HOutput", label="(1, 32, 64, 64)")
dot.edge("H4Vis", "HOutput", label="(1, 16, 128, 128)")
dot.edge("IRFused", "Fusion", label="(1, 16, 256, 256)")
dot.edge("Fusion", "ConvOutput", label="fused_features\n(1, 32, 256, 256)")
dot.edge("ConvOutput", "Output", label="(1, 3, 256, 256)")

# 设置节点排列层次
with dot.subgraph() as s:
    s.attr(rank="same")
    s.node("InputVis")
with dot.subgraph() as s:
    s.attr(rank="same")
    s.node("cluster_encoder_vis")
with dot.subgraph() as s:
    s.attr(rank="same")
    s.node("cluster_cra_vis")
with dot.subgraph() as s:
    s.attr(rank="same")
    s.node("cluster_dehaze_vis")
with dot.subgraph() as s:
    s.attr(rank="same")
    s.node("cluster_decoder16x")
with dot.subgraph() as s:
    s.attr(rank="same")
    s.node("cluster_decoder8x")
with dot.subgraph() as s:
    s.attr(rank="same")
    s.node("cluster_decoder4x")
with dot.subgraph() as s:
    s.attr(rank="same")
    s.node("cluster_decoder2x")
with dot.subgraph() as s:
    s.attr(rank="same")
    s.node("cluster_h_layers")
with dot.subgraph() as s:
    s.attr(rank="same")
    s.node("Fusion")
    s.node("IRFused")
    s.node("HOutput")
with dot.subgraph() as s:
    s.attr(rank="same")
    s.node("ConvOutput")
with dot.subgraph() as s:
    s.attr(rank="same")
    s.node("Output")

# 保存和渲染流程图
dot.save("visible_stream_flow_detailed.dot")
dot.render("visible_stream_flow_detailed", view=True)

print("流程图已生成：'visible_stream_flow_detailed.png'")
