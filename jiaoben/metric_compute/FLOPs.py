import torch
from thop import profile
# 确保导入了你的模型类
from model import VIFNetInconsistencyTeacher


def count_model_complexity():
    # 1. 硬件设备选择
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. 实例化模型并移动到设备
    # 注意：这里使用未经过 DataParallel 包装的原始类
    model = VIFNetInconsistencyTeacher().to(device)
    model.eval()

    # 3. 创建虚拟输入 (Dummy Inputs)
    # 根据你的训练代码 (EMA.py)，输入尺寸通常是 256x256
    # 形状为 (Batch_size, Channels, Height, Width)
    input_vis = torch.randn(1, 3, 256, 256).to(device)  # 可见光输入
    input_ir = torch.randn(1, 3, 256, 256).to(device)  # 红外输入

    # 4. 使用 thop 计算 FLOPs 和参数量
    # 注意：由于是双输入模型，需要以元组形式传入 inputs
    flops, params = profile(model, inputs=(input_vis, input_ir), verbose=False)

    # 5. 格式化输出结果
    print("-" * 30)
    print(f"输入尺寸: 256x256")
    print(f"总参数量 (Params): {params / 1e6:.2f} M")
    print(f"浮点运算量 (FLOPs): {flops / 1e9:.2f} G")
    print("-" * 30)


if __name__ == "__main__":
    count_model_complexity()