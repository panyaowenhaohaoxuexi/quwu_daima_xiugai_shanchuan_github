# RGB--TIR 二值路由去雾 V2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在不修改旧主模型、旧 HDE 验证代码和旧训练脚本的前提下，新建可训练的 RGB--TIR 去雾 V2：自动生成二值路由标签，以雾气、RGB 信息失效和 TIR 结构共同决定融合或硬补全。

**Architecture:** 先从现有 `M_GT` 和 TIR 自动构造并可视化 `Auto_Route_GT`；新数据加载器返回五元组 `(hazy, clear, tir, density, route_gt)`。新模型复用但不修改 `model/hde.py`，在其输出特征上增加信息可用性路由器，并以教师路由训练融合/补全分支。损失固定为重建、雾气浓度和路由 BCE 三项。

**Tech Stack:** Python 3.10、PyTorch、Torchvision、Pillow、NumPy、OpenCV（仅用于 Otsu/Sobel 标签生成）。

---

## 固定约束

- 不改动 `model/Teacher.py`、`model/hde.py`、`Teacher.py`、`test_single_pair.py` 或 HDE 独立验证工程。
- 不加载旧 `Teacher_Train/source_best.pt` 到 V2；该权重对应旧的 M-only 路由，参数结构也不兼容。
- `Transmission_Map_GT` 当前内容按雾气浓度 `M=1-T` 读取，不做反相。
- 训练和验证包含 `mist`、`middle`、`dense`、`local_extreme` 四个等级。
- `R_GT` 由代码生成，用户不手工绘制；推理只读取含雾 RGB 与 TIR。

### Task 1: 自动生成并审核二值 `Auto_Route_GT`

**Files:**
- Create: `tools/generate_auto_route_gt.py`
- Create: `tests/test_auto_route_gt.py`
- Create: `tools/visualize_auto_route_gt.py`
- Output: `F:\1_paper_pan\1_Dehaze_Paper\2_Dataset\1_main_benchmark\1_FLIR\{train,test}\Auto_Route_GT\{mist,middle,dense,local_extreme}\*.png`

- [ ] **Step 1: 写出会失败的标签构造单元测试**

  在 `tests/test_auto_route_gt.py` 定义以下三个最小样例：

  ```python
  def test_route_gt_is_one_only_for_opaque_tir_structure():
      density = np.array([[0.99, 0.99], [0.20, 0.99]], dtype=np.float32)
      tir = np.array([[0.0, 1.0], [0.5, 0.5]], dtype=np.float32)
      route = build_route_gt(density, tir, transmission_max=0.05)
      assert route.dtype == np.uint8
      assert route[0, 0] == 255 or route[0, 1] == 255
      assert route[1, 0] == 0

  def test_route_gt_rejects_opaque_but_structureless_sky():
      density = np.full((9, 9), 0.99, dtype=np.float32)
      tir = np.full((9, 9), 0.5, dtype=np.float32)
      assert not build_route_gt(density, tir, transmission_max=0.05).any()

  def test_route_gt_rejects_structured_low_fog_region():
      density = np.full((9, 9), 0.20, dtype=np.float32)
      tir = np.eye(9, dtype=np.float32)
      assert not build_route_gt(density, tir, transmission_max=0.05).any()
  ```

- [ ] **Step 2: 运行测试，确认尚未定义实现**

  Run: `D:\anaconda\envs\CoA\python.exe -m pytest tests\test_auto_route_gt.py -q`  
  Expected: FAIL，原因是 `tools.generate_auto_route_gt` 或 `build_route_gt` 尚不存在。

- [ ] **Step 3: 实现确定性的自动标签函数与批处理 CLI**

  在 `tools/generate_auto_route_gt.py` 实现：

  ```python
  def build_route_gt(density_m: np.ndarray, tir: np.ndarray, *, transmission_max: float = 0.05) -> np.ndarray:
      """Return uint8 route label: 255=completion, 0=fusion."""
      opaque = (1.0 - density_m) <= transmission_max
      gx = cv2.Sobel(tir, cv2.CV_32F, 1, 0, ksize=3)
      gy = cv2.Sobel(tir, cv2.CV_32F, 0, 1, ksize=3)
      magnitude = cv2.magnitude(gx, gy)
      if float(magnitude.max()) <= 1e-8:
          structure = np.zeros_like(opaque, dtype=bool)
      else:
          scaled = np.uint8(np.clip(magnitude / magnitude.max(), 0, 1) * 255)
          _, thresholded = cv2.threshold(scaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
          structure = cv2.dilate(thresholded, np.ones((3, 3), np.uint8), iterations=1).astype(bool)
      return np.where(opaque & structure, 255, 0).astype(np.uint8)
  ```

  CLI 必须：读取四个等级的 16-bit `Transmission_Map_GT` 和对应 TIR，使用数据加载器同样的 TIR 归一化规则，保存同 stem 的单通道 PNG；缺少配对文件时失败；默认不覆盖已有标签，`--overwrite` 才允许覆盖。

- [ ] **Step 4: 运行单元测试**

  Run: `D:\anaconda\envs\CoA\python.exe -m pytest tests\test_auto_route_gt.py -q`  
  Expected: PASS。

- [ ] **Step 5: 生成 train/test 标签并输出审核图**

  Run:

  ```powershell
  D:\anaconda\envs\CoA\python.exe tools\generate_auto_route_gt.py --data-root F:\1_paper_pan\1_Dehaze_Paper\2_Dataset\1_main_benchmark\1_FLIR --splits train test --haze-levels mist middle dense local_extreme
  D:\anaconda\envs\CoA\python.exe tools\visualize_auto_route_gt.py --data-root F:\1_paper_pan\1_Dehaze_Paper\2_Dataset\1_main_benchmark\1_FLIR --split train --haze-level local_extreme --count 12 --output-dir F:\1_paper_pan\1_Dehaze_Paper\2_Dataset\1_main_benchmark\1_FLIR\route_gt_debug
  ```

  Expected: 每个等级的标签数与对应 hazy 图数一致；审核图同时显示含雾 RGB、TIR、`M_GT` 和红色 `R_GT` 覆盖层。

- [ ] **Step 6: 人工审阅审核图后再进入 Task 2**

  验收规则：极浓雾中遮蔽轮廓的区域为红色；低雾道路、仍有颜色纹理的区域不为红色；平坦天空不为红色。若不满足，先只调整标签生成器，不改模型。

### Task 2: 新建 V2 五元组数据加载器

**Files:**
- Create: `data/route_supervision_dataset.py`
- Create: `tests/test_route_supervision_dataset.py`

- [ ] **Step 1: 写出数据配对与几何一致性的失败测试**

  测试必须验证以下接口：

  ```python
  dataset = RouteSupervisionDataset(root, train=True, size=8, haze_levels=("local_extreme",))
  hazy, clear, tir, density, route_gt = dataset[0]
  assert hazy.shape == clear.shape == tir.shape == (3, 8, 8)
  assert density.shape == route_gt.shape == (1, 8, 8)
  assert set(torch.unique(route_gt).tolist()) <= {0.0, 1.0}
  ```

  另写一项测试：水平翻转和旋转后，二值 `route_gt` 只能使用 nearest 插值，不能产生 `0<value<1`。

- [ ] **Step 2: 运行测试，确认模块不存在**

  Run: `D:\anaconda\envs\CoA\python.exe -m pytest tests\test_route_supervision_dataset.py -q`  
  Expected: FAIL，原因是 `RouteSupervisionDataset` 尚不存在。

- [ ] **Step 3: 实现独立加载器**

  `RouteSupervisionDataset` 不修改 `data/data_loader.py`，但可复用其中只读的图像/TIR 标量加载函数。它必须：

  ```python
  DEFAULT_V2_HAZE_LEVELS = ("mist", "middle", "dense", "local_extreme")
  ROUTE_DIRNAME = "Auto_Route_GT"
  DENSITY_GT_SEMANTICS = "density"
  ```

  读取 `Auto_Route_GT/<level>/<stem>.png`，把 `0/255` 转成 `[1,H,W]` 的 `0/1` float。RGB/TIR/density 用 bilinear 同步缩放，route 标签用 nearest 同步缩放、裁剪、翻转和旋转。缺少 route 标签或尺寸不对时抛出含路径的异常。

- [ ] **Step 4: 运行数据集测试**

  Run: `D:\anaconda\envs\CoA\python.exe -m pytest tests\test_route_supervision_dataset.py -q`  
  Expected: PASS。

- [ ] **Step 5: 对真实 FLIR 数据做只读加载核查**

  Run: `D:\anaconda\envs\CoA\python.exe -c "from data.route_supervision_dataset import RouteSupervisionDataset; d=RouteSupervisionDataset(r'F:\1_paper_pan\1_Dehaze_Paper\2_Dataset\1_main_benchmark\1_FLIR\train', train=False, size='full'); print(len(d), d.level_counts, [tuple(x.shape) for x in d[0]])"`  
  Expected: 四个 haze level 均有样本，且第五个 tensor 为二值路由标签。

### Task 3: 新建信息可用性路由器与 V2 模型

**Files:**
- Create: `model/availability_router.py`
- Create: `model/fog_routed_rgb_tir_v2.py`
- Create: `tests/test_availability_router.py`
- Create: `tests/test_fog_routed_v2.py`

- [ ] **Step 1: 写出路由器的失败测试**

  `AvailabilityRouter` 的测试必须构造固定特征并验证：高 `M` 本身不能保证补全；RGB 可用或 TIR 无结构时 `r_soft` 下降；输出尺寸和梯度正确。

  ```python
  router = AvailabilityRouter(channels=16)
  result = router(density, rgb_feature, tir_feature, structural_difference)
  assert set(result) == {"route_logits", "route_soft", "route_hard", "p_fog", "p_rgb_fail", "p_tir_struct"}
  assert result["route_soft"].shape == density.shape
  assert torch.all((result["route_soft"] >= 0) & (result["route_soft"] <= 1))
  ```

- [ ] **Step 2: 运行测试，确认模块不存在**

  Run: `D:\anaconda\envs\CoA\python.exe -m pytest tests\test_availability_router.py tests\test_fog_routed_v2.py -q`  
  Expected: FAIL，原因是 V2 模块尚不存在。

- [ ] **Step 3: 实现路由器**

  `AvailabilityRouter` 接收同空间尺度的 `density`、纯 RGB 编码特征、TIR 结构特征和方向性差异特征，分别输出 `p_fog`、`p_rgb_fail`、`p_tir_struct`，再产生：

  ```python
  route_soft = p_fog * p_rgb_fail * p_tir_struct
  route_hard = (route_soft >= 0.5).to(route_soft.dtype)
  route_hard = route_hard.detach() - route_soft.detach() + route_soft
  ```

  `p_fog` 的子网只输入 density，并以非负权重实现对 `M` 单调不减；另外两项可读取多模态特征。禁止把绝对 RGB--TIR 差异单独作为补全判据。

- [ ] **Step 4: 实现独立 V2 模型**

  `FogRoutedRGBTIRDehazerV2` 必须复用 `model.hde.HDE`，但不得改动该文件。模型输出：

  ```python
  {"pred_clear", "density_map", "route_logits", "route_soft", "route_hard",
   "p_fog", "p_rgb_fail", "p_tir_struct"}
  ```

  模型提供 `route_override` 参数，仅在训练使用。`route_override=route_gt` 时，融合分支只处理 `1-route_gt`，补全分支只处理 `route_gt`；补全分支的输入图中不得存在 RGB 特征、RGB 残差或 RGB 跳连。两个分支的特征送入一个共享 RGB 解码器，禁止输出两张图后再做像素拼接。

- [ ] **Step 5: 运行模型测试**

  Run: `D:\anaconda\envs\CoA\python.exe -m pytest tests\test_availability_router.py tests\test_fog_routed_v2.py -q`  
  Expected: PASS，包含 CPU 前向、反向、教师路由和“全补全时 RGB 特征无梯度”检查。

### Task 4: 新建 V2 损失、训练入口与配置

**Files:**
- Create: `loss/fog_routed_v2_loss.py`
- Create: `training/fog_routed_v2_source.py`
- Create: `train_fog_routed_v2.py`
- Create: `Teacher_RouteV2/config.json`
- Create: `tests/test_fog_routed_v2_loss.py`
- Create: `tests/test_fog_routed_v2_source_step.py`

- [ ] **Step 1: 写出三项损失的失败测试**

  ```python
  result = compute_fog_routed_v2_loss(pred_clear, clear, density_map, density_gt, route_soft, route_gt)
  assert set(result) == {"total", "reconstruction", "density", "route"}
  assert result["total"] == result["reconstruction"] + result["density"] + result["route"]
  ```

  再验证当 `route_soft == route_gt` 时 route loss 比相反预测更小。

- [ ] **Step 2: 运行测试，确认模块不存在**

  Run: `D:\anaconda\envs\CoA\python.exe -m pytest tests\test_fog_routed_v2_loss.py tests\test_fog_routed_v2_source_step.py -q`  
  Expected: FAIL，原因是 V2 loss/source step 尚不存在。

- [ ] **Step 3: 实现唯一的源域目标**

  `compute_fog_routed_v2_loss` 只计算：

  ```python
  L_rec = L1(pred_clear, clear_rgb)
  L_density = SmoothL1(density_map, density_gt)
  L_route = BCE(route_soft, route_gt)
  L_total = L_rec + lambda_density * L_density + lambda_route * L_route
  ```

  不实现旧的 `OmegaSampler`、`compute_q`、counterfactual 前向、boundary loss 或 binary loss。

- [ ] **Step 4: 实现 V2 训练入口**

  `train_fog_routed_v2.py` 只调用新数据集、新模型和新损失。配置文件必须显式包含：

  ```json
  {
    "density_gt_semantics": "density",
    "haze_levels": ["mist", "middle", "dense", "local_extreme"],
    "lambda_density": 1.0,
    "lambda_route": 1.0
  }
  ```

  每个训练 batch 使用 `route_override=route_gt`；每次验证/推理使用预测 hard route。checkpoint 格式单独标记为 `fog_routed_v2`，禁止使用旧 checkpoint 读取器加载。

- [ ] **Step 5: 运行 CPU 单步训练测试**

  Run: `D:\anaconda\envs\CoA\python.exe -m pytest tests\test_fog_routed_v2_loss.py tests\test_fog_routed_v2_source_step.py -q`  
  Expected: PASS，三项损失有限，HDE、路由器、融合分支和补全分支均有有限梯度。

### Task 5: 新建 V2 单图测试与可视化验收

**Files:**
- Create: `test_single_pair_route_v2.py`
- Create: `utils/visualize_route_v2.py`
- Create: `tests/test_single_pair_route_v2.py`

- [ ] **Step 1: 写出输出契约的失败测试**

  测试断言单图脚本需要的输出路径和模型字段存在：

  ```python
  assert {"pred_clear", "density_map", "route_soft", "route_hard"} <= output.keys()
  assert torch.equal(output["route_hard"], (output["route_soft"] >= 0.5).to(output["route_soft"].dtype))
  ```

- [ ] **Step 2: 运行测试，确认脚本不存在**

  Run: `D:\anaconda\envs\CoA\python.exe -m pytest tests\test_single_pair_route_v2.py -q`  
  Expected: FAIL，原因是 V2 单图脚本尚不存在。

- [ ] **Step 3: 实现 V2 单图测试脚本**

  脚本参数为 `--checkpoint --hazy --tir --output-dir --device`；不接受 clear RGB、density GT 或 route GT 输入。保存：

  ```text
  <stem>_dehazed.png
  <stem>_density_map.png
  <stem>_route_soft.png
  <stem>_route_hard.png
  <stem>_diagnostic.png
  ```

  `diagnostic.png` 按顺序展示含雾 RGB、TIR、`M`、`r_soft`、红色硬补全覆盖层和最终去雾图。

- [ ] **Step 4: 运行脚本与测试**

  Run: `D:\anaconda\envs\CoA\python.exe -m pytest tests\test_single_pair_route_v2.py -q`  
  Expected: PASS。训练得到首个 V2 checkpoint 后，使用用户指定 FLIR pair 运行新脚本，人工核验路由区域是否符合方案第 1 节。

## 验收顺序

1. 先确认 Task 1 的 `R_GT` 可视化正确；若不正确，不进行模型代码。
2. 再确认 Task 2 的五元组加载、数据语义和四个 haze level 正确。
3. 完成 Task 3--4 后先跑 CPU 单步和小规模训练，检查路由正负样本比例、分支梯度和 `r_soft` 分布。
4. 最后才进行完整训练与单图/批量 FLIR 测试。
