from .SSIM import SSIM
from .Feature_alignment import FA, FeatureAlignmentLoss
from .cr import ContrastLoss
from .KL import *
# 添加了下面两行
from .mssim import *
from .Dice import *
# --- [新增] 导入感知损失 (用于风格损失) ---
from .perceptual import PerceptualLoss
# --- [新增结束] ---