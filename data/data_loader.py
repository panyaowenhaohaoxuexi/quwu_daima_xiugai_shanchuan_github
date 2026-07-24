"""
这段Python代码定义了一套用于加载图像去雾（Image Dehazing）数据的PyTorch Dataset 类，它们是数据加载器（DataLoader）的蓝图。
这段代码的核心是一个名为 preprocess_feature 的辅助函数，它负责将输入的有雾图像转换为Tensor，并使用CLIP模型特定的均值和标准差进行归一化，这表明有雾图像将被用于与CLIP相关的计算。
代码文件主要包含两类数据集：第一类是用于有监督训练和评估的。
RESIDE_Dataset 和 RESIDE_Dataset_2 都是为加载成对的（有雾图像, 清晰图像）数据而设计的，它们在训练时会进行随机裁剪、翻转和旋转等数据增强。
它们之间的唯一区别在于匹配有雾图像和清晰图像的文件名逻辑。TestDataset 也加载成对数据，但主要用于评估，因此只进行裁剪而不进行随机增强，并会额外返回图像文件名。
第二类是 CLIP_loader，这是一个无监督的数据集。它只加载有雾图像（例如来自 real_foggy 目录），而不加载任何对应的清晰图像。
这个类对有雾图像进行数据增强和CLIP预处理后，只返回处理后的有雾图像。
这种数据加载器是为那些不需要成对清晰图像的训练方法而设计的，例如使用CLIP损失进行语义引导或使用EMA（指数移动平均）教师模型进行自我训练的场景。
"""

import os
import random
import hashlib
from typing import Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F
from torch.utils.data.dataloader import default_collate
from PIL import Image
from torchvision.transforms import Normalize, ToTensor, RandomCrop, RandomHorizontalFlip, Resize
from torchvision.transforms import functional as FF  # 导入 torchvision 的 functional 接口，用于更灵活的变换
from .stateful_sampler import manifest_fingerprint


COMMON_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
DEFAULT_HAZE_LEVELS = ("mist", "middle", "dense")


def _normalization_bounds(normalization, fixed_min=None, fixed_max=None,
                          calibrated_min=None, calibrated_max=None):
    if normalization == "fixed_range":
        lower, upper = fixed_min, fixed_max
    elif normalization == "dataset_calibrated_range":
        lower, upper = calibrated_min, calibrated_max
    else:
        return None
    if lower is None or upper is None or not float(upper) > float(lower):
        raise ValueError("normalization range requires max > min")
    return float(lower), float(upper)


def _array_to_unit_float(array, *, path, normalization="dtype_range", fixed_min=None,
                         fixed_max=None, calibrated_min=None, calibrated_max=None,
                         permit_int32_dtype_range=False, known_integer_bits=None):
    """Convert a scalar or image array to finite float32 values in [0, 1]."""
    array = np.asarray(array)
    if array.size == 0:
        raise ValueError(f"empty image: {path}")
    if not np.isfinite(array).all():
        raise ValueError(f"non-finite values in {path}")
    if normalization not in ("dtype_range", "fixed_range", "dataset_calibrated_range"):
        raise ValueError(f"unsupported normalization={normalization!r}")

    if normalization == "dtype_range":
        if np.issubdtype(array.dtype, np.uint8):
            output = array.astype(np.float32) / 255.0
        elif np.issubdtype(array.dtype, np.uint16):
            output = array.astype(np.float32) / 65535.0
        elif np.issubdtype(array.dtype, np.signedinteger):
            if known_integer_bits == 16 and array.min() >= 0 and array.max() <= 65535:
                output = array.astype(np.float32) / 65535.0
            elif not permit_int32_dtype_range:
                raise ValueError(
                    f"integer mode with unknown physical range in {path}; use fixed_range"
                )
            else:
                info = np.iinfo(array.dtype)
                output = (array.astype(np.float32) - info.min) / float(info.max - info.min)
        elif np.issubdtype(array.dtype, np.floating):
            output = array.astype(np.float32)
            if output.min() < 0.0 or output.max() > 1.0:
                raise ValueError(f"float image outside [0,1] in {path}; use fixed_range")
        else:
            raise ValueError(f"unsupported image dtype {array.dtype} in {path}")
    else:
        lower, upper = _normalization_bounds(
            normalization, fixed_min, fixed_max, calibrated_min, calibrated_max
        )
        output = (array.astype(np.float32) - lower) / (upper - lower)
    return torch.from_numpy(np.ascontiguousarray(output)).float().clamp(0.0, 1.0)


def _open_preserving_known_bit_depth(path):
    """Pillow exposes some 16-bit PNGs as mode I/int32; recover only known metadata."""
    with Image.open(path) as image:
        array = np.asarray(image)
        known_bits = None
        if image.format == "PNG":
            with open(path, "rb") as handle:
                header = handle.read(25)
            if header[:8] == b"\x89PNG\r\n\x1a\n" and len(header) >= 25 and header[24] == 16:
                known_bits = 16
        elif image.format == "TIFF":
            bits = image.tag_v2.get(258)
            if isinstance(bits, tuple):
                bits = bits[0]
            if bits == 16:
                known_bits = 16
        return image.mode, array, known_bits


def load_scalar_map_as_float_tensor(path, normalization="dtype_range", *, fixed_min=None,
                                    fixed_max=None, calibrated_min=None, calibrated_max=None):
    """Load a scalar density/transmission map without lossy PIL conversion."""
    _, array, known_bits = _open_preserving_known_bit_depth(path)
    if array.ndim != 2:
        raise ValueError(f"scalar map must be single channel: {path}, shape={array.shape}")
    tensor = _array_to_unit_float(
        array, path=path, normalization=normalization, fixed_min=fixed_min,
        fixed_max=fixed_max, calibrated_min=calibrated_min, calibrated_max=calibrated_max,
        known_integer_bits=known_bits,
    )
    return tensor.unsqueeze(0)


def convert_density_semantics(raw_map, density_gt_semantics="transmission"):
    if density_gt_semantics not in ("transmission", "density"):
        raise ValueError(f"unsupported density_gt_semantics={density_gt_semantics!r}")
    return (1.0 - raw_map if density_gt_semantics == "transmission" else raw_map).clamp(0.0, 1.0)


def load_tir_as_float_tensor(path, normalization_config: Optional[Mapping] = None):
    """Load TIR preserving its original precision and expose a replicated 3-channel tensor."""
    config = dict(normalization_config or {})
    normalization = config.get("normalization", "dtype_range")
    mode, array, known_bits = _open_preserving_known_bit_depth(path)
    if array.ndim == 2:
        one_channel = _array_to_unit_float(
            array, path=path, normalization=normalization,
            fixed_min=config.get("fixed_min"), fixed_max=config.get("fixed_max"),
            calibrated_min=config.get("calibrated_min"), calibrated_max=config.get("calibrated_max"),
            known_integer_bits=known_bits,
        ).unsqueeze(0)
    elif array.ndim == 3 and array.shape[2] in (3, 4):
        rgb = array[..., :3]
        if np.issubdtype(rgb.dtype, np.integer):
            differences = (int(np.abs(rgb[..., 0].astype(np.int64) - rgb[..., 1].astype(np.int64)).max()),
                           int(np.abs(rgb[..., 0].astype(np.int64) - rgb[..., 2].astype(np.int64)).max()),
                           int(np.abs(rgb[..., 1].astype(np.int64) - rgb[..., 2].astype(np.int64)).max()))
            tolerance = int(config.get("channel_tolerance_code_values", 1))
        else:
            normalized = _array_to_unit_float(
                rgb, path=path, normalization=normalization,
                fixed_min=config.get("fixed_min"), fixed_max=config.get("fixed_max"),
                calibrated_min=config.get("calibrated_min"), calibrated_max=config.get("calibrated_max"),
                known_integer_bits=known_bits,
            )
            differences = (float((normalized[..., 0] - normalized[..., 1]).abs().max()),
                           float((normalized[..., 0] - normalized[..., 2]).abs().max()),
                           float((normalized[..., 1] - normalized[..., 2]).abs().max()))
            tolerance = float(config.get("channel_tolerance_float", 1e-5))
        if max(differences) > tolerance:
            raise ValueError(
                f"TIR channel tolerance exceeded: path={path}, mode={mode}, dtype={rgb.dtype}, "
                f"max_diffs={differences}, tolerance={tolerance}"
            )
        rgb_float = _array_to_unit_float(
            rgb, path=path, normalization=normalization,
            fixed_min=config.get("fixed_min"), fixed_max=config.get("fixed_max"),
            calibrated_min=config.get("calibrated_min"), calibrated_max=config.get("calibrated_max"),
            known_integer_bits=known_bits,
        )
        one_channel = rgb_float.mean(dim=2, keepdim=False).unsqueeze(0)
    else:
        raise ValueError(f"unsupported TIR shape in {path}: mode={mode}, shape={array.shape}")
    return one_channel.repeat(3, 1, 1).contiguous()


def stable_seed_mixer(global_augmentation_seed, sampler_epoch, sample_index, sample_occurrence=0):
    payload = "|".join(map(str, (global_augmentation_seed, sampler_epoch, sample_index, sample_occurrence)))
    return int.from_bytes(hashlib.blake2b(payload.encode("utf-8"), digest_size=8).digest(), "little")


def _is_auto_format(format):
    return format is None or str(format).lower() in ("auto", "*")


def _normalize_format(format):
    if _is_auto_format(format):
        return "auto"
    format = str(format).strip().lower()
    if not format.startswith("."):
        format = "." + format
    return format


def _is_image_file(filename, format="auto"):
    ext = os.path.splitext(filename)[1].lower()
    normalized = _normalize_format(format)
    if normalized == "auto":
        return ext in COMMON_IMAGE_EXTS
    return ext == normalized


def _format_description(format):
    normalized = _normalize_format(format)
    if normalized == "auto":
        return "auto (.jpg/.jpeg/.png/.bmp/.tif/.tiff)"
    return normalized


def _find_matching_image_by_stem(directory, image_name, format="auto"):
    if not directory:
        return None

    exact_path = os.path.join(directory, image_name)
    if os.path.isfile(exact_path):
        return exact_path

    if not _is_auto_format(format):
        return None

    stem, _ = os.path.splitext(image_name)
    for ext in COMMON_IMAGE_EXTS:
        candidate = os.path.join(directory, stem + ext)
        if os.path.isfile(candidate):
            return candidate

    lower_stem = stem.lower()
    matches = []
    try:
        for filename in os.listdir(directory):
            candidate_stem, candidate_ext = os.path.splitext(filename)
            if candidate_stem.lower() == lower_stem and candidate_ext.lower() in COMMON_IMAGE_EXTS:
                matches.append(filename)
    except FileNotFoundError:
        return None

    if not matches:
        return None

    priority = {ext: index for index, ext in enumerate(COMMON_IMAGE_EXTS)}
    matches.sort(key=lambda name: (priority.get(os.path.splitext(name)[1].lower(), 999), name.lower()))
    return os.path.join(directory, matches[0])


def _make_output_name(image_name, haze_level):
    return f"{haze_level}_{image_name}" if haze_level else image_name


def _list_hazy_files(directory, format):
    try:
        return sorted(
            filename for filename in os.listdir(directory)
            if os.path.isfile(os.path.join(directory, filename)) and _is_image_file(filename, format)
        )
    except FileNotFoundError:
        return []


def _build_multimodal_samples(
    hazy_visible_path,
    infrared_path,
    clear_visible_path=None,
    format="auto",
    haze_levels=None,
    require_clear=True,
    dataset_name="MultiModalHazeDataset",
):
    haze_levels = list(haze_levels or DEFAULT_HAZE_LEVELS)
    direct_images = _list_hazy_files(hazy_visible_path, format)
    direct_count = len(direct_images)
    level_dirs = [
        level for level in haze_levels
        if os.path.isdir(os.path.join(hazy_visible_path, level))
    ]

    if direct_images:
        layout = "flat"
        if level_dirs:
            print(f"[{dataset_name}] WARNING: both direct images and haze-level folders were found under hazy/. Using flat layout.")
        hazy_entries = [(None, hazy_visible_path, direct_images)]
    elif level_dirs:
        layout = "multi_level"
        hazy_entries = [
            (level, os.path.join(hazy_visible_path, level), _list_hazy_files(os.path.join(hazy_visible_path, level), format))
            for level in haze_levels
            if os.path.isdir(os.path.join(hazy_visible_path, level))
        ]
    else:
        layout = "empty"
        hazy_entries = []

    samples = []
    level_counts = {level: 0 for level in haze_levels}
    missing_ir = 0
    missing_clear = 0

    for haze_level, hazy_dir, image_names in hazy_entries:
        for image_name in image_names:
            ir_path = _find_matching_image_by_stem(infrared_path, image_name, format)
            if ir_path is None:
                missing_ir += 1
                continue

            clear_path = None
            if require_clear:
                clear_path = _find_matching_image_by_stem(clear_visible_path, image_name, format)
                if clear_path is None:
                    missing_clear += 1
                    continue

            samples.append({
                "image_name": image_name,
                "output_name": _make_output_name(image_name, haze_level),
                "hazy_path": os.path.join(hazy_dir, image_name),
                "ir_path": ir_path,
                "clear_path": clear_path,
                "haze_level": haze_level,
            })
            if haze_level:
                level_counts[haze_level] += 1

    result = {
        "samples": samples,
        "layout": layout,
        "level_counts": level_counts,
        "missing_ir": missing_ir,
        "missing_clear": missing_clear,
        "direct_count": direct_count,
    }

    _print_sample_stats(dataset_name, result, format)

    if not samples:
        raise RuntimeError(
            f"[{dataset_name}] no valid samples after pairing. "
            f"hazy_visible_path={hazy_visible_path}, infrared_path={infrared_path}, "
            f"clear_visible_path={clear_visible_path}, layout={layout}, "
            f"format={_format_description(format)}, missing_ir={missing_ir}, missing_clear={missing_clear}"
        )

    return result


def _print_sample_stats(dataset_name, result, format):
    print(f"[{dataset_name}] format: {_format_description(format)}")
    print(f"[{dataset_name}] detected layout: {result['layout']}")
    if result["layout"] == "multi_level":
        counts = ", ".join(f"{level}={result['level_counts'].get(level, 0)}" for level in DEFAULT_HAZE_LEVELS)
        print(f"[{dataset_name}] haze levels: {counts}")
        print(f"[{dataset_name}] total samples: {len(result['samples'])}")
    else:
        print(f"[{dataset_name}] samples: {len(result['samples'])}")
    if result["missing_ir"] or result["missing_clear"]:
        print(f"[{dataset_name}] skipped missing pairs: missing_ir={result['missing_ir']}, missing_clear={result['missing_clear']}")


def _assign_sample_metadata(dataset, result):
    dataset.samples = result["samples"]
    dataset.layout = result["layout"]
    dataset.level_counts = result["level_counts"]
    dataset.missing_ir = result["missing_ir"]
    dataset.missing_clear = result["missing_clear"]
    dataset.direct_count = result["direct_count"]
    dataset.image_list = [sample["output_name"] for sample in dataset.samples]


def _sky_mask_candidates(sky_mask_path, image_name, sky_mask_suffix, sky_mask_ext, haze_level=None):
    stem, original_ext = os.path.splitext(image_name)
    ext_or_original = sky_mask_ext if sky_mask_ext else original_ext

    def scoped_candidates(base_dir):
        return [
            os.path.join(base_dir, stem + sky_mask_suffix + ext_or_original),
            os.path.join(base_dir, image_name),
            os.path.join(base_dir, stem + ".png"),
            os.path.join(base_dir, stem + ".jpg"),
            os.path.join(base_dir, stem + ".jpeg"),
            os.path.join(base_dir, stem + "_sky.png"),
            os.path.join(base_dir, stem + "_sky.jpg"),
            os.path.join(base_dir, stem + "_sky.jpeg"),
        ]

    candidates = []
    if haze_level:
        candidates.extend(scoped_candidates(os.path.join(sky_mask_path, haze_level)))
    candidates.extend(scoped_candidates(sky_mask_path))
    return candidates


def preprocess_feature(img):
    """
    对图像进行预处理，使其适用于CLIP模型的输入。
    1. 将PIL图像转换为Tensor (范围 [0, 1])。
    2. 使用CLIP特定的均值和标准差进行归一化。
    """
    img = ToTensor()(img)
    # CLIP模型专用的归一化参数
    clip_normalizer = Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    img = clip_normalizer(img)
    return img


class RESIDE_Dataset(data.Dataset):
    """
    用于加载 RESIDE (或类似格式) 数据集的数据加载器。
    它假定有雾图像和清晰图像是通过文件名中的ID（例如 '0001_1.png' 和 '0001.png'）来配对的。
    """

    def __init__(self, path, train, size=256, format='.png'):
        super(RESIDE_Dataset, self).__init__()
        self.size = size  # 裁剪的目标尺寸
        self.train = train  # 是否为训练模式 (决定是否进行数据增强)
        self.format = format  # 清晰图像的文件格式
        # 获取所有有雾图像的文件名列表
        self.haze_imgs_dir = os.listdir(os.path.join(path, 'hazy'))
        # 获取所有有雾图像的完整路径列表
        self.haze_imgs = [os.path.join(path, 'hazy', img) for img in self.haze_imgs_dir]
        # 清晰图像的目录路径
        self.clear_dir = os.path.join(path, 'clear')

    def __getitem__(self, index):
        # 1. 加载有雾图像
        haze = Image.open(self.haze_imgs[index])

        # 如果图像尺寸小于目标尺寸，则随机换一张图像 (防止RandomCrop失败)
        if isinstance(self.size, int):
            while haze.size[0] < self.size or haze.size[1] < self.size:
                index = random.randint(0, 100)
                haze = Image.open(self.haze_imgs[index])

        # 2. 根据有雾图像的文件名找到对应的清晰图像
        img = self.haze_imgs[index]
        split_name = os.path.split(img)[-1].split('_')  # 例如 '0001_1.png' -> ['0001', '1.png']
        id = split_name[0]  # 获取ID '0001'
        clear_name = id + self.format  # 构造清晰图像名 '0001.png'
        clear = Image.open(os.path.join(self.clear_dir, clear_name))  # 加载清晰图像

        # 3. 随机裁剪
        if not isinstance(self.size, str):
            # 获取随机裁剪参数 (保证haze和clear使用相同的裁剪位置)
            i, j, h, w = RandomCrop.get_params(haze, output_size=(self.size, self.size))
            haze = FF.crop(haze, i, j, h, w)
            clear = FF.crop(clear, i, j, h, w)

        # 4. 数据增强和预处理
        haze, clear = self.augData(haze.convert("RGB"), clear.convert("RGB"))
        return haze, clear

    def augData(self, data, target):
        """ 对 (data, target) 图像对进行数据增强和预处理 """
        if self.train:
            # 随机水平翻转
            rand_hor = random.randint(0, 1)
            # 随机旋转 (0, 90, 180, 270 度)
            rand_rot = random.randint(0, 3)
            data = RandomHorizontalFlip(rand_hor)(data)
            target = RandomHorizontalFlip(rand_hor)(target)
            if rand_rot:
                data = FF.rotate(data, 90 * rand_rot)
                target = FF.rotate(target, 90 * rand_rot)

        # 将目标图像(clear)转换为Tensor (范围 [0, 1])
        target = ToTensor()(target)
        # 将输入图像(haze)进行CLIP预处理 (ToTensor + Normalize)
        return preprocess_feature(data), target

    def __len__(self):
        # 返回数据集中有雾图像的总数
        return len(self.haze_imgs)


class RESIDE_Dataset_2(data.Dataset):
    """
    与 RESIDE_Dataset 类似的数据加载器，但文件名配对逻辑不同。
    它假设有雾图像和清晰图像的文件名主体完全相同，只是扩展名可能不同。
    例如 'image1.jpg' (hazy) 配对 'image1.jpg' (clear) 或 'image1.png' (clear)。
    """

    def __init__(self, path, train, size=256, format='.jpg'):
        super(RESIDE_Dataset_2, self).__init__()
        self.size = size
        self.train = train
        self.format = format
        self.haze_imgs_dir = os.listdir(os.path.join(path, 'hazy'))
        self.haze_imgs = [os.path.join(path, 'hazy', img) for img in self.haze_imgs_dir]
        self.clear_dir = os.path.join(path, 'clear')

    def __getitem__(self, index):
        haze = Image.open(self.haze_imgs[index])
        if isinstance(self.size, int):
            while haze.size[0] < self.size or haze.size[1] < self.size:
                index = random.randint(0, 100)
                haze = Image.open(self.haze_imgs[index])

        img = self.haze_imgs[index]
        split_name = os.path.split(img)[-1]  # 'image1.jpg'

        # --- 关键区别 ---
        id = os.path.splitext(split_name)[0]  # 'image1'
        # ----------------

        clear_name = f"{id}{self.format}"  # 'image1.jpg' (或 .png, 取决于 self.format)
        clear = Image.open(os.path.join(self.clear_dir, clear_name))

        if not isinstance(self.size, str):
            i, j, h, w = RandomCrop.get_params(haze, output_size=(self.size, self.size))
            haze = FF.crop(haze, i, j, h, w)
            clear = FF.crop(clear, i, j, h, w)

        haze, clear = self.augData(haze.convert("RGB"), clear.convert("RGB"))
        return haze, clear

    def augData(self, data, target):
        """ 数据增强和预处理 (与 RESIDE_Dataset 相同) """
        if self.train:
            rand_hor = random.randint(0, 1)
            rand_rot = random.randint(0, 3)
            data = RandomHorizontalFlip(rand_hor)(data)
            target = RandomHorizontalFlip(rand_hor)(target)
            if rand_rot:
                data = FF.rotate(data, 90 * rand_rot)
                target = FF.rotate(target, 90 * rand_rot)
        target = ToTensor()(target)
        return preprocess_feature(data), target

    def __len__(self):
        return len(self.haze_imgs)


# --- 修改后的 TestDataset 类 ---
class CLIP_loader(data.Dataset):
    """
    用于无监督训练的数据加载器 (例如用于EMA和CLIP损失的训练)。
    它 **只** 加载有雾图像，不加载清晰图像。
    """

    def __init__(self, hazy_path, train, size=256):
        self.hazy_path = hazy_path
        self.train = train
        self.hazy_image_list = os.listdir(hazy_path)
        self.hazy_image_list.sort()
        self.size = size

    def __getitem__(self, index):
        hazy_image_name = self.hazy_image_list[index]
        hazy_image_path = os.path.join(self.hazy_path, hazy_image_name)
        hazy = Image.open(hazy_image_path).convert('RGB')
        width, height = hazy.size
        # 裁剪尺寸为 (size, height, width) 中的最小值
        crop_size = min(self.size, height, width)

        # 随机裁剪
        if not isinstance(self.size, str):
            i, j, h, w = RandomCrop.get_params(hazy, output_size=(crop_size, crop_size))
            hazy = FF.crop(hazy, i, j, h, w)

        # 缩放到统一尺寸
        hazy = Resize((self.size, self.size))(hazy)

        # 数据增强和预处理
        hazy = self.augData(hazy.convert("RGB"))

        # 只返回有雾图像
        return hazy

    def augData(self, data):
        """ 只对单张图像进行数据增强和预处理 """
        if self.train:
            rand_hor = random.randint(0, 1)
            rand_rot = random.randint(0, 3)
            data = RandomHorizontalFlip(rand_hor)(data)
            if rand_rot:
                data = FF.rotate(data, 90 * rand_rot)
        # 进行CLIP预处理
        return preprocess_feature(data)

    def __len__(self):
        return len(self.hazy_image_list)

# --- 添加新的 MultiModalHazeDataset 类 ---
# Updated multimodal loaders. These definitions intentionally override the
# earlier compatibility versions in this file while preserving public class names.
class TestDataset(data.Dataset):
    def __init__(
        self,
        hazy_visible_path,
        infrared_path,
        clear_visible_path,
        size=256,
        format="auto",
        sky_mask_path=None,
        use_sky_mask=False,
        sky_mask_suffix="_sky",
        sky_mask_ext=".png",
        require_sky_mask=False,
    ):
        super(TestDataset, self).__init__()
        self.hazy_visible_path = hazy_visible_path
        self.infrared_path = infrared_path
        self.clear_visible_path = clear_visible_path
        self.size = size
        self.format = format
        self.sky_mask_path = sky_mask_path
        self.use_sky_mask = use_sky_mask
        self.sky_mask_suffix = sky_mask_suffix
        self.sky_mask_ext = sky_mask_ext
        self.require_sky_mask = require_sky_mask

        result = _build_multimodal_samples(
            hazy_visible_path=hazy_visible_path,
            infrared_path=infrared_path,
            clear_visible_path=clear_visible_path,
            format=format,
            require_clear=True,
            dataset_name="TestDataset",
        )
        _assign_sample_metadata(self, result)

        if self.use_sky_mask:
            print("[SkyMask][Test] enabled")
            print(f"[SkyMask][Test] sky_mask_path={self.sky_mask_path}")
            print(f"[SkyMask][Test] suffix={self.sky_mask_suffix}")
            print(f"[SkyMask][Test] ext={self.sky_mask_ext}")
            print(f"[SkyMask][Test] require={self.require_sky_mask}")
            if not self.sky_mask_path or not os.path.isdir(self.sky_mask_path):
                msg = f"[SkyMask][Test] sky mask directory not found: {self.sky_mask_path}"
                if self.require_sky_mask:
                    raise RuntimeError(msg)
                print(f"WARNING: {msg}. Missing masks will fall back to all-zero masks.")

    def find_sky_mask_path(self, image_name, haze_level=None):
        if not self.sky_mask_path:
            return None
        for path in _sky_mask_candidates(
            self.sky_mask_path,
            image_name,
            self.sky_mask_suffix,
            self.sky_mask_ext,
            haze_level,
        ):
            if path and os.path.exists(path):
                return path
        return None

    def __getitem__(self, index):
        sample = self.samples[index]
        image_name = sample["image_name"]
        output_name = sample["output_name"]

        try:
            hazy_vis = Image.open(sample["hazy_path"]).convert("RGB")
            infrared = Image.open(sample["ir_path"]).convert("RGB")
            clear_vis = Image.open(sample["clear_path"]).convert("RGB")

            sky_mask = None
            if self.use_sky_mask:
                mask_path = self.find_sky_mask_path(image_name, sample["haze_level"])
                if mask_path is not None:
                    sky_mask = Image.open(mask_path).convert("L")
                elif self.require_sky_mask:
                    raise FileNotFoundError(f"Sky mask not found for {image_name} in {self.sky_mask_path}")
                else:
                    sky_mask = Image.new("L", hazy_vis.size, 0)

            if isinstance(self.size, int):
                w, h = hazy_vis.size
                if w >= self.size and h >= self.size:
                    crop_size = (self.size, self.size)
                    hazy_vis = FF.center_crop(hazy_vis, crop_size)
                    infrared = FF.center_crop(infrared, crop_size)
                    clear_vis = FF.center_crop(clear_vis, crop_size)
                    if sky_mask is not None:
                        sky_mask = FF.center_crop(sky_mask, crop_size)
                else:
                    resize_size = [self.size, self.size]
                    hazy_vis = FF.resize(hazy_vis, resize_size, interpolation=FF.InterpolationMode.BILINEAR)
                    infrared = FF.resize(infrared, resize_size, interpolation=FF.InterpolationMode.BILINEAR)
                    clear_vis = FF.resize(clear_vis, resize_size, interpolation=FF.InterpolationMode.BILINEAR)
                    if sky_mask is not None:
                        sky_mask = FF.resize(sky_mask, resize_size, interpolation=FF.InterpolationMode.NEAREST)
                    print(f"Warning: image {output_name} is smaller than {self.size}x{self.size}; resized.")

            hazy_vis_tensor = preprocess_feature(hazy_vis)
            infrared_tensor = preprocess_feature(infrared)
            clear_vis_tensor = ToTensor()(clear_vis)

            if self.use_sky_mask:
                sky_mask_tensor = ToTensor()(sky_mask)
                sky_mask_tensor = (sky_mask_tensor >= 0.5).float()
                return hazy_vis_tensor, infrared_tensor, clear_vis_tensor, sky_mask_tensor, output_name

            return hazy_vis_tensor, infrared_tensor, clear_vis_tensor, output_name

        except FileNotFoundError as e:
            print(f"Error loading image: {e}. Skipping index {index}.")
            if self.use_sky_mask and self.require_sky_mask:
                raise
            if self.use_sky_mask:
                return None, None, None, None, None
            return None, None, None, None
        except Exception as e:
            print(f"An unexpected error occurred at index {index} ({output_name}): {e}")
            if self.use_sky_mask and self.require_sky_mask:
                raise
            if self.use_sky_mask:
                return None, None, None, None, None
            return None, None, None, None

    def __len__(self):
        return len(self.samples)


class MultiModalHazeDataset(data.Dataset):
    def __init__(
        self,
        hazy_visible_path,
        infrared_path,
        clear_visible_path,
        train,
        size=256,
        format="auto",
        sky_mask_path="",
        use_sky_mask=False,
        sky_mask_suffix="_sky",
        sky_mask_ext=".png",
        require_sky_mask=False,
    ):
        super(MultiModalHazeDataset, self).__init__()
        self.size = size
        self.train = train
        self.format = format
        self.hazy_visible_path = hazy_visible_path
        self.infrared_path = infrared_path
        self.clear_visible_path = clear_visible_path
        self.sky_mask_path = sky_mask_path
        self.use_sky_mask = use_sky_mask
        self.sky_mask_suffix = sky_mask_suffix
        self.sky_mask_ext = sky_mask_ext
        self.require_sky_mask = require_sky_mask

        result = _build_multimodal_samples(
            hazy_visible_path=hazy_visible_path,
            infrared_path=infrared_path,
            clear_visible_path=clear_visible_path,
            format=format,
            require_clear=True,
            dataset_name="MultiModalHazeDataset",
        )
        _assign_sample_metadata(self, result)

        if self.use_sky_mask:
            print("[SkyMask] enabled")
            print(f"[SkyMask] sky_mask_path={self.sky_mask_path}")
            print(f"[SkyMask] suffix={self.sky_mask_suffix}")
            print(f"[SkyMask] ext={self.sky_mask_ext}")
            print(f"[SkyMask] require={self.require_sky_mask}")
            if not self.sky_mask_path or not os.path.isdir(self.sky_mask_path):
                msg = f"[SkyMask] sky mask directory not found: {self.sky_mask_path}"
                if self.require_sky_mask:
                    raise RuntimeError(msg)
                print(f"WARNING: {msg}. Missing masks will fall back to all-zero masks.")
            sample_check = self.samples[:20]
            found = sum(
                1 for sample in sample_check
                if self.find_sky_mask_path(sample["image_name"], sample["haze_level"]) is not None
            )
            print(f"[SkyMask] sample check: found {found} / {len(sample_check)}")

    def find_sky_mask_path(self, image_name, haze_level=None):
        if not self.sky_mask_path:
            return None
        for path in _sky_mask_candidates(
            self.sky_mask_path,
            image_name,
            self.sky_mask_suffix,
            self.sky_mask_ext,
            haze_level,
        ):
            if path and os.path.exists(path):
                return path
        return None

    def __getitem__(self, index):
        sample = self.samples[index]
        image_name = sample["image_name"]

        try:
            hazy_vis = Image.open(sample["hazy_path"]).convert("RGB")
            infrared = Image.open(sample["ir_path"]).convert("RGB")
            clear_vis = Image.open(sample["clear_path"]).convert("RGB")

            sky_mask = None
            if self.use_sky_mask:
                mask_path = self.find_sky_mask_path(image_name, sample["haze_level"])
                if mask_path is not None:
                    sky_mask = Image.open(mask_path).convert("L")
                elif self.require_sky_mask:
                    raise FileNotFoundError(f"Sky mask not found for {image_name} in {self.sky_mask_path}")
                else:
                    sky_mask = Image.new("L", hazy_vis.size, 0)

            if isinstance(self.size, int):
                min_h = min(hazy_vis.size[1], infrared.size[1], clear_vis.size[1])
                min_w = min(hazy_vis.size[0], infrared.size[0], clear_vis.size[0])
                if min_w < self.size or min_h < self.size:
                    target_h = max(self.size, min_h)
                    target_w = max(self.size, min_w)
                    hazy_vis = hazy_vis.resize((target_w, target_h), Image.BILINEAR)
                    infrared = infrared.resize((target_w, target_h), Image.BILINEAR)
                    clear_vis = clear_vis.resize((target_w, target_h), Image.BILINEAR)
                    if sky_mask is not None:
                        sky_mask = sky_mask.resize((target_w, target_h), Image.NEAREST)

                i, j, h, w = RandomCrop.get_params(hazy_vis, output_size=(self.size, self.size))
                hazy_vis = FF.crop(hazy_vis, i, j, h, w)
                infrared = FF.crop(infrared, i, j, h, w)
                clear_vis = FF.crop(clear_vis, i, j, h, w)
                if sky_mask is not None:
                    sky_mask = FF.crop(sky_mask, i, j, h, w)

            return self.aug_and_preprocess(hazy_vis, infrared, clear_vis, sky_mask)

        except FileNotFoundError as e:
            print(f"Error loading image: {e}. Skipping index {index}.")
            if self.use_sky_mask and self.require_sky_mask:
                raise
            if self.use_sky_mask:
                return None, None, None, None
            return None, None, None
        except Exception as e:
            print(f"An unexpected error occurred at index {index} ({sample['output_name']}): {e}")
            if self.use_sky_mask:
                return None, None, None, None
            return None, None, None

    def aug_and_preprocess(self, hazy_vis, infrared, clear_vis, sky_mask=None):
        if self.train:
            rand_hor = random.randint(0, 1)
            if rand_hor == 1:
                hazy_vis = FF.hflip(hazy_vis)
                infrared = FF.hflip(infrared)
                clear_vis = FF.hflip(clear_vis)
                if sky_mask is not None:
                    sky_mask = FF.hflip(sky_mask)

            rand_rot = random.randint(0, 3)
            if rand_rot > 0:
                hazy_vis = FF.rotate(hazy_vis, 90 * rand_rot)
                infrared = FF.rotate(infrared, 90 * rand_rot)
                clear_vis = FF.rotate(clear_vis, 90 * rand_rot)
                if sky_mask is not None:
                    sky_mask = FF.rotate(sky_mask, 90 * rand_rot)

        hazy_vis_processed = preprocess_feature(hazy_vis)
        infrared_processed = preprocess_feature(infrared)
        clear_vis_processed = ToTensor()(clear_vis)

        if sky_mask is not None:
            sky_mask_tensor = ToTensor()(sky_mask)
            sky_mask_tensor = (sky_mask_tensor >= 0.5).float()
            return hazy_vis_processed, infrared_processed, clear_vis_processed, sky_mask_tensor

        return hazy_vis_processed, infrared_processed, clear_vis_processed

    def __len__(self):
        return len(self.samples)


SYNTH_IMAGE_EXTS = (".png", ".jpg", ".jpeg")


def _find_synth_file_by_stem(directory, stem):
    """Find one file by stem using .png > .jpg > .jpeg priority."""
    found = []
    lower_stem = stem.lower()
    try:
        for filename in os.listdir(directory):
            candidate_stem, candidate_ext = os.path.splitext(filename)
            if candidate_stem.lower() == lower_stem and candidate_ext.lower() in SYNTH_IMAGE_EXTS:
                found.append(filename)
    except FileNotFoundError:
        return None, False

    if not found:
        return None, False

    priority = {ext: index for index, ext in enumerate(SYNTH_IMAGE_EXTS)}
    found.sort(key=lambda name: (priority[os.path.splitext(name)[1].lower()], name.lower()))
    return os.path.join(directory, found[0]), len(found) > 1


def _build_synth_stem_index(directory):
    """Build a stem -> sorted candidate path list index for fast pairing."""
    index = {}
    priority = {ext: rank for rank, ext in enumerate(SYNTH_IMAGE_EXTS)}
    try:
        for filename in os.listdir(directory):
            stem, ext = os.path.splitext(filename)
            ext = ext.lower()
            if ext not in SYNTH_IMAGE_EXTS:
                continue
            key = stem.lower()
            index.setdefault(key, []).append(os.path.join(directory, filename))
    except FileNotFoundError:
        return index

    for key, paths in index.items():
        paths.sort(key=lambda path: (priority[os.path.splitext(path)[1].lower()], os.path.basename(path).lower()))
    return index


def _lookup_synth_stem(index, stem):
    paths = index.get(stem.lower(), [])
    if not paths:
        return None, False
    return paths[0], len(paths) > 1


class SynthMultiModalDataset(data.Dataset):
    """Formal synthetic RGB--TIR dataset returning (hazy, clear, tir, density)."""

    def __init__(self, root, train=True, size=256, haze_levels=None,
                 density_gt_semantics="transmission", density_map_normalization="dtype_range",
                 density_fixed_min=None, density_fixed_max=None,
                 density_calibrated_min=None, density_calibrated_max=None,
                 tir_normalization_config=None, pair_alignment_policy="strict",
                 augmentation_seed_base=0, sampler_epoch=0):
        super(SynthMultiModalDataset, self).__init__()
        self.root = root
        self.train = train
        self.size = size
        self.haze_levels = tuple(haze_levels or DEFAULT_HAZE_LEVELS)
        self.density_gt_semantics = density_gt_semantics
        self.density_map_normalization = density_map_normalization
        self.density_normalization_kwargs = {
            "fixed_min": density_fixed_min, "fixed_max": density_fixed_max,
            "calibrated_min": density_calibrated_min, "calibrated_max": density_calibrated_max,
        }
        self.tir_normalization_config = dict(tir_normalization_config or {})
        self.pair_alignment_policy = pair_alignment_policy
        self.augmentation_seed_base = int(augmentation_seed_base)
        self.sampler_epoch = int(sampler_epoch)
        if pair_alignment_policy not in ("strict", "resize_tir_to_rgb"):
            raise ValueError("pair_alignment_policy must be 'strict' or 'resize_tir_to_rgb'")

        self.clear_dir = os.path.join(root, "clear")
        self.hazy_root = os.path.join(root, "hazy")
        self.ir_dir = os.path.join(root, "ir")
        self.density_root = os.path.join(root, "Transmission_Map_GT")

        self.samples = []
        self.level_counts = {level: 0 for level in self.haze_levels}
        self.missing_counts = {"clear": 0, "ir": 0, "density": 0}
        self.ambiguous_counts = {"clear": 0, "ir": 0, "density": 0}
        self._build_samples()

    def _build_samples(self):
        clear_index = _build_synth_stem_index(self.clear_dir)
        ir_index = _build_synth_stem_index(self.ir_dir)
        density_indexes = {
            level: _build_synth_stem_index(os.path.join(self.density_root, level))
            for level in self.haze_levels
        }
        for level in self.haze_levels:
            hazy_dir = os.path.join(self.hazy_root, level)

            for filename in _list_hazy_files(hazy_dir, "auto"):
                ext = os.path.splitext(filename)[1].lower()
                if ext not in SYNTH_IMAGE_EXTS:
                    continue

                stem = os.path.splitext(filename)[0]
                hazy_path = os.path.join(hazy_dir, filename)

                clear_path, clear_ambiguous = _lookup_synth_stem(clear_index, stem)
                if clear_path is None:
                    self.missing_counts["clear"] += 1
                    continue

                ir_path, ir_ambiguous = _lookup_synth_stem(ir_index, stem)
                if ir_path is None:
                    self.missing_counts["ir"] += 1
                    continue

                density_path, density_ambiguous = _lookup_synth_stem(density_indexes[level], stem)
                if density_path is None:
                    self.missing_counts["density"] += 1
                    continue

                self.ambiguous_counts["clear"] += int(clear_ambiguous)
                self.ambiguous_counts["ir"] += int(ir_ambiguous)
                self.ambiguous_counts["density"] += int(density_ambiguous)

                self.samples.append({
                    "stem": stem,
                    "image_name": filename,
                    "haze_level": level,
                    "hazy_path": hazy_path,
                    "clear_path": clear_path,
                    "ir_path": ir_path,
                    "density_path": density_path,
                })
                self.level_counts[level] += 1

        counts = ", ".join(f"{level}={self.level_counts.get(level, 0)}" for level in self.haze_levels)
        print(f"[SynthMultiModalDataset] samples: {len(self.samples)} ({counts})")
        if any(self.missing_counts.values()):
            print(f"[SynthMultiModalDataset] skipped missing pairs: {self.missing_counts}")
        if any(self.ambiguous_counts.values()):
            print(
                "[SynthMultiModalDataset] WARNING: duplicate stem with multiple extensions; "
                f"using priority .png > .jpg > .jpeg. counts={self.ambiguous_counts}"
            )

    @staticmethod
    def _load_rgb(path):
        with Image.open(path) as image:
            return FF.pil_to_tensor(image.convert("RGB")).float().div_(255.0)

    @staticmethod
    def _spatial_size(path):
        with Image.open(path) as image:
            return image.size[::-1]

    def _check_alignment(self, sample):
        sizes = {key: self._spatial_size(sample[f"{key}_path"])
                 for key in ("hazy", "clear", "ir", "density")}
        hazy_size = sizes["hazy"]
        valid = (sizes["clear"] == hazy_size and sizes["density"] == hazy_size and
                 (sizes["ir"] == hazy_size or self.pair_alignment_policy == "resize_tir_to_rgb"))
        if not valid:
            paths = {key: sample[f"{key}_path"] for key in sizes}
            raise ValueError(
                f"pair alignment failed policy={self.pair_alignment_policy}, sizes={sizes}, paths={paths}"
            )

    @staticmethod
    def _resize_tensor(tensor, size):
        return F.interpolate(tensor.unsqueeze(0), size=size, mode="bilinear", align_corners=False).squeeze(0)

    def _target_size(self, tensor):
        if self.size in (None, "full"):
            return tuple(tensor.shape[-2:])
        if isinstance(self.size, int):
            return self.size, self.size
        return tuple(self.size)

    def _sync_augment(self, tensors, index):
        if not self.train:
            return tensors
        generator = torch.Generator()
        generator.manual_seed(stable_seed_mixer(self.augmentation_seed_base, self.sampler_epoch, index))
        if int(torch.randint(0, 2, (), generator=generator)):
            tensors = [torch.flip(value, dims=(-1,)) for value in tensors]
        turns = int(torch.randint(0, 4, (), generator=generator))
        if turns:
            tensors = [torch.rot90(value, turns, dims=(-2, -1)) for value in tensors]
        return tensors

    def _load_density(self, path):
        raw = load_scalar_map_as_float_tensor(
            path, normalization=self.density_map_normalization, **self.density_normalization_kwargs
        )
        return convert_density_semantics(raw, self.density_gt_semantics)

    def inspect_density_sample(self, index=0):
        if not self.samples:
            raise RuntimeError("cannot inspect density: no valid samples")
        sample = self.samples[index]
        raw = load_scalar_map_as_float_tensor(
            sample["density_path"], normalization=self.density_map_normalization,
            **self.density_normalization_kwargs
        )
        density = convert_density_semantics(raw, self.density_gt_semantics)
        return {
            "path": sample["density_path"], "semantics": self.density_gt_semantics,
            "raw_min": float(raw.min()), "raw_max": float(raw.max()), "raw_mean": float(raw.mean()),
            "density_min": float(density.min()), "density_max": float(density.max()),
            "density_mean": float(density.mean()),
        }

    def __getitem__(self, index):
        sample = self.samples[index]
        sample_name = f"{sample['haze_level']}/{sample['image_name']}"

        self._check_alignment(sample)
        hazy_vis = self._load_rgb(sample["hazy_path"])
        clear_vis = self._load_rgb(sample["clear_path"])
        infrared = load_tir_as_float_tensor(sample["ir_path"], self.tir_normalization_config)
        density_gt = self._load_density(sample["density_path"])
        target_size = self._target_size(hazy_vis)
        if tuple(hazy_vis.shape[-2:]) != target_size:
            hazy_vis = self._resize_tensor(hazy_vis, target_size)
            clear_vis = self._resize_tensor(clear_vis, target_size)
            infrared = self._resize_tensor(infrared, target_size)
            density_gt = self._resize_tensor(density_gt, target_size).clamp(0.0, 1.0)
        elif tuple(infrared.shape[-2:]) != target_size:
            infrared = self._resize_tensor(infrared, target_size)
        hazy_vis, clear_vis, infrared, density_gt = self._sync_augment(
            [hazy_vis, clear_vis, infrared, density_gt], index
        )
        return hazy_vis, clear_vis, infrared, density_gt.clamp(0.0, 1.0)

    def manifest_fingerprint(self):
        return manifest_fingerprint([
            {
                "sample": os.path.relpath(item["hazy_path"], self.root).replace("\\", "/"),
                "clear": os.path.relpath(item["clear_path"], self.root).replace("\\", "/"),
                "tir": os.path.relpath(item["ir_path"], self.root).replace("\\", "/"),
                "density": os.path.relpath(item["density_path"], self.root).replace("\\", "/"),
                "haze_level": item["haze_level"],
            }
            for item in self.samples
        ])

    def set_sampler_epoch(self, sampler_epoch):
        """Synchronize stateless augmentation descriptions with sampler state."""
        self.sampler_epoch = int(sampler_epoch)

    def __len__(self):
        return len(self.samples)


def collate_synth(batch):
    batch = [item for item in batch if item is not None and item[0] is not None]
    if not batch:
        return torch.tensor([]), torch.tensor([]), torch.tensor([]), torch.tensor([])
    return default_collate(batch)


def collate_real(batch):
    """Collate real `(RGB, TIR, metadata)` batches without inventing padding."""
    if not batch:
        return torch.tensor([]), torch.tensor([]), []
    sizes = {tuple(item[0].shape[-2:]) for item in batch}
    if len(sizes) != 1:
        raise ValueError("real batch samples must have the same spatial size; use batch_size=1 otherwise")
    hazy = torch.stack([item[0] for item in batch])
    tir = torch.stack([item[1] for item in batch])
    metadata = [item[2] for item in batch]
    return hazy, tir, metadata


class RealMultiModalDataset(data.Dataset):
    """Formal real-domain EMA dataset: `(hazy_rgb, tir, sample_metadata)` only."""

    def __init__(self, hazy_dir, tir_dir, *, format="auto", pair_alignment_policy="strict",
                 tir_normalization_config=None):
        if pair_alignment_policy not in ("strict", "resize_tir_to_rgb"):
            raise ValueError("pair_alignment_policy must be 'strict' or 'resize_tir_to_rgb'")
        self.hazy_dir, self.tir_dir = os.fspath(hazy_dir), os.fspath(tir_dir)
        self.format = format
        self.pair_alignment_policy = pair_alignment_policy
        self.tir_normalization_config = dict(tir_normalization_config or {})
        self.samples = []
        for name in sorted(os.listdir(self.hazy_dir)):
            hazy_path = os.path.join(self.hazy_dir, name)
            if not os.path.isfile(hazy_path) or not _is_image_file(name, format):
                continue
            tir_path = _find_matching_image_by_stem(self.tir_dir, name, format)
            if tir_path is None:
                raise FileNotFoundError(f"missing TIR pair for real sample: {hazy_path}")
            self.samples.append((hazy_path, tir_path))
        if not self.samples:
            raise ValueError("real dataset has no paired hazy/TIR samples")

    def manifest_fingerprint(self):
        return manifest_fingerprint([
            {"sample": os.path.basename(hazy), "tir": os.path.basename(tir)}
            for hazy, tir in self.samples
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        hazy_path, tir_path = self.samples[index]
        with Image.open(hazy_path) as image:
            hazy = FF.pil_to_tensor(image.convert("RGB")).float().div_(255.0)
        tir = load_tir_as_float_tensor(tir_path, self.tir_normalization_config)
        original_size = tuple(hazy.shape[-2:])
        if tuple(tir.shape[-2:]) != original_size:
            if self.pair_alignment_policy != "resize_tir_to_rgb":
                raise ValueError(
                    "real RGB/TIR alignment mismatch: "
                    f"hazy_path={hazy_path}, hazy_size={original_size}, "
                    f"tir_path={tir_path}, tir_size={tuple(tir.shape[-2:])}, "
                    f"policy={self.pair_alignment_policy}"
                )
            tir = F.interpolate(tir.unsqueeze(0), size=original_size, mode="bilinear", align_corners=False)[0]
        metadata = {
            "sample_id": os.path.splitext(os.path.basename(hazy_path))[0],
            "filename": os.path.basename(hazy_path),
            "original_size": original_size,
            "hazy_path": hazy_path,
            "tir_path": tir_path,
        }
        return hazy, tir, metadata


class MultiModalCLIPLoader(data.Dataset):
    def __init__(self, hazy_visible_path, infrared_path, train, size=256, format="auto"):
        super(MultiModalCLIPLoader, self).__init__()
        self.size = size
        self.train = train
        self.format = format
        self.hazy_visible_path = hazy_visible_path
        self.infrared_path = infrared_path

        result = _build_multimodal_samples(
            hazy_visible_path=hazy_visible_path,
            infrared_path=infrared_path,
            clear_visible_path=None,
            format=format,
            require_clear=False,
            dataset_name="MultiModalCLIPLoader",
        )
        _assign_sample_metadata(self, result)

    def __getitem__(self, index):
        sample = self.samples[index]

        try:
            hazy_vis = Image.open(sample["hazy_path"]).convert("RGB")
            infrared = Image.open(sample["ir_path"]).convert("RGB")

            if isinstance(self.size, int):
                min_h = min(hazy_vis.size[1], infrared.size[1])
                min_w = min(hazy_vis.size[0], infrared.size[0])
                if min_w < self.size or min_h < self.size:
                    target_h = max(self.size, min_h)
                    target_w = max(self.size, min_w)
                    hazy_vis = hazy_vis.resize((target_w, target_h), Image.BILINEAR)
                    infrared = infrared.resize((target_w, target_h), Image.BILINEAR)

                i, j, h, w = RandomCrop.get_params(hazy_vis, output_size=(self.size, self.size))
                hazy_vis = FF.crop(hazy_vis, i, j, h, w)
                infrared = FF.crop(infrared, i, j, h, w)

            hazy_vis_tensor, infrared_tensor = self.aug_and_preprocess(hazy_vis, infrared)
            return hazy_vis_tensor, infrared_tensor

        except FileNotFoundError as e:
            print(f"Error loading image: {e}. Skipping index {index}.")
            return None, None
        except Exception as e:
            print(f"An unexpected error occurred at index {index} ({sample['output_name']}): {e}")
            return None, None

    def aug_and_preprocess(self, hazy_vis, infrared):
        if self.train:
            rand_hor = random.randint(0, 1)
            if rand_hor == 1:
                hazy_vis = FF.hflip(hazy_vis)
                infrared = FF.hflip(infrared)

            rand_rot = random.randint(0, 3)
            if rand_rot > 0:
                hazy_vis = FF.rotate(hazy_vis, 90 * rand_rot)
                infrared = FF.rotate(infrared, 90 * rand_rot)

        hazy_vis_processed = preprocess_feature(hazy_vis)
        infrared_processed = preprocess_feature(infrared)
        return hazy_vis_processed, infrared_processed

    def __len__(self):
        return len(self.samples)

# --- 确保文件末尾有换行 ---
