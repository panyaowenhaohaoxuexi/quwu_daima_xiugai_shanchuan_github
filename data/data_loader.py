"""Formal RGB--TIR data loading for Source training, EMA adaptation, and evaluation."""

import hashlib
import os
from typing import Mapping, Optional

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as data
from PIL import Image
from torch.utils.data.dataloader import default_collate
from torchvision.transforms import functional as TF


COMMON_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
SYNTH_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
DEFAULT_HAZE_LEVELS = ("1_mist", "2_middle", "3_dense", "4_local_extreme")


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
                         known_integer_bits=None):
    """Convert a finite scalar/image array to float32 in [0, 1]."""
    array = np.asarray(array)
    if not array.size:
        raise ValueError(f"empty image: {path}")
    if not np.isfinite(array).all():
        raise ValueError(f"non-finite values in {path}")
    if normalization == "dtype_range":
        if np.issubdtype(array.dtype, np.uint8):
            output = array.astype(np.float32) / 255.0
        elif np.issubdtype(array.dtype, np.uint16):
            output = array.astype(np.float32) / 65535.0
        elif np.issubdtype(array.dtype, np.signedinteger) and known_integer_bits == 16:
            if array.min() < 0 or array.max() > 65535:
                raise ValueError(
                    f"invalid unsigned 16-bit range in {path}: "
                    f"min={array.min()}, max={array.max()}"
                )
            output = array.astype(np.float32) / 65535.0
        elif np.issubdtype(array.dtype, np.floating):
            output = array.astype(np.float32)
            if output.min() < 0.0 or output.max() > 1.0:
                raise ValueError(f"float image outside [0,1] in {path}; use fixed_range")
        else:
            raise ValueError(f"integer mode with unknown physical range in {path}; use fixed_range")
    elif normalization in ("fixed_range", "dataset_calibrated_range"):
        lower, upper = _normalization_bounds(
            normalization, fixed_min, fixed_max, calibrated_min, calibrated_max
        )
        output = (array.astype(np.float32) - lower) / (upper - lower)
    else:
        raise ValueError(f"unsupported normalization={normalization!r}")
    return torch.from_numpy(np.ascontiguousarray(output)).float().clamp(0.0, 1.0)


def _open_preserving_known_bit_depth(path):
    """Recover 16-bit PNG/TIFF semantics even when Pillow exposes mode ``I``."""
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
            known_bits = (bits[0] if isinstance(bits, tuple) else bits) if bits else None
        return image.mode, array, known_bits


def load_scalar_map_as_float_tensor(path, normalization="dtype_range", *, fixed_min=None,
                                    fixed_max=None, calibrated_min=None, calibrated_max=None):
    """Load a single-channel density/transmission map without lossy RGB conversion."""
    _, array, known_bits = _open_preserving_known_bit_depth(path)
    if array.ndim != 2:
        raise ValueError(f"scalar map must be single channel: {path}, shape={array.shape}")
    return _array_to_unit_float(
        array, path=path, normalization=normalization, fixed_min=fixed_min,
        fixed_max=fixed_max, calibrated_min=calibrated_min, calibrated_max=calibrated_max,
        known_integer_bits=known_bits,
    ).unsqueeze(0)


def convert_density_semantics(raw_map, density_gt_semantics="density"):
    if density_gt_semantics not in ("transmission", "density"):
        raise ValueError(f"unsupported density_gt_semantics={density_gt_semantics!r}")
    return (1.0 - raw_map if density_gt_semantics == "transmission" else raw_map).clamp(0.0, 1.0)


def load_tir_as_float_tensor(path, normalization_config: Optional[Mapping] = None):
    """Load TIR at original precision and return its replicated three-channel signal."""
    config = dict(normalization_config or {})
    normalization = config.get("normalization", "dtype_range")
    mode, array, known_bits = _open_preserving_known_bit_depth(path)
    array = np.asarray(array)
    if not array.size:
        raise ValueError(f"empty image: {path}")
    if not np.isfinite(array).all():
        raise ValueError(f"non-finite values in {path}")

    def normalize_one_channel(raw):
        raw = np.asarray(raw)
        if not raw.size:
            raise ValueError(f"empty image: {path}")
        if not np.isfinite(raw).all():
            raise ValueError(f"non-finite values in {path}")
        if normalization != "percentile":
            return _array_to_unit_float(
                raw, path=path, normalization=normalization, fixed_min=config.get("fixed_min"),
                fixed_max=config.get("fixed_max"), calibrated_min=config.get("calibrated_min"),
                calibrated_max=config.get("calibrated_max"), known_integer_bits=known_bits,
            )
        value = torch.tensor(np.array(raw, copy=True), dtype=torch.float32)
        if config.get("percentile_scope", "per_image") == "per_image":
            low = torch.quantile(value, float(config.get("percentile_low", 1.0)) / 100.0)
            high = torch.quantile(value, float(config.get("percentile_high", 99.0)) / 100.0)
            if not bool(high > low):
                return torch.zeros_like(value)
        elif config.get("percentile_scope") == "dataset":
            low_value, high_value = config.get("dataset_percentile_low_value"), config.get("dataset_percentile_high_value")
            if low_value is None or high_value is None or not float(high_value) > float(low_value):
                raise ValueError("dataset percentile requires high value > low value")
            low, high = value.new_tensor(float(low_value)), value.new_tensor(float(high_value))
        else:
            raise ValueError(f"unsupported TIR percentile scope={config.get('percentile_scope')!r}")
        return ((value - low) / (high - low)).clamp(0.0, 1.0)

    if array.ndim == 2:
        one_channel = normalize_one_channel(array).unsqueeze(0)
    elif array.ndim == 3 and array.shape[2] in (3, 4):
        rgb = array[..., :3]
        if np.issubdtype(rgb.dtype, np.integer):
            diffs = tuple(int(np.abs(rgb[..., a].astype(np.int64) - rgb[..., b].astype(np.int64)).max())
                          for a, b in ((0, 1), (0, 2), (1, 2)))
            tolerance = int(config.get("channel_tolerance_code_values", 1))
        else:
            diffs = tuple(float(np.abs(rgb[..., a] - rgb[..., b]).max()) for a, b in ((0, 1), (0, 2), (1, 2)))
            tolerance = float(config.get("channel_tolerance_float", 1e-5))
        if max(diffs) > tolerance:
            raise ValueError(f"TIR channel tolerance exceeded: path={path}, mode={mode}, max_diffs={diffs}, tolerance={tolerance}")
        if normalization == "percentile":
            one_channel = normalize_one_channel(rgb.astype(np.float64).mean(axis=2)).unsqueeze(0)
        else:
            one_channel = _array_to_unit_float(
                rgb, path=path, normalization=normalization, fixed_min=config.get("fixed_min"),
                fixed_max=config.get("fixed_max"), calibrated_min=config.get("calibrated_min"),
                calibrated_max=config.get("calibrated_max"), known_integer_bits=known_bits,
            ).mean(dim=2, keepdim=False).unsqueeze(0)
    else:
        raise ValueError(f"unsupported TIR shape in {path}: mode={mode}, shape={array.shape}")
    return one_channel.repeat(3, 1, 1).contiguous()


def stable_seed_mixer(global_augmentation_seed, sampler_epoch, sample_index):
    payload = "|".join(map(str, (global_augmentation_seed, sampler_epoch, sample_index)))
    return int.from_bytes(hashlib.blake2b(payload.encode("utf-8"), digest_size=8).digest(), "little")


def _is_image_file(name, extensions=COMMON_IMAGE_EXTS):
    return os.path.splitext(name)[1].lower() in extensions


def _list_image_files(directory, extensions):
    try:
        return sorted(name for name in os.listdir(directory) if os.path.isfile(os.path.join(directory, name)) and _is_image_file(name, extensions))
    except FileNotFoundError:
        return []


def _stem_index(directory, extensions):
    result = {}
    for name in _list_image_files(directory, extensions):
        stem = os.path.splitext(name)[0].lower()
        result.setdefault(stem, []).append(os.path.join(directory, name))
    for stem, paths in result.items():
        if len(paths) > 1:
            raise ValueError(
                f"duplicate stem in {directory}: stem={stem}, candidates={sorted(paths)}"
            )
    return result


def _lookup_stem(index, stem):
    paths = index.get(stem.lower(), [])
    return (paths[0] if paths else None), len(paths) > 1


def _load_rgb(path):
    with Image.open(path) as image:
        return TF.pil_to_tensor(image.convert("RGB")).float().div_(255.0)


def _image_size(path):
    with Image.open(path) as image:
        return image.size[::-1]


class SynthMultiModalDataset(data.Dataset):
    """Paired synthetic `(hazy RGB, clear RGB, TIR, density, completion mask)` samples."""

    def __init__(self, root, train=True, size=256, haze_levels=None, density_gt_semantics="density",
                 density_map_normalization="dtype_range", density_fixed_min=None, density_fixed_max=None,
                 density_calibrated_min=None, density_calibrated_max=None, tir_normalization_config=None,
                 pair_alignment_policy="strict", augmentation_seed_base=0, sampler_epoch=0):
        self.root, self.train, self.size = os.fspath(root), bool(train), size
        self.haze_levels = tuple(haze_levels or DEFAULT_HAZE_LEVELS)
        self.density_gt_semantics, self.density_map_normalization = density_gt_semantics, density_map_normalization
        self.density_normalization_kwargs = {"fixed_min": density_fixed_min, "fixed_max": density_fixed_max,
                                             "calibrated_min": density_calibrated_min, "calibrated_max": density_calibrated_max}
        self.tir_normalization_config = dict(tir_normalization_config or {})
        if pair_alignment_policy not in ("strict", "resize_tir_to_rgb"):
            raise ValueError("pair_alignment_policy must be 'strict' or 'resize_tir_to_rgb'")
        self.pair_alignment_policy = pair_alignment_policy
        self.augmentation_seed_base, self.sampler_epoch = int(augmentation_seed_base), int(sampler_epoch)
        self.samples, self.level_counts = [], {level: 0 for level in self.haze_levels}
        self.missing_counts = {"clear": 0, "ir": 0, "density": 0, "mask": 0}
        clear_index, tir_index = _stem_index(os.path.join(root, "clear"), SYNTH_IMAGE_EXTS), _stem_index(os.path.join(root, "ir"), SYNTH_IMAGE_EXTS)
        for level in self.haze_levels:
            density_index = _stem_index(os.path.join(root, "Transmission_Map_GT", level), SYNTH_IMAGE_EXTS)
            mask_index = _stem_index(os.path.join(root, "mask_GT", level), SYNTH_IMAGE_EXTS)
            hazy_dir = os.path.join(root, "hazy", level)
            _stem_index(hazy_dir, SYNTH_IMAGE_EXTS)
            for filename in _list_image_files(hazy_dir, SYNTH_IMAGE_EXTS):
                stem = os.path.splitext(filename)[0]
                clear_path, _ = _lookup_stem(clear_index, stem)
                tir_path, _ = _lookup_stem(tir_index, stem)
                density_path, _ = _lookup_stem(density_index, stem)
                mask_path, _ = _lookup_stem(mask_index, stem)
                if clear_path is None or tir_path is None or density_path is None or mask_path is None:
                    self.missing_counts["clear"] += int(clear_path is None)
                    self.missing_counts["ir"] += int(tir_path is None)
                    self.missing_counts["density"] += int(density_path is None)
                    self.missing_counts["mask"] += int(mask_path is None)
                    continue
                self.samples.append({"image_name": filename, "haze_level": level,
                                      "hazy_path": os.path.join(hazy_dir, filename), "clear_path": clear_path,
                                      "tir_path": tir_path, "density_path": density_path, "mask_path": mask_path})
                self.level_counts[level] += 1

    def set_sampler_epoch(self, epoch):
        self.sampler_epoch = int(epoch)

    def geometry_description(self, index, original_size):
        height, width = map(int, original_size)
        if not self.train or self.size in (None, "full"):
            return {"resized_height": height, "resized_width": width, "crop_top": 0, "crop_left": 0,
                    "crop_height": height, "crop_width": width, "horizontal_flip": False, "rot90_turns": 0}
        if not isinstance(self.size, int):
            raise ValueError("formal training size must be an integer or 'full'")
        size = int(self.size)
        scale = max(1.0, float(size) / min(height, width))
        resized_height, resized_width = max(size, round(height * scale)), max(size, round(width * scale))
        generator = torch.Generator().manual_seed(stable_seed_mixer(self.augmentation_seed_base, self.sampler_epoch, index))
        return {"resized_height": resized_height, "resized_width": resized_width,
                "crop_top": int(torch.randint(0, resized_height - size + 1, (), generator=generator)),
                "crop_left": int(torch.randint(0, resized_width - size + 1, (), generator=generator)),
                "crop_height": size, "crop_width": size,
                "horizontal_flip": bool(torch.randint(0, 2, (), generator=generator)),
                "rot90_turns": int(torch.randint(0, 4, (), generator=generator))}

    @staticmethod
    def _apply_geometry(tensors, description, interpolation_modes=None):
        size = (description["resized_height"], description["resized_width"])
        interpolation_modes = interpolation_modes or ("bilinear",) * len(tensors)
        if len(interpolation_modes) != len(tensors):
            raise ValueError("interpolation_modes must match tensors")
        resized = []
        for tensor, mode in zip(tensors, interpolation_modes):
            kwargs = {"size": size, "mode": mode}
            if mode in ("linear", "bilinear", "bicubic", "trilinear"):
                kwargs["align_corners"] = False
            resized.append(F.interpolate(tensor.unsqueeze(0), **kwargs)[0])
        tensors = resized
        top, left, height, width = (description[key] for key in ("crop_top", "crop_left", "crop_height", "crop_width"))
        tensors = [tensor[..., top:top + height, left:left + width] for tensor in tensors]
        if description["horizontal_flip"]:
            tensors = [torch.flip(tensor, dims=(-1,)) for tensor in tensors]
        return [torch.rot90(tensor, description["rot90_turns"], dims=(-2, -1)) for tensor in tensors]

    def __getitem__(self, index):
        sample = self.samples[index]
        sizes = {name: _image_size(sample[f"{name}_path"]) for name in ("hazy", "clear", "tir", "density", "mask")}
        if sizes["clear"] != sizes["hazy"] or sizes["density"] != sizes["hazy"] or sizes["mask"] != sizes["hazy"] or (
            sizes["tir"] != sizes["hazy"] and self.pair_alignment_policy != "resize_tir_to_rgb"
        ):
            raise ValueError(f"pair alignment failed policy={self.pair_alignment_policy}, sizes={sizes}")
        hazy, clear = _load_rgb(sample["hazy_path"]), _load_rgb(sample["clear_path"])
        tir = load_tir_as_float_tensor(sample["tir_path"], self.tir_normalization_config)
        density = convert_density_semantics(load_scalar_map_as_float_tensor(
            sample["density_path"], normalization=self.density_map_normalization, **self.density_normalization_kwargs
        ), self.density_gt_semantics)
        completion_mask = (load_scalar_map_as_float_tensor(sample["mask_path"]) > 0.5).float()
        if tir.shape[-2:] != hazy.shape[-2:]:
            tir = F.interpolate(tir.unsqueeze(0), size=hazy.shape[-2:], mode="bilinear", align_corners=False)[0]
        if self.train:
            hazy, clear, tir, density, completion_mask = self._apply_geometry(
                [hazy, clear, tir, density, completion_mask], self.geometry_description(index, hazy.shape[-2:]),
                interpolation_modes=("bilinear", "bilinear", "bilinear", "bilinear", "nearest"),
            )
        return hazy, clear, tir, density.clamp(0.0, 1.0), (completion_mask > 0.5).float()

    def __len__(self):
        return len(self.samples)


def collate_synth(batch):
    batch = [item for item in batch if item is not None and item[0] is not None]
    return default_collate(batch) if batch else (torch.tensor([]),) * 5


class RealMultiModalDataset(data.Dataset):
    """Paired real `(hazy RGB, TIR, metadata)` samples for EMA adaptation."""

    def __init__(self, hazy_dir, tir_dir, *, format="auto", pair_alignment_policy="strict", tir_normalization_config=None):
        if pair_alignment_policy not in ("strict", "resize_tir_to_rgb"):
            raise ValueError("pair_alignment_policy must be 'strict' or 'resize_tir_to_rgb'")
        self.hazy_dir, self.tir_dir = os.fspath(hazy_dir), os.fspath(tir_dir)
        self.pair_alignment_policy, self.tir_normalization_config = pair_alignment_policy, dict(tir_normalization_config or {})
        index = _stem_index(self.tir_dir, COMMON_IMAGE_EXTS)
        self.samples = []
        for name in _list_image_files(self.hazy_dir, COMMON_IMAGE_EXTS):
            tir_path, _ = _lookup_stem(index, os.path.splitext(name)[0])
            if tir_path is None:
                raise FileNotFoundError(f"missing TIR pair for real sample: {os.path.join(self.hazy_dir, name)}")
            self.samples.append((os.path.join(self.hazy_dir, name), tir_path))
        if not self.samples:
            raise ValueError("real dataset has no paired hazy/TIR samples")

    def __getitem__(self, index):
        hazy_path, tir_path = self.samples[index]
        hazy, tir = _load_rgb(hazy_path), load_tir_as_float_tensor(tir_path, self.tir_normalization_config)
        if tir.shape[-2:] != hazy.shape[-2:]:
            if self.pair_alignment_policy == "strict":
                raise ValueError(f"real RGB/TIR alignment mismatch: hazy_path={hazy_path}, tir_path={tir_path}")
            tir = F.interpolate(tir.unsqueeze(0), size=hazy.shape[-2:], mode="bilinear", align_corners=False)[0]
        return hazy, tir, {"sample_id": os.path.splitext(os.path.basename(hazy_path))[0], "filename": os.path.basename(hazy_path),
                           "original_size": tuple(hazy.shape[-2:]), "hazy_path": hazy_path, "tir_path": tir_path}

    def __len__(self):
        return len(self.samples)


def collate_real(batch):
    if not batch:
        return torch.tensor([]), torch.tensor([]), []
    if len({tuple(item[0].shape[-2:]) for item in batch}) != 1:
        raise ValueError("real batch samples must have the same spatial size; use batch_size=1 otherwise")
    return torch.stack([item[0] for item in batch]), torch.stack([item[1] for item in batch]), [item[2] for item in batch]
