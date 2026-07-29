# Offline HDE Transmission Inference Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a new single-pair script that uses an offline HDE transmission PNG with `source_best.pt` to generate the route and dehazed outputs.

**Architecture:** `module_verify_HDE` remains an independent producer of raw transmission `T`. New `test_single_pair_external_hde.py` reads only the saved single-channel image, converts it to `D=1-T`, replaces density and recomputes routing in a normal main-model context, then calls the unchanged decoder. Existing `test_single_pair.py` is not modified.

**Tech Stack:** Python 3, PyTorch, Pillow/torchvision, pytest.

---

### Task 1: Add failing tests for the new script helpers

**Files:**
- Create: `E:/Github_code_upload/Multimodal_Dehaze_code/tests/test_single_pair_external_density.py`
- Create: `E:/Github_code_upload/Multimodal_Dehaze_code/test_single_pair_external_hde.py`

- [ ] **Step 1: Write failing tests**

```python
from pathlib import Path

import pytest
import torch
from PIL import Image

from test_single_pair_external_hde import (
    build_parser, inject_external_density, load_external_transmission_map,
)


class _Router:
    def __call__(self, density_map, temperature):
        return {
            "route_logits": density_map + float(temperature),
            "route_soft": density_map * 0.5,
            "route_hard": (density_map > 0.5).to(density_map.dtype),
        }


def test_external_transmission_is_inverted_and_requires_exact_size(tmp_path):
    path = Path(tmp_path) / "transmission.png"
    Image.new("L", (5, 3), color=64).save(path)
    transmission, density = load_external_transmission_map(path, expected_size=(3, 5))
    assert transmission.shape == density.shape == (1, 1, 3, 5)
    assert torch.allclose(density, 1.0 - transmission)
    with pytest.raises(ValueError, match="size mismatch"):
        load_external_transmission_map(path, expected_size=(4, 5))


def test_injection_replaces_only_density_and_recomputes_route():
    original = torch.zeros(1, 1, 2, 2)
    context = {
        "density_map": original, "route_logits": original, "route_soft": original,
        "route_hard": original, "tir_structure_pyramid": {"h2": torch.ones(1, 32, 1, 1)},
    }
    density = torch.tensor([[[[0.0, 0.25], [0.75, 1.0]]]])
    injected = inject_external_density(context, density, _Router(), route_temperature=0.2)
    assert injected is not context
    assert injected["tir_structure_pyramid"] is context["tir_structure_pyramid"]
    assert torch.equal(context["density_map"], original)
    assert torch.equal(injected["density_map"], density)
    assert torch.equal(injected["route_soft"], density * 0.5)
    assert torch.equal(injected["route_hard"], (density > 0.5).float())


def test_external_transmission_argument_is_required():
    with pytest.raises(SystemExit):
        build_parser().parse_args([
            "--checkpoint", "checkpoint.pt", "--hazy", "hazy.png", "--tir", "tir.png", "--output_dir", "out"
        ])
```

- [ ] **Step 2: Run the test and verify RED**

Run: `D:\anaconda\envs\CoA\python.exe -m pytest -q tests/test_single_pair_external_density.py`

Expected: FAIL during collection because `test_single_pair_external_hde` does not exist.

### Task 2: Implement the new standalone inference script

**Files:**
- Create: `E:/Github_code_upload/Multimodal_Dehaze_code/test_single_pair_external_hde.py`
- Test: `E:/Github_code_upload/Multimodal_Dehaze_code/tests/test_single_pair_external_density.py`

- [ ] **Step 1: Implement minimal image and context helpers**

```python
def load_external_transmission_map(path, expected_size):
    transmission = load_scalar_map_as_float_tensor(path).unsqueeze(0)
    if tuple(transmission.shape[-2:]) != tuple(expected_size):
        raise ValueError(
            "external transmission size mismatch: "
            f"map={tuple(transmission.shape[-2:])}, rgb={tuple(expected_size)}"
        )
    return transmission, 1.0 - transmission


def inject_external_density(context, density_map, router, route_temperature):
    if tuple(density_map.shape) != tuple(context["density_map"].shape):
        raise ValueError(
            f"external density shape mismatch: map={tuple(density_map.shape)}, "
            f"expected={tuple(context['density_map'].shape)}"
        )
    injected = dict(context)
    injected.update({"density_map": density_map, **router(density_map, route_temperature)})
    return injected
```

- [ ] **Step 2: Implement `main` using only main-project imports and a required `--external_transmission_map` argument**

Reuse the checkpoint loading, RGB/TIR loading, strict alignment, and output saving conventions in `test_single_pair.py`. Do not import `module_verify_HDE`. Encode the main model normally, call `inject_external_density`, then call `model.decode_with_route(context, route_mode="hard")`.

Save `{stem}_external_transmission.png`, `{stem}_density_map.png`, `{stem}_route_soft.png`, `{stem}_route_hard.png`, `{stem}_boundary_map.png`, and `{stem}_dehazed.png`. Print the hard-completion fraction.

- [ ] **Step 3: Run the focused test and verify GREEN**

Run: `D:\anaconda\envs\CoA\python.exe -m pytest -q tests/test_single_pair_external_density.py`

Expected: PASS.

### Task 3: Verify the offline-map inference end to end

**Files:**
- Test: `E:/Github_code_upload/Multimodal_Dehaze_code/test_single_pair_external_hde.py`

- [ ] **Step 1: Run the new script on the known FLIR dense pair**

Run:

```powershell
& 'D:\anaconda\envs\CoA\python.exe' test_single_pair_external_hde.py --checkpoint Teacher_Train\source_best.pt --hazy 'F:\1_paper_pan\1_Dehaze_Paper\2_Dataset\1_main_benchmark\1_FLIR\test\hazy\dense\FLIR_04884.jpg' --tir 'F:\1_paper_pan\1_Dehaze_Paper\2_Dataset\1_main_benchmark\1_FLIR\test\ir\FLIR_04884.jpg' --external_transmission_map 'F:\1_paper_pan\1_Dehaze_Paper\module_verify_HDE\outputs\single_test\FLIR_04884_density.png' --output_dir 'F:\1_paper_pan\1_Dehaze_Paper\1_images_duibi_experiments\main_benchmark\1_Ours\FLIR_test_external_hde'
```

Expected: all outputs are `640 x 512`; only the PNG map is read from the HDE verification project.

- [ ] **Step 2: Check density-map semantics**

Run:

```powershell
& 'D:\anaconda\envs\CoA\python.exe' -c "from PIL import Image; import numpy as np; a=np.asarray(Image.open(r'F:\1_paper_pan\1_Dehaze_Paper\1_images_duibi_experiments\main_benchmark\1_Ours\FLIR_test_external_hde\FLIR_04884_external_transmission.png'),dtype=np.float32)/255; d=np.asarray(Image.open(r'F:\1_paper_pan\1_Dehaze_Paper\1_images_duibi_experiments\main_benchmark\1_Ours\FLIR_test_external_hde\FLIR_04884_density_map.png'),dtype=np.float32)/255; print(float(np.abs(d-(1-a)).max()))"
```

Expected: no more than `1/255` error from PNG quantization.

- [ ] **Step 3: Commit only new feature files when requested**

Do not stage existing untracked assets. Commit only `test_single_pair_external_hde.py`, `tests/test_single_pair_external_density.py`, and the two design documents if the user requests a commit.
