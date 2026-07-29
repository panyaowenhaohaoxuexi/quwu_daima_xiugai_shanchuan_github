# Local Extreme Fog Synthesis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate a fourth, paired FLIR haze level whose local fog-bank cores fully obscure RGB information.

**Architecture:** Create a standalone `batch_local_extreme_haze.py` beside the existing generator. It imports the existing depth, atmospheric-light, I/O, and fogmap helpers, then combines depth transmission with deterministic, soft elliptical local optical-depth fields. It writes only `local_extreme` folders and never alters existing levels.

**Tech Stack:** Python, NumPy, Pillow, OpenCV/SciPy, Depth Anything V2, pytest.

---

### Task 1: Define and test local optical-depth synthesis

**Files:**
- Create: `E:/Github_code_upload/2_image_haze_depth_Generation/tests/test_batch_local_extreme_haze.py`
- Create: `E:/Github_code_upload/2_image_haze_depth_Generation/batch_local_extreme_haze.py`

- [ ] **Step 1: Write failing tests**

```python
def test_local_optical_depth_is_deterministic_and_localized():
    field_a = sample_local_optical_depth((64, 96), seed=7, blob_count=1)
    field_b = sample_local_optical_depth((64, 96), seed=7, blob_count=1)
    np.testing.assert_allclose(field_a, field_b)
    assert field_a.max() > 3.0
    assert (field_a < 1e-3).mean() > 0.2


def test_local_extreme_transmission_and_density_are_physically_paired():
    t_depth = np.ones((9, 9), dtype=np.float32) * 0.9
    tau = np.zeros((9, 9), dtype=np.float32)
    tau[4, 4] = 4.0
    t_final, density = combine_depth_and_local_fog(t_depth, tau)
    assert t_final[4, 4] <= 0.05
    np.testing.assert_allclose(density, 1.0 - t_final)
```

- [ ] **Step 2: Run the focused tests and verify they fail because the module does not exist**

Run: `D:/anaconda/envs/CoA/python.exe -m pytest tests/test_batch_local_extreme_haze.py -q -p no:cacheprovider`

Expected: import failure for `batch_local_extreme_haze`.

- [ ] **Step 3: Implement minimal pure functions**

Implement `sample_local_optical_depth()` with 1–3 seeded elliptical Gaussian blobs and `combine_depth_and_local_fog()` as:

```python
t_final = np.clip(t_depth * np.exp(-tau_local), 0.0, 1.0)
density = 1.0 - t_final
```

Choose per-blob peak optical depth so the core reaches `T <= 0.05` even when `T_depth <= 1`.

- [ ] **Step 4: Run the focused tests and verify they pass**

Run: `D:/anaconda/envs/CoA/python.exe -m pytest tests/test_batch_local_extreme_haze.py -q -p no:cacheprovider`

Expected: `2 passed`.

### Task 2: Add paired one-image generation

**Files:**
- Modify: `E:/Github_code_upload/2_image_haze_depth_Generation/batch_local_extreme_haze.py`
- Modify: `E:/Github_code_upload/2_image_haze_depth_Generation/tests/test_batch_local_extreme_haze.py`

- [ ] **Step 1: Write a failing paired-output test**

```python
def test_process_one_image_writes_local_extreme_hazy_and_density_pair(tmp_path):
    process_one_image(..., output_hazy=tmp_path / 'hazy/local_extreme/sample.jpg',
                      output_density=tmp_path / 'Transmission_Map_GT/local_extreme/sample.png')
    assert output_hazy.exists()
    assert output_density.exists()
```

- [ ] **Step 2: Run it and verify it fails because `process_one_image` is absent**

Run: `D:/anaconda/envs/CoA/python.exe -m pytest tests/test_batch_local_extreme_haze.py -q -p no:cacheprovider`

Expected: failure naming the missing function.

- [ ] **Step 3: Implement the generator CLI**

Reuse `DepthEstimator`, `depth_to_meters`, `smooth_depth_for_haze`, `estimate_atmospheric_light`, `save_rgb_float`, and `save_fogmap` from `batch_asm_haze.py`. Use current defaults (`depth sigma=3`, no transmission smoothing, sky mask disabled), one `local_extreme` output per clear image, and a filename-derived seed for reproducibility.

- [ ] **Step 4: Re-run the new generator tests**

Run: `D:/anaconda/envs/CoA/python.exe -m pytest tests/test_batch_local_extreme_haze.py -q -p no:cacheprovider`

Expected: all local-fog tests pass.

### Task 3: Visual approval and safe batch generation

**Files:**
- Create: `C:/Users/24098/.codex/visualizations/2026/07/29/.../local_extreme_preview.png`
- Create at batch time: `F:/1_paper_pan/1_Dehaze_Paper/2_Dataset/1_main_benchmark/1_FLIR/{train,test}/{hazy,Transmission_Map_GT}/local_extreme/`

- [ ] **Step 1: Run one randomly chosen test image to a temporary preview directory**

Render clear RGB, local-extreme hazy RGB, transmission, and density map. Confirm that at least one fog-bank core has `T <= 0.05` and that its density label is correspondingly near white.

- [ ] **Step 2: Obtain user approval of the preview**

Do not start the full batch until the displayed local-fog shape is accepted.

- [ ] **Step 3: Launch train then test generation sequentially**

Use the standalone script with explicit input/output directories. It must create only the new `local_extreme` children, never overwrite existing `mist`, `middle`, or `dense` files.

- [ ] **Step 4: Verify completion**

Check that each of the four new folders contains exactly its source split's image count and that each hazy filename has one same-stem density PNG.
