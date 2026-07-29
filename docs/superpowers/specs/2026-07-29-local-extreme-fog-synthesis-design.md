# Local Extreme Fog Synthesis Design

## Goal

Add an independent generator for a fourth FLIR haze level, `local_extreme`, in
which one to three spatially local, optically dense fog banks completely remove
RGB information in their cores. Existing depth-based haze data and its generator
remain unchanged.

## Physics and labels

For every input image, the new generator first creates the current depth-based
transmission `T_depth`. It then samples smooth elliptical Gaussian optical-depth
fields `tau_local` and combines them as:

`T_final = T_depth * exp(-tau_local)`.

The saved supervision is the fog density `M = 1 - T_final`, as single-channel
uint8 PNG. Each fog-bank core must satisfy `T_final <= 0.05`; its feathered
boundary preserves a physically continuous transition.

## Outputs

The generator writes only new folders and preserves `clear`, `ir`, and existing
`mist`, `middle`, and `dense` outputs:

- `train/hazy/local_extreme/<stem>.jpg`
- `train/Transmission_Map_GT/local_extreme/<stem>.png`
- `test/hazy/local_extreme/<stem>.jpg`
- `test/Transmission_Map_GT/local_extreme/<stem>.png`

## Scope and verification

The new script reuses depth estimation, atmospheric-light estimation, image I/O,
and fogmap encoding from `batch_asm_haze.py` without editing that file. Tests
will prove deterministic local attenuation, core opacity, label identity
(`M = 1 - T_final`), and output pairing. A one-image visual preview is required
before the train/test batch run.
