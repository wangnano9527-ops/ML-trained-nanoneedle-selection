# Preprocessing Parameters

This file separates parameters into three groups:

- **Automatically estimated**: measured from each mask/image; you do not set these.
- **Manual tuning**: useful knobs when the result is too strict or too loose.
- **Advanced**: safe defaults for the current data; change only when the imaging setup changes.

The same information is also available programmatically:

```python
from needle_select import describe_preprocess_parameters

for spec in describe_preprocess_parameters():
    print(spec.name, spec.group, spec.user_level, spec.default)
```

## Automatically Estimated Per Image

| Quantity | Where recorded | Meaning |
| --- | --- | --- |
| `area_peak_px` | `data/preprocess_summary.json` | Dominant connected-component area peak for real nano-needle dots. |
| `min_area_px` | `data/preprocess_summary.json` | Actual area cutoff, computed as `area_peak_px * min_area_factor`. |
| `lattice_angle_deg` | `data/manifest.csv` | Rotation of the estimated array axis, modulo 90 degrees. |
| `lattice_pitch_px` | `data/manifest.csv` | Estimated center-to-center nano-needle spacing. Current data is about 31 px. |
| `lattice_phase_tolerance_px` | `data/manifest.csv` | Actual grid residual tolerance, computed from pitch and `lattice_phase_tolerance`. |
| `lattice_candidate_components` | `data/manifest.csv` | Points passing lattice phase or axial-neighbor support before final cluster cleanup. |

## Manual Tuning Parameters

These live in `configs/preprocess.toml` under `[manual]` and can also be passed to `scripts/preprocess_raw_data.py`.

| Parameter | Default | Increase when | Decrease when |
| --- | ---: | --- | --- |
| `channel` | `0` | Use a later TIFF page/channel. | Use channel1/page 0. |
| `min_area_factor` | `0.45` | Tiny debris remains. | Real but small needles are removed. |
| `lattice_phase_tolerance` | `0.36` | Edge/warped valid points are removed. | Gap/off-grid points remain. |
| `lattice_min_axial_neighbors` | `2` | Too many isolated off-grid points remain. | Edge points with missing neighbors are removed. |
| `use_lattice_filter` | `true` | Usually keep true. | Set false only for a non-regular pattern. |

Recommended first adjustments:

```powershell
# Stricter lattice filtering.
python scripts/preprocess_raw_data.py --config configs/preprocess.toml --lattice-phase-tolerance 0.32

# More forgiving lattice filtering.
python scripts/preprocess_raw_data.py --config configs/preprocess.toml --lattice-phase-tolerance 0.40
```

## Advanced Parameters

These are in `[advanced]`. Most users should not need to touch them.

| Parameter | Default | Role |
| --- | ---: | --- |
| `connectivity` | `2` | 8-connected component labeling for masks. |
| `area_hist_bins` | `48` | Resolution of the automatic dot-area peak histogram. |
| `network_radius_factor` | `2.4` | Radius, in pitch units, for final main-cluster cleanup. |
| `lattice_neighbor_k` | `10` | Number of local neighbors used to estimate lattice vectors. |
| `lattice_vector_distance_tolerance` | `0.45` | Distance window around pitch seed for candidate lattice vectors. |
| `lattice_axis_angle_tolerance_deg` | `18.0` | Angular tolerance when refining the two perpendicular axes. |
| `lattice_axial_distance_tolerance` | `0.32` | Allowed distance error for up/down/left/right neighbor support. |
| `lattice_axial_lateral_tolerance` | `0.28` | Allowed sideways error for axial neighbor support. |
| `lattice_min_final_fraction` | `0.35` | Safety fallback: if lattice filtering keeps too few points, use density fallback. |
