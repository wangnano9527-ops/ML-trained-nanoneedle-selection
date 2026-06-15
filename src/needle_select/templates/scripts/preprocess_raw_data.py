from __future__ import annotations

import argparse
from dataclasses import fields
from pathlib import Path
import sys
import tomllib

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from needle_select.preprocess import MaskCleanConfig, preprocess_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess raw nano-needle tif/mask pairs.")
    parser.add_argument("--config", default=Path("configs/preprocess.toml"), type=Path)
    parser.add_argument("--raw-dir", default=None, type=Path)
    parser.add_argument("--out-dir", default=None, type=Path)
    parser.add_argument("--channel", default=None, type=int, help="Zero-based channel/page index.")
    parser.add_argument("--min-area-factor", default=None, type=float)
    parser.add_argument("--network-radius-factor", default=None, type=float)
    parser.add_argument("--network-min-neighbors", default=None, type=int)
    parser.add_argument("--min-network-cluster-size", default=None, type=int)
    parser.add_argument(
        "--lattice-filter",
        dest="use_lattice_filter",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--lattice-phase-tolerance",
        default=None,
        type=float,
        help="Allowed grid phase residual as a fraction of the lattice pitch.",
    )
    parser.add_argument("--lattice-min-axial-neighbors", default=None, type=int)
    parser.add_argument("--lattice-axis-angle-tolerance-deg", default=None, type=float)
    parser.add_argument("--no-overwrite", action="store_true")
    return parser.parse_args()


def load_preprocess_settings(config_path: Path) -> dict:
    settings = {
        "paths": {
            "raw_dir": Path("raw data"),
            "out_dir": Path("data"),
        },
        "channel": 0,
        "config": {},
    }
    if config_path and config_path.exists():
        with config_path.open("rb") as handle:
            payload = tomllib.load(handle)
        paths = payload.get("paths", {})
        settings["paths"]["raw_dir"] = Path(paths.get("raw_dir", settings["paths"]["raw_dir"]))
        settings["paths"]["out_dir"] = Path(paths.get("out_dir", settings["paths"]["out_dir"]))
        manual = payload.get("manual", {})
        advanced = payload.get("advanced", {})
        if "channel" in manual:
            settings["channel"] = int(manual["channel"])
        config_keys = {field.name for field in fields(MaskCleanConfig)}
        settings["config"] = {
            key: value
            for section in (manual, advanced)
            for key, value in section.items()
            if key in config_keys
        }
    elif config_path:
        print(f"Config not found, using built-in defaults: {config_path}")
    return settings


def main() -> None:
    args = parse_args()
    settings = load_preprocess_settings(args.config)
    raw_dir = args.raw_dir or settings["paths"]["raw_dir"]
    out_dir = args.out_dir or settings["paths"]["out_dir"]
    channel = args.channel if args.channel is not None else settings["channel"]

    config_values = dict(settings["config"])
    for name in [
        "min_area_factor",
        "network_radius_factor",
        "network_min_neighbors",
        "min_network_cluster_size",
        "use_lattice_filter",
        "lattice_phase_tolerance",
        "lattice_min_axial_neighbors",
        "lattice_axis_angle_tolerance_deg",
    ]:
        value = getattr(args, name)
        if value is not None:
            config_values[name] = value
    config = MaskCleanConfig(**config_values)
    results = preprocess_dataset(
        raw_dir,
        out_dir,
        config=config,
        channel_index=channel,
        overwrite=not args.no_overwrite,
    )
    removed_small = sum(result.stats.removed_small_components for result in results)
    removed_network = sum(result.stats.removed_network_components for result in results)
    print(f"Processed {len(results)} tif/mask pairs.")
    print(f"Removed {removed_small} small components and {removed_network} off-network components.")
    print(f"Wrote outputs to {out_dir}.")


if __name__ == "__main__":
    main()
