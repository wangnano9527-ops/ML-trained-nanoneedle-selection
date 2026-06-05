from __future__ import annotations

import argparse
import csv
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tomllib
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


PACKAGE_MODULES = {
    "numpy": "numpy",
    "Pillow": "PIL",
    "scipy": "scipy",
    "torch": "torch",
    "tqdm": "tqdm",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check whether this machine can train the model.")
    parser.add_argument("--config", default=Path("configs/train.toml"), type=Path)
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    parser.add_argument("--strict-gpu", action="store_true", help="Fail if CUDA GPU is unavailable.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_report(args.config, strict_gpu=args.strict_gpu)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print_human_report(report)
    raise SystemExit(0 if report["ready_for_training"] else 1)


def build_report(config_path: Path, *, strict_gpu: bool = False) -> dict[str, Any]:
    config = load_toml_if_exists(config_path)
    data_config = config.get("data", {})
    training_config = config.get("training", {})
    output_config = config.get("output", {})

    checks: list[dict[str, str]] = []
    warnings: list[str] = []
    failures: list[str] = []

    python_ok = sys.version_info >= (3, 11)
    add_check(checks, "python", python_ok, platform.python_version(), "Python >= 3.11 recommended")
    if not python_ok:
        failures.append("Python is too old. Use Python 3.11+.")

    packages = check_packages()
    for package, info in packages.items():
        add_check(
            checks,
            f"package:{package}",
            info["installed"],
            info.get("version") or "missing",
            f"module {info['module']}",
        )
        if package in {"numpy", "Pillow", "scipy"} and not info["installed"]:
            failures.append(f"Missing preprocessing package: {package}")
        if package in {"torch", "tqdm"} and not info["installed"]:
            failures.append(f"Missing training package: {package}")

    torch_info = inspect_torch()
    if torch_info["installed"]:
        if torch_info["cuda_available"]:
            best_memory_gb = max((gpu["total_memory_gb"] for gpu in torch_info["gpus"]), default=0.0)
            if best_memory_gb < 6.0:
                warnings.append(
                    f"CUDA is available, but the largest GPU has only {best_memory_gb:.1f} GB VRAM."
                )
        else:
            message = "PyTorch is installed but CUDA is not available; CPU training may be very slow."
            if strict_gpu:
                failures.append(message)
            else:
                warnings.append(message)

    nvidia_smi = inspect_nvidia_smi()
    if not nvidia_smi["available"]:
        warnings.append("nvidia-smi was not found. This is OK for CPU, but check GPU drivers if using NVIDIA.")

    manifest_path = Path(data_config.get("manifest", "data/manifest.csv"))
    split_path = Path(data_config.get("splits", "data/splits.csv"))
    manifest_info = inspect_manifest(
        manifest_path,
        image_column=data_config.get("image_column", "image_path"),
        mask_column=data_config.get("mask_column", "mask_path"),
    )
    add_check(
        checks,
        "data:manifest",
        manifest_info["ok"],
        f"{manifest_info['row_count']} rows",
        str(manifest_path),
    )
    if not manifest_info["ok"]:
        failures.extend(manifest_info["errors"])

    split_info = inspect_splits(split_path)
    add_check(
        checks,
        "data:splits",
        split_info["ok"],
        split_info["summary"],
        str(split_path),
    )
    if not split_info["ok"]:
        failures.extend(split_info["errors"])

    disk_info = inspect_disk(Path(output_config.get("run_dir", "runs/unet_baseline")).parent)
    add_check(
        checks,
        "disk:run_dir",
        disk_info["free_gb"] >= 5.0,
        f"{disk_info['free_gb']:.1f} GB free",
        str(disk_info["path"]),
    )
    if disk_info["free_gb"] < 5.0:
        warnings.append("Less than 5 GB free near the run directory; checkpoints may fill the disk.")

    ready = not failures
    return {
        "ready_for_training": ready,
        "project_root": str(PROJECT_ROOT),
        "config": str(config_path),
        "checks": checks,
        "warnings": warnings,
        "failures": failures,
        "torch": torch_info,
        "nvidia_smi": nvidia_smi,
        "manifest": manifest_info,
        "splits": split_info,
        "disk": disk_info,
        "next_commands": suggested_commands(ready, manifest_info, split_info),
    }


def load_toml_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("rb") as handle:
        return tomllib.load(handle)


def check_packages() -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for package, module in PACKAGE_MODULES.items():
        installed = importlib.util.find_spec(module) is not None
        version = None
        if installed:
            try:
                version = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                version = "installed"
        result[package] = {"module": module, "installed": installed, "version": version}
    return result


def inspect_torch() -> dict[str, Any]:
    if importlib.util.find_spec("torch") is None:
        return {"installed": False, "cuda_available": False, "gpus": []}
    import torch

    cuda_available = bool(torch.cuda.is_available())
    gpus = []
    if cuda_available:
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            gpus.append(
                {
                    "index": index,
                    "name": props.name,
                    "total_memory_gb": round(props.total_memory / (1024**3), 2),
                    "capability": f"{props.major}.{props.minor}",
                }
            )
    return {
        "installed": True,
        "version": torch.__version__,
        "cuda_available": cuda_available,
        "cuda_version": getattr(torch.version, "cuda", None),
        "gpus": gpus,
    }


def inspect_nvidia_smi() -> dict[str, Any]:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return {"available": False, "path": None, "summary": None}
    try:
        output = subprocess.check_output(
            [
                executable,
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader",
            ],
            text=True,
            stderr=subprocess.STDOUT,
            timeout=10,
        )
    except Exception as exc:  # pragma: no cover - machine dependent.
        return {"available": False, "path": executable, "summary": str(exc)}
    return {"available": True, "path": executable, "summary": output.strip()}


def inspect_manifest(manifest_path: Path, *, image_column: str, mask_column: str) -> dict[str, Any]:
    errors: list[str] = []
    rows: list[dict[str, str]] = []
    if not manifest_path.exists():
        return {"ok": False, "row_count": 0, "errors": [f"Missing manifest: {manifest_path}"]}

    with manifest_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"sample_id", image_column, mask_column}
        missing_columns = required - set(reader.fieldnames or [])
        if missing_columns:
            return {
                "ok": False,
                "row_count": 0,
                "errors": [f"Manifest missing columns: {sorted(missing_columns)}"],
            }
        rows = list(reader)

    base_dir = manifest_path.parent.parent if manifest_path.parent.name == "data" else manifest_path.parent
    missing_files: list[str] = []
    for row in rows:
        for column in (image_column, mask_column):
            path = resolve_path(row[column], base_dir)
            if not path.exists():
                missing_files.append(str(path))
                if len(missing_files) >= 5:
                    break
        if len(missing_files) >= 5:
            break
    if missing_files:
        errors.append("Missing data files, first examples: " + "; ".join(missing_files))
    if not rows:
        errors.append("Manifest has no rows.")
    return {"ok": not errors, "row_count": len(rows), "errors": errors}


def inspect_splits(split_path: Path) -> dict[str, Any]:
    if not split_path.exists():
        return {
            "ok": False,
            "summary": "missing",
            "errors": [f"Missing split file: {split_path}. Run scripts/make_splits.py."],
        }
    counts = {"train": 0, "val": 0, "test": 0}
    with split_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if set(reader.fieldnames or []) < {"sample_id", "split"}:
            return {"ok": False, "summary": "bad columns", "errors": ["Split file needs sample_id,split columns."]}
        for row in reader:
            split = row["split"]
            counts[split] = counts.get(split, 0) + 1
    errors = []
    if counts.get("train", 0) == 0 or counts.get("val", 0) == 0:
        errors.append("Split file needs at least one train and one val sample.")
    summary = " ".join(f"{key}={counts.get(key, 0)}" for key in ["train", "val", "test"])
    return {"ok": not errors, "summary": summary, "counts": counts, "errors": errors}


def inspect_disk(path: Path) -> dict[str, Any]:
    path = path.resolve()
    existing = path
    while not existing.exists() and existing.parent != existing:
        existing = existing.parent
    usage = shutil.disk_usage(existing)
    return {
        "path": str(path),
        "free_gb": round(usage.free / (1024**3), 2),
        "total_gb": round(usage.total / (1024**3), 2),
    }


def resolve_path(value: str, base_dir: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def add_check(checks: list[dict[str, str]], name: str, ok: bool, detail: str, hint: str) -> None:
    checks.append({"name": name, "status": "PASS" if ok else "FAIL", "detail": detail, "hint": hint})


def suggested_commands(ready: bool, manifest_info: dict[str, Any], split_info: dict[str, Any]) -> list[str]:
    if ready:
        return [
            "python scripts/run_training_pipeline.py --config configs/train.toml",
            "python scripts/train_unet.py --config configs/train.toml",
        ]
    commands = []
    if not manifest_info["ok"]:
        commands.append("python scripts/preprocess_raw_data.py --config configs/preprocess.toml")
    if not split_info["ok"]:
        commands.append("python scripts/make_splits.py --config configs/train.toml")
    commands.append("python -m pip install -r requirements-ml.txt")
    return commands


def print_human_report(report: dict[str, Any]) -> None:
    print("Training Environment Check")
    print(f"project: {report['project_root']}")
    print(f"config: {report['config']}")
    print()
    for check in report["checks"]:
        print(f"{check['status']:4} {check['name']}: {check['detail']} ({check['hint']})")
    if report["warnings"]:
        print("\nWarnings:")
        for warning in report["warnings"]:
            print(f"- {warning}")
    if report["failures"]:
        print("\nFailures:")
        for failure in report["failures"]:
            print(f"- {failure}")
    print("\nNext commands:")
    for command in report["next_commands"]:
        print(f"- {command}")
    print(f"\nready_for_training={report['ready_for_training']}")


if __name__ == "__main__":
    main()

