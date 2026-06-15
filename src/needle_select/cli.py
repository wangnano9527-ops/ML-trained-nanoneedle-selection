from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .project_api import (
    check_environment,
    describe_project,
    init_project,
    list_capabilities,
    list_public_steps,
    run_project,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="needle-select", description="Needle Select reusable project CLI.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    describe = subparsers.add_parser("describe", help="Describe project capabilities and data flow.")
    describe.add_argument("--json", action="store_true")

    capabilities = subparsers.add_parser("capabilities", help="List public capabilities.")
    capabilities.add_argument("--json", action="store_true")

    steps = subparsers.add_parser("steps", help="List public pipeline steps.")
    steps.add_argument("--json", action="store_true")

    init = subparsers.add_parser("init-project", help="Create a clean run directory skeleton.")
    init.add_argument("target_dir", type=Path)
    init.add_argument("--overwrite", action="store_true")
    init.add_argument("--json", action="store_true")

    doctor = subparsers.add_parser("doctor", help="Check environment and configured paths.")
    doctor.add_argument("--config", type=Path, default=None)
    doctor.add_argument("--json", action="store_true")

    run = subparsers.add_parser("run", help="Run configured workflow steps.")
    run.add_argument("--config", required=True, type=Path)
    run.add_argument("--steps", default=None, help="Comma-separated steps. Defaults to doctor,preprocess,make-splits,train,predict.")
    run.add_argument("--dry-run", action="store_true")
    run.add_argument("--json", action="store_true")

    plan = subparsers.add_parser("plan", help="Print the workflow plan without executing it.")
    plan.add_argument("--config", required=True, type=Path)
    plan.add_argument("--steps", default=None, help="Comma-separated steps.")
    plan.add_argument("--json", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "describe":
        return emit(describe_project(), as_json=args.json)
    if args.command == "capabilities":
        return emit({"capabilities": list_capabilities()}, as_json=args.json)
    if args.command == "steps":
        return emit({"steps": list_public_steps()}, as_json=args.json)
    if args.command == "init-project":
        return emit(init_project(args.target_dir, overwrite=args.overwrite), as_json=args.json)
    if args.command == "doctor":
        return emit(check_environment(args.config), as_json=args.json)
    if args.command == "run":
        result = run_project(args.config, steps=parse_steps(args.steps), dry_run=args.dry_run)
        return emit(result, as_json=args.json)
    if args.command == "plan":
        result = run_project(args.config, steps=parse_steps(args.steps), dry_run=True)
        return emit(result, as_json=args.json)
    parser.error(f"Unhandled command: {args.command}")
    return 2


def parse_steps(value: str | None) -> list[str] | None:
    if not value:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def emit(payload: dict, *, as_json: bool) -> int:
    if as_json:
        print(json.dumps(payload, indent=2))
    else:
        print_human(payload)
    return 0


def print_human(payload: dict) -> None:
    if "summary" in payload and "capabilities" in payload:
        print(f"{payload['name']}: {payload['summary']}")
        print("\nCapabilities:")
        for capability in payload["capabilities"]:
            print(f"- {capability['name']}: {capability['description']}")
        print("\nPublic steps:")
        for step in payload["public_steps"]:
            print(f"- {step['name']}: {step['description']}")
        return
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
