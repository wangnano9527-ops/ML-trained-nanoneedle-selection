from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from needle_select import (  # noqa: E402
    check_environment,
    describe_project,
    init_project,
    list_capabilities,
    list_public_steps,
    run_project,
)
from needle_select.cli import main  # noqa: E402
from needle_select.config import load_project_config  # noqa: E402


class PublicApiTests(unittest.TestCase):
    def test_describe_lists_capabilities_and_steps(self) -> None:
        description = describe_project()
        capability_names = {item["name"] for item in list_capabilities()}
        step_names = {item["name"] for item in list_public_steps()}

        self.assertEqual(description["package"], "needle_select")
        self.assertIn("profile_aware_inference", capability_names)
        self.assertIn("preflight_screen", capability_names)
        self.assertIn("infer", step_names)
        self.assertIn("screen", step_names)
        self.assertIn("data_flow", description)

    def test_init_project_generates_portable_skeleton(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = init_project(tmp)
            target = Path(tmp)

            self.assertTrue((target / "data").is_dir())
            self.assertTrue((target / "output").is_dir())
            self.assertTrue((target / "work").is_dir())
            self.assertTrue((target / "configs" / "needle_select_project.toml").is_file())
            self.assertTrue((target / "configs" / "inference_profile.json").is_file())
            self.assertTrue((target / "scripts" / "infer_needles.py").is_file())
            self.assertTrue((target / "screen.ps1").is_file())
            self.assertTrue((target / "screen.cmd").is_file())
            self.assertTrue((target / "screen.sh").is_file())
            self.assertFalse((target / "__init__.py").exists())
            self.assertIn("README_RUN.md", result["copied_files"])

            config = load_project_config(target / "configs" / "needle_select_project.toml")
            self.assertEqual(config.paths.project_root, target.resolve())
            self.assertEqual(config.inference.patch_size, 256)

    def test_run_project_dry_run_builds_plan(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            init_project(tmp)
            config_path = Path(tmp) / "configs" / "needle_select_project.toml"
            result = run_project(config_path, steps=["doctor", "infer"], dry_run=True)

            self.assertTrue(result["dry_run"])
            self.assertEqual([step["name"] for step in result["steps"]], ["doctor", "infer"])
            self.assertTrue(any("infer_needles.py" in part for part in result["steps"][1]["command"]))

    def test_check_environment_returns_ready_shape(self) -> None:
        result = check_environment(ROOT / "configs" / "smoke.project.toml")

        self.assertIn("packages", result)
        self.assertIn("paths", result)
        self.assertIn("gpu", result)
        self.assertIsInstance(result["ready"], bool)

    def test_cli_describe_json(self) -> None:
        output = io.StringIO()
        with redirect_stdout(output):
            code = main(["describe", "--json"])

        self.assertEqual(code, 0)
        payload = json.loads(output.getvalue())
        self.assertEqual(payload["package"], "needle_select")

    def test_cli_init_project_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "consumer"
            output = io.StringIO()
            with redirect_stdout(output):
                code = main(["init-project", str(target), "--json"])

            self.assertEqual(code, 0)
            payload = json.loads(output.getvalue())
            self.assertEqual(Path(payload["target_dir"]), target.resolve())
            self.assertTrue((target / "doctor.ps1").is_file())


if __name__ == "__main__":
    unittest.main()
