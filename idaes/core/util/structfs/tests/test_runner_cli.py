#################################################################################
# The Institute for the Design of Advanced Energy Systems Integrated Platform
# Framework (IDAES IP) was produced under the DOE Institute for the
# Design of Advanced Energy Systems (IDAES).
#
# Copyright (c) 2018-2026 by the software owners: The Regents of the
# University of California, through Lawrence Berkeley National Laboratory,
# National Technology & Engineering Solutions of Sandia, LLC, Carnegie Mellon
# University, West Virginia University Research Corporation, et al.
# All rights reserved.  Please see the files COPYRIGHT.md and LICENSE.md
# for full copyright and license information.
#################################################################################
import json
from types import SimpleNamespace

import pytest

from .. import runner_cli


def _write_runner_module(path, body="ctx['ran'] = True"):
    path.write_text(
        "\n".join(
            [
                "from idaes.core.util.structfs.runner import Runner",
                "FS = Runner(('build',))",
                "@FS.step('build')",
                "def build(ctx):",
                f"    {body}",
            ]
        )
    )


@pytest.mark.unit
def test_load_module_from_file_and_errors(tmp_path):
    module_path = tmp_path / "demo_runner.py"
    _write_runner_module(module_path)

    mod = runner_cli._load_module(str(module_path))
    assert mod.FS.list_steps(all_steps=True) == ["build"]

    with pytest.raises(FileNotFoundError):
        runner_cli._load_module(str(tmp_path / "missing.py"))

    with pytest.raises(ValueError, match="Relative module names not allowed"):
        runner_cli._load_module(".demo")


@pytest.mark.unit
def test_main_info_mode(tmp_path, monkeypatch):
    module_path = tmp_path / "info_runner.py"
    output_path = tmp_path / "out.json"
    _write_runner_module(module_path)

    monkeypatch.setattr(
        "sys.argv",
        ["runner_cli", str(module_path), str(output_path), "--info"],
    )

    assert runner_cli.main() == 0

    data = json.loads(output_path.read_text())
    assert data["steps"] == ["build"]
    assert data["class_name"] == "Runner"


@pytest.mark.unit
def test_main_run_mode_and_runtime_error(tmp_path, monkeypatch):
    module_path = tmp_path / "run_runner.py"
    output_path = tmp_path / "run.json"
    _write_runner_module(module_path)

    monkeypatch.setattr(
        "sys.argv",
        ["runner_cli", str(module_path), str(output_path)],
    )
    assert runner_cli.main() == 0
    data = json.loads(output_path.read_text())
    assert data["status"] == 0
    assert data["last_run"] == ["build"]

    failing_module = tmp_path / "failing_runner.py"
    failing_output = tmp_path / "failing.json"
    _write_runner_module(failing_module, body="raise RuntimeError('boom')")
    monkeypatch.setattr(
        "sys.argv",
        ["runner_cli", str(failing_module), str(failing_output)],
    )
    assert runner_cli.main() == 5
    data = json.loads(failing_output.read_text())
    assert data["status"] == 5
    assert "While running steps" in data["error"]


@pytest.mark.unit
def test_main_open_module_and_object_errors(tmp_path, monkeypatch):
    bad_output = tmp_path / "missing" / "out.json"
    monkeypatch.setattr("sys.argv", ["runner_cli", "nope", str(bad_output)])
    assert runner_cli.main() == -1

    output_path = tmp_path / "obj.json"
    monkeypatch.setattr(
        runner_cli,
        "_load_module",
        lambda _: SimpleNamespace(FS=object()),
    )
    monkeypatch.setattr("sys.argv", ["runner_cli", "demo", str(output_path)])
    assert runner_cli.main() == 3
    data = json.loads(output_path.read_text())
    assert data["status"] == 3

    missing_obj_path = tmp_path / "missing_obj.json"
    monkeypatch.setattr(
        runner_cli,
        "_load_module",
        lambda _: SimpleNamespace(),
    )
    monkeypatch.setattr("sys.argv", ["runner_cli", "demo", str(missing_obj_path)])
    assert runner_cli.main() == 4
    data = json.loads(missing_obj_path.read_text())
    assert data["status"] == 4

    missing_mod_path = tmp_path / "missing_mod.json"
    monkeypatch.setattr(
        runner_cli,
        "_load_module",
        lambda _: (_ for _ in ()).throw(ModuleNotFoundError("demo")),
    )
    monkeypatch.setattr("sys.argv", ["runner_cli", "demo", str(missing_mod_path)])
    assert runner_cli.main() == 2
    data = json.loads(missing_mod_path.read_text())
    assert data["status"] == 2
