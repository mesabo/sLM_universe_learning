"""Smoke tests for `shared/` — must pass before any class is "done".

These tests do NOT download model weights. They only:
  - Import every shared module.
  - Verify configs load and validate.
  - Verify `run_eval` writes a JSON and respects expected_band.

Heavier tests that download weights live in their respective class folders
and run as part of `run.sh`.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from shared import (
    backbones,
    config,
    continual,
    datasets,
    eval_harness,
    launcher,
    logging_utils,
    paths,
    repro,
    training,
)


def test_imports_resolve():
    """All shared modules import cleanly (catches syntax / dep errors fast)."""
    assert all(
        m is not None
        for m in [
            backbones, config, continual, datasets, eval_harness,
            launcher, logging_utils, paths, repro, training,
        ]
    )


def test_continual_history_and_metrics_two_tasks():
    """Worked example: AG News → TREC sequential training with clear forgetting."""
    h = continual.History()
    h.add_stage(0, accuracies={0: 0.25, 1: 0.30})   # before training: ~random
    h.add_stage(1, accuracies={0: 0.92, 1: 0.31})   # after task A: A learned, B unaffected
    h.add_stage(2, accuracies={0: 0.55, 1: 0.88})   # after task B: B learned, A forgotten

    s = continual.summarize(h)
    assert s["n_tasks"] == 2
    assert s["n_stages"] == 3
    assert abs(s["avg_accuracy"] - (0.55 + 0.88) / 2) < 1e-9
    # BWT = R[2][0] - R[1][0] = 0.55 - 0.92 = -0.37 (forgetting)
    assert abs(s["BWT"] - (0.55 - 0.92)) < 1e-9
    # FWT = R[1][1] - R[0][1] = 0.31 - 0.30 = +0.01 (negligible)
    assert abs(s["FWT"] - (0.31 - 0.30)) < 1e-9


def test_continual_stage_order_enforced():
    h = continual.History()
    h.add_stage(0, {0: 0.5})
    with pytest.raises(ValueError):
        h.add_stage(2, {0: 0.6})  # skipped stage 1


def test_continual_handles_too_few_stages():
    h = continual.History()
    h.add_stage(0, {0: 0.5, 1: 0.5})
    h.add_stage(1, {0: 0.9, 1: 0.5})
    s = continual.summarize(h)
    assert s["n_tasks"] == 2 and s["n_stages"] == 2
    # Need T+1=3 stages for BWT/FWT; we have only 2.
    assert s["BWT"] != s["BWT"]   # NaN
    assert s["FWT"] != s["FWT"]   # NaN


def test_classification_metrics_perfect_and_random():
    perfect = training.classification_metrics([0, 1, 2, 1], [0, 1, 2, 1])
    assert perfect == {"accuracy": 1.0, "f1_macro": 1.0}
    random = training.classification_metrics([0, 0, 0, 0], [0, 1, 0, 1])
    assert random["accuracy"] == 0.5
    assert 0.0 <= random["f1_macro"] <= 1.0


def test_lora_rank_sweep_grid_expands_to_twelve():
    lines = launcher.expand(
        str(paths.project_root() / "courses/course1_finetuning/grids/lora_rank_sweep.yaml")
    )
    assert len(lines) == 12
    assert all("train.py" in line for line in lines)


def test_project_root_has_pyproject():
    root = paths.project_root()
    assert (root / "pyproject.toml").is_file()


def test_backbones_config_loads_and_lists_five():
    cfg = config.load_backbones_config()
    assert cfg["default"] == "sentence-transformers/all-MiniLM-L6-v2"
    assert len(cfg["candidates"]) == 5
    kinds = {entry["kind"] for entry in cfg["candidates"]}
    assert kinds == {"sentence-encoder", "decoder"}


def test_backbones_listed_in_helpers():
    listed = backbones.list_backbones()
    assert "HuggingFaceTB/SmolLM2-135M-Instruct" in listed
    spec = backbones.get_backbone_spec("HuggingFaceTB/SmolLM2-135M-Instruct")
    assert spec["kind"] == "decoder"


def test_unknown_backbone_raises():
    with pytest.raises(ValueError):
        backbones.get_backbone_spec("does/not-exist")


def test_hardware_config_env_override(monkeypatch):
    monkeypatch.setenv("CUDA_DEVICES", "1,2")
    cfg = config.load_hardware_config()
    assert cfg["cuda_devices"] == [1, 2]
    assert cfg["max_parallel"] == 2


def test_apply_overrides_dotted():
    base = {"a": {"b": 1}, "c": 2}
    out = config.apply_overrides(base, ["a.b=99", "c=3"])
    assert out == {"a": {"b": 99}, "c": 3}
    assert base == {"a": {"b": 1}, "c": 2}  # immutability


def test_repro_set_seed_and_hash():
    repro.set_seed(42)
    h1 = repro.config_hash({"x": 1, "y": [1, 2]})
    h2 = repro.config_hash({"y": [1, 2], "x": 1})
    assert h1 == h2 and len(h1) == 12


def test_repro_env_manifest_keys():
    manifest = repro.env_manifest()
    assert {"timestamp", "platform", "python", "versions", "cuda"} <= manifest.keys()


def test_eval_harness_writes_json_and_band(tmp_path, monkeypatch):
    monkeypatch.setenv("RESULTS_ROOT", str(tmp_path))
    out = eval_harness.run_eval(
        method="m",
        backbone="sentence-transformers/all-MiniLM-L6-v2",
        course="course_test",
        klass="class_test",
        task="t",
        config={"k": 1},
        metrics={"acc": 0.9},
        expected_band={"acc": [0.5, 1.0]},
    )
    assert out.is_file()
    data = json.loads(out.read_text())
    assert data["metrics"]["acc"] == 0.9
    assert data["config_hash"] == repro.config_hash({"k": 1})


def test_eval_harness_band_failure_raises_when_imported(tmp_path, monkeypatch):
    """When called from test (no __main__.__file__), failure raises rather than sys.exit."""
    monkeypatch.setenv("RESULTS_ROOT", str(tmp_path))
    # Force the __main__ module to look like an interactive context.
    import sys

    main_mod = sys.modules["__main__"]
    if hasattr(main_mod, "__file__"):
        monkeypatch.delattr(main_mod, "__file__", raising=False)
    with pytest.raises(eval_harness.MetricBandFailure):
        eval_harness.run_eval(
            method="m",
            backbone="b",
            course="c",
            klass="k",
            task="t",
            config={},
            metrics={"acc": 0.1},
            expected_band={"acc": [0.5, 1.0]},
        )


def test_launcher_expand_grid():
    spec_path = Path(paths.project_root() / "tests" / "_grid_fixture.yaml")
    spec_path.write_text(
        "script: courses/course0_bridge/chapter1_hf_tour/class1_automodel/train.py\n"
        "fixed:\n  --config: courses/course0_bridge/chapter1_hf_tour/class1_automodel/configs/default.yaml\n"
        "grid:\n  backbone: [a, b]\n  seed: [1, 2]\n",
    )
    try:
        lines = launcher.expand(str(spec_path))
        assert len(lines) == 4
        assert all(line.startswith("python ") for line in lines)
    finally:
        spec_path.unlink()
