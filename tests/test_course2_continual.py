"""Tests for course2_continual — continual learning suite.

Industry standard: unit tests for the buffer, math checks for BWT,
config validation, and regression guard that save_strategy != "no".
"""
import random
import sys
from pathlib import Path
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
CH2 = REPO / "courses/course2_continual/chapter2_replay/class1_simple_rehearsal"
sys.path.insert(0, str(CH2))


def _import_buffer():
    """Import ReplayBuffer from train.py."""
    import importlib.util
    mod_name = "train_ch2"
    if mod_name in sys.modules:
        return sys.modules[mod_name].ReplayBuffer
    spec = importlib.util.spec_from_file_location(mod_name, CH2 / "train.py")
    mod = importlib.util.module_from_spec(spec)
    # Register before exec so @dataclass can resolve __module__ correctly.
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod.ReplayBuffer


class TestReplayBuffer:
    def test_capacity_never_exceeded(self):
        B = _import_buffer()
        buf = B(capacity=10)
        buf.add([{"x": i} for i in range(100)])
        assert len(buf._buf) <= 10

    def test_fill_rate_full(self):
        B = _import_buffer()
        buf = B(capacity=10)
        buf.add([{"x": i} for i in range(50)])
        assert buf.fill_rate() == 1.0

    def test_fill_rate_partial(self):
        B = _import_buffer()
        buf = B(capacity=10)
        buf.add([{"x": i} for i in range(5)])
        assert buf.fill_rate() == 0.5

    def test_sample_length(self):
        B = _import_buffer()
        buf = B(capacity=20)
        buf.add([{"x": i} for i in range(20)])
        s = buf.sample(5)
        assert len(s) == 5

    def test_sample_empty(self):
        B = _import_buffer()
        buf = B(capacity=10)
        assert buf.sample(5) == []

    def test_sample_less_than_n(self):
        B = _import_buffer()
        buf = B(capacity=10)
        buf.add([{"x": i} for i in range(3)])
        s = buf.sample(10)
        assert len(s) == 3


def test_bwt_negative_means_forgetting():
    """BWT < 0 means the model forgot previous tasks after fine-tuning."""
    acc_matrix = [[0.9, None], [0.7, 0.85]]
    bwt = acc_matrix[1][0] - acc_matrix[0][0]
    assert bwt < 0, "BWT should be negative when forgetting occurred"
    assert abs(bwt - (-0.2)) < 1e-6


def test_bwt_zero_means_no_forgetting():
    """BWT = 0 means perfect retention of previous task."""
    acc_matrix = [[0.85, None], [0.85, 0.80]]
    bwt = acc_matrix[1][0] - acc_matrix[0][0]
    assert abs(bwt) < 1e-6


def test_save_strategy_not_disabled():
    """All course2 train.py files must not use save_strategy='no'."""
    import glob
    for train_py in glob.glob(str(REPO / "courses/course2_continual/**/train.py"), recursive=True):
        content = open(train_py).read()
        assert 'save_strategy="no"' not in content and "save_strategy='no'" not in content, (
            f"{train_py} still uses save_strategy='no' — checkpointing is disabled"
        )


def test_no_raw_print_in_hot_path():
    """EWC train.py hot path should use structured logger, not print()."""
    ewc_train = REPO / "courses/course2_continual/chapter3_regularization/class1_ewc/train.py"
    if not ewc_train.exists():
        pytest.skip("EWC train.py not found")
    content = ewc_train.read_text()
    # Allow print() only outside the training loop
    # Check for the specific hot-path print pattern the audit found
    assert 'print(f"[EWC]' not in content, "Hot-path print() found in EWC trainer — use get_logger"


def test_chapter2_configs_exist():
    """smoke.yaml and default.yaml must exist for chapter2."""
    smoke = CH2 / "configs/smoke.yaml"
    default = CH2 / "configs/default.yaml"
    assert smoke.exists(), "smoke.yaml missing in chapter2_replay"
    assert default.exists(), "default.yaml missing in chapter2_replay"


def test_chapter2_run_sh_exists_and_executable():
    """run.sh must exist and be executable."""
    import os
    run_sh = CH2 / "run.sh"
    assert run_sh.exists(), "run.sh missing in chapter2_replay"
    assert os.access(run_sh, os.X_OK), "run.sh not executable — run chmod +x"
