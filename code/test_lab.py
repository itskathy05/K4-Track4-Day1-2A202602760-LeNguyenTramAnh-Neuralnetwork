"""Fast unit/integration checks; full experiments intentionally run on Kaggle."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import openpyxl
import pytest
import torch

CODE_DIR = Path(__file__).resolve().parent
REPO_ROOT = CODE_DIR.parent
sys.path.insert(0, str(CODE_DIR))

from data import apply_standardizer, fit_standardizer, iterate_batches, make_val_split
from model import EXPECTED_PARAMS, MLP, count_params
from optimizer import build_optimizer, clip_gradients
from results_table import load_results, save_result, to_row, write_xlsx
from train import DEFAULT_CFG, compute_loss, macro_f1_from_confusion, run_experiment
from workflow import config_fingerprint


@pytest.mark.parametrize("hidden", list(EXPECTED_PARAMS))
def test_model_shapes_and_parameter_counts(hidden):
    model = MLP(hidden=hidden, dropout=0.2, init="he")
    assert model(torch.randn(8, 54)).shape == (8, 7)
    assert count_params(model) == EXPECTED_PARAMS[hidden]
    assert not any(isinstance(layer, torch.nn.Softmax) for layer in model.modules())


def test_initializers_and_gradients():
    zero = MLP(init="zeros")
    assert all(torch.count_nonzero(m.weight) == 0 for m in zero.modules() if isinstance(m, torch.nn.Linear))
    model = MLP(init="he")
    loss = compute_loss(model(torch.randn(16, 54)), torch.arange(16) % 7, "ce")
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_split_standardization_and_batches():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(700, 54)).astype(np.float32)
    X[:, 10:] = (X[:, 10:] > 0).astype(np.float32)
    y = np.repeat(np.arange(7), 100).astype(np.int64)
    X_tr, y_tr, X_val, y_val = make_val_split(X, y, 0.2, 42)
    mean, std = fit_standardizer(X_tr)
    scaled = apply_standardizer(X_tr, mean, std)
    assert np.allclose(scaled[:, :10].mean(0), 0, atol=1e-5)
    assert np.allclose(scaled[:, :10].std(0), 1, atol=1e-5)
    assert np.array_equal(scaled[:, 10:], X_tr[:, 10:])
    batches = list(iterate_batches(torch.from_numpy(scaled), torch.from_numpy(y_tr), 128,
                                   torch.Generator().manual_seed(1)))
    assert sum(len(yb) for _, yb in batches) == len(y_tr)
    assert len(y_val) == 140


def test_metric_and_gradient_clipping():
    cm = np.eye(7, dtype=np.int64) * 5
    assert macro_f1_from_confusion(cm) == pytest.approx(1.0)
    p = torch.nn.Parameter(torch.tensor([3.0, 4.0]))
    p.grad = torch.tensor([3.0, 4.0])
    before = clip_gradients([p], 1.0)
    assert before == pytest.approx(5.0)
    assert p.grad.norm().item() == pytest.approx(1.0)
    with pytest.raises(ValueError):
        build_optimizer("unknown", [p], 0.1)


def _synthetic_data(seed=5):
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(448, 54, generator=g)
    y = ((X[:, 0] > 0).long() + (X[:, 1] > 0).long() * 2 + (X[:, 2] > 0).long() * 3) % 7
    return {
        "X_tr": X[:320], "y_tr": y[:320], "X_val": X[320:384], "y_val": y[320:384],
        "X_eval": X[384:], "y_eval": y[384:], "eval_row_id": np.arange(64),
    }


def test_small_experiment_is_reproducible():
    cfg = {**DEFAULT_CFG, "exp_id": "smoke", "lr": 0.03, "epochs": 2, "batch": 64,
           "train_eval_size": 320, "seed": 7}
    first = run_experiment(cfg, _synthetic_data())
    second = run_experiment(cfg, _synthetic_data())
    assert first["summary"]["val_macro_f1"] == pytest.approx(second["summary"]["val_macro_f1"])
    assert first["summary"]["n_updates"] == 10


def test_json_and_excel_round_trip(tmp_path):
    result = run_experiment({**DEFAULT_CFG, "exp_id": "roundtrip", "lr": 0.03,
                             "epochs": 1, "batch": 64, "train_eval_size": 100}, _synthetic_data())
    result["fingerprint"] = config_fingerprint(result["cfg"])
    save_result(result, tmp_path / "results")
    loaded = load_results(tmp_path / "results")
    assert loaded[0]["fingerprint"] == result["fingerprint"]
    out = tmp_path / "experiments.xlsx"
    write_xlsx([to_row(loaded[0])], REPO_ROOT / "templates/experiment_table_template.xlsx", out,
               seed_ids=["roundtrip"], summary_notes={"baseline": "smoke"})
    wb = openpyxl.load_workbook(out, data_only=False)
    assert wb.sheetnames == ["Legend", "Experiments", "Seeds", "Summary"]
    assert wb["Experiments"]["A2"].value == "roundtrip"
    assert str(wb["Experiments"]["AD2"].value).startswith("=")
