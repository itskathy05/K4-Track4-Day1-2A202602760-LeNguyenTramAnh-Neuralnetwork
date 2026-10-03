"""Experiment orchestration for the staged Kaggle notebook.

This module contains policy (experiment matrix, resume and validation-only model
selection), while ``train.py`` contains the single training implementation.
Nothing in this module uses eval labels to select a configuration.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import shutil
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from model import EXPECTED_PARAMS, MLP, count_params
from plots import plot_compare, plot_run
from results_table import load_results, save_result
from train import DEFAULT_CFG, compute_loss, evaluate, run_experiment, set_seed


ALLOWED_HIDDEN = ((256, 128), (512, 256), (256, 128, 64))


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in sorted(value.items())}
    if isinstance(value, (tuple, list)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def config_fingerprint(cfg: dict) -> str:
    """Stable fingerprint used to reject stale resume artifacts."""
    payload = json.dumps(_jsonable(cfg), ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _checkpoint_path(checkpoint_dir: str | Path, exp_id: str) -> Path:
    return Path(checkpoint_dir) / f"{exp_id}.pt"


def run_or_resume(cfg: dict, data: dict, results_dir: str | Path,
                  figures_dir: str | Path, checkpoint_dir: str | Path) -> dict:
    """Run exactly one configuration, or load it when config and checkpoint match."""
    cfg = {**DEFAULT_CFG, **copy.deepcopy(cfg)}
    cfg["hidden"] = tuple(cfg["hidden"])
    fingerprint = config_fingerprint(cfg)
    results_dir, figures_dir, checkpoint_dir = map(Path, (results_dir, figures_dir, checkpoint_dir))
    for path in (results_dir, figures_dir, checkpoint_dir):
        path.mkdir(parents=True, exist_ok=True)
    json_path = results_dir / f"{cfg['exp_id']}.json"
    checkpoint_path = _checkpoint_path(checkpoint_dir, cfg["exp_id"])

    if json_path.exists() and checkpoint_path.exists():
        saved = json.loads(json_path.read_text(encoding="utf-8"))
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if saved.get("fingerprint") != fingerprint or checkpoint.get("fingerprint") != fingerprint:
            raise RuntimeError(
                f"Artifact cũ của {cfg['exp_id']} có cấu hình khác. "
                "Đổi exp_id hoặc xoá đúng hai artifact của run này."
            )
        saved["best_state"] = checkpoint.get("best_state")
        print(f"[resume] {cfg['exp_id']}")
        return saved

    print(f"[run] {cfg['exp_id']} | {cfg['description']}")
    result = run_experiment(cfg, data)
    result["fingerprint"] = fingerprint
    save_result(result, str(results_dir))
    plot_run(result, str(figures_dir / f"{cfg['exp_id']}.png"))
    tmp = checkpoint_path.with_suffix(".pt.tmp")
    torch.save({"fingerprint": fingerprint, "best_state": result.get("best_state")}, tmp)
    os.replace(tmp, checkpoint_path)
    return result


def run_many(configs: list[dict], data: dict, results_dir: str | Path,
             figures_dir: str | Path, checkpoint_dir: str | Path) -> list[dict]:
    return [run_or_resume(c, data, results_dir, figures_dir, checkpoint_dir) for c in configs]


def select_best(results: list[dict]) -> dict:
    """Validation-only ranking: macro-F1, then loss, then smaller learning rate."""
    eligible = [r for r in results if not r["summary"].get("diverged") and r["summary"].get("val_macro_f1") is not None]
    if not eligible:
        raise ValueError("Không có run hợp lệ để chọn")
    return sorted(
        eligible,
        key=lambda r: (
            -float(r["summary"]["val_macro_f1"]),
            float(r["summary"]["best_val_loss"]),
            float(r["cfg"]["lr"]),
        ),
    )[0]


def seed_statistics(results: list[dict]) -> dict:
    f1 = np.asarray([r["summary"]["val_macro_f1"] for r in results], dtype=float)
    acc = np.asarray([r["summary"]["val_acc"] for r in results], dtype=float)
    ddof = 1 if len(f1) > 1 else 0
    return {
        "n": int(len(f1)), "f1_mean": float(f1.mean()), "f1_std": float(f1.std(ddof=ddof)),
        "acc_mean": float(acc.mean()), "acc_std": float(acc.std(ddof=ddof)),
        "noise_2sigma": float(2 * f1.std(ddof=ddof)),
    }


def baseline_sweep_configs() -> list[dict]:
    return [{
        **DEFAULT_CFG, "exp_id": f"base-lr{str(lr).replace('.', 'p')}-s1",
        "group": "baseline", "description": f"Dò learning rate baseline: {lr}",
        "lr": lr, "seed": 1,
    } for lr in (0.01, 0.03, 0.1)]


def official_baseline_configs(selected_lr: float) -> list[dict]:
    return [{
        **DEFAULT_CFG, "exp_id": f"base-s{seed}", "group": "baseline",
        "description": f"Baseline chính thức, seed {seed}", "lr": selected_lr, "seed": seed,
    } for seed in (1, 2, 3)]


def optimizer_configs(base: dict) -> list[dict]:
    specs = {
        "sgd": ((0.01, 0.03, 0.1), 0.0),
        "adam": ((3e-4, 1e-3, 3e-3), 0.0),
        "adamw": ((3e-4, 1e-3, 3e-3), 0.01),
    }
    configs = []
    for optimizer, (lrs, weight_decay) in specs.items():
        for lr in lrs:
            token = f"{lr:g}".replace(".", "p").replace("-", "m")
            configs.append({
                **base, "exp_id": f"opt-{optimizer}-lr{token}", "group": "optimizer",
                "description": f"{optimizer} với lr={lr:g}", "optimizer": optimizer,
                "lr": lr, "weight_decay": weight_decay, "seed": 1,
            })
    return configs


def topic_configs(base: dict, clip_norm: float, bf16_supported: bool) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = {}
    groups["loss"] = [
        {**base, "exp_id": "loss-mse", "group": "loss", "description": "MSE với nhãn one-hot", "loss": "mse"},
        {**base, "exp_id": "loss-ce-balanced", "group": "loss", "description": "CE trọng số nghịch đảo tần suất", "class_weight": "balanced"},
        {**base, "exp_id": "loss-ce-sqrt", "group": "loss", "description": "CE trọng số căn nghịch đảo tần suất", "class_weight": "sqrt_balanced"},
    ]
    groups["hparam"] = [
        {**base, "exp_id": "batch-128", "group": "hparam", "description": "Batch size 128", "batch": 128},
        {**base, "exp_id": "batch-2048", "group": "hparam", "description": "Batch size 2048", "batch": 2048},
        {**base, "exp_id": "arch-wide", "group": "hparam", "description": "M-wide 54-512-256-7", "hidden": (512, 256)},
        {**base, "exp_id": "arch-deep", "group": "hparam", "description": "M-deep 54-256-128-64-7", "hidden": (256, 128, 64)},
        {**base, "exp_id": "epochs-40", "group": "hparam", "description": "Baseline 40 epoch", "epochs": 40},
    ]
    groups["dropout"] = [{
        **base, "exp_id": f"drop-{str(q).replace('.', 'p')}", "group": "dropout",
        "description": f"Dropout q={q}", "dropout": q,
    } for q in (0.1, 0.3, 0.5)]
    groups["clipping"] = [
        {**base, "exp_id": "clip-normal", "group": "clipping", "description": f"Clip c={clip_norm:.4g} ở lr baseline", "clip_norm": clip_norm},
        {**base, "exp_id": "highlr-no-clip", "group": "clipping", "description": "Learning rate x10, không clip", "lr": base["lr"] * 10, "clip_norm": None},
        {**base, "exp_id": "highlr-clip", "group": "clipping", "description": f"Learning rate x10, clip c={clip_norm:.4g}", "lr": base["lr"] * 10, "clip_norm": clip_norm},
    ]
    groups["amp"] = [{
        **base, "exp_id": "amp-fp16", "group": "amp", "description": "Mixed precision FP16", "precision": "fp16",
    }]
    if bf16_supported:
        groups["amp"].append({
            **base, "exp_id": "amp-bf16", "group": "amp", "description": "Mixed precision BF16", "precision": "bf16",
        })
    groups["init"] = [{
        **base, "exp_id": f"init-{init}", "group": "init", "description": f"Khởi tạo {init}", "init": init,
    } for init in ("zeros", "normal", "xavier", "default")]
    return groups


def make_comparison_figures(results_dir: str | Path, figures_dir: str | Path) -> None:
    results = load_results(str(results_dir))
    groups: dict[str, list[dict]] = {}
    baseline = next((r for r in results if r["cfg"].get("exp_id") == "base-s1"), None)
    for result in results:
        if result["history"].get("epoch"):
            groups.setdefault(result["cfg"].get("group", "other"), []).append(result)
    for group, items in groups.items():
        if group not in {"baseline", "final"} and baseline is not None:
            items = [baseline, *items]
        if len(items) >= 2:
            plot_compare(
                items, "val_macro_f1", str(Path(figures_dir) / f"compare_{group}.png"),
                title=f"{group}: validation macro-F1",
            )


def build_candidate_configs(all_results: list[dict], baseline_cfg: dict,
                            noise_2sigma: float) -> tuple[dict, dict]:
    """Build two candidates using validation results only and deterministic tie rules."""
    valid = [r for r in all_results if r["summary"].get("val_macro_f1") is not None and not r["summary"].get("diverged")]
    excluded = {"baseline", "amp", "clipping", "init", "final"}
    singles = [r for r in valid if r["cfg"].get("group") not in excluded and r["cfg"].get("init") != "zeros"]
    ranked = sorted(singles or valid, key=lambda r: -r["summary"]["val_macro_f1"])
    candidate_a = {**ranked[0]["cfg"], "precision": "fp32", "seed": 1}

    combined = copy.deepcopy(baseline_cfg)
    combined.update({"precision": "fp32", "seed": 1})
    baseline_f1 = float(next(r for r in valid if r["cfg"]["exp_id"] == "base-s1")["summary"]["val_macro_f1"])

    def best_group(group):
        items = [r for r in valid if r["cfg"].get("group") == group]
        return max(items, key=lambda r: r["summary"]["val_macro_f1"]) if items else None

    best_optimizer = best_group("optimizer")
    optimizer_pool = [r for r in valid if r["cfg"].get("group") == "baseline" and r["cfg"]["exp_id"].startswith("base-lr")]
    if optimizer_pool:
        optimizer_pool.append(best_optimizer) if best_optimizer else None
        best_optimizer = max(optimizer_pool, key=lambda r: r["summary"]["val_macro_f1"])
    if best_optimizer:
        for key in ("optimizer", "lr", "weight_decay"):
            combined[key] = best_optimizer["cfg"][key]

    best_hparam = best_group("hparam")
    if best_hparam and best_hparam["cfg"]["exp_id"] in {"arch-wide", "arch-deep"}:
        combined["hidden"] = tuple(best_hparam["cfg"]["hidden"])
    epoch40 = next((r for r in valid if r["cfg"]["exp_id"] == "epochs-40"), None)
    if epoch40 and (epoch40["summary"]["val_macro_f1"] - baseline_f1 > noise_2sigma or epoch40["summary"]["best_epoch"] >= 36):
        combined["epochs"] = 40

    for group, keys in (("loss", ("loss", "class_weight")), ("dropout", ("dropout",)), ("init", ("init",))):
        best = best_group(group)
        if best and best["summary"]["val_macro_f1"] - baseline_f1 > noise_2sigma:
            for key in keys:
                combined[key] = best["cfg"].get(key, combined.get(key))

    normal_clip = next((r for r in valid if r["cfg"]["exp_id"] == "clip-normal"), None)
    high_no = next((r for r in valid if r["cfg"]["exp_id"] == "highlr-no-clip"), None)
    high_yes = next((r for r in valid if r["cfg"]["exp_id"] == "highlr-clip"), None)
    if normal_clip and high_no and high_yes:
        rescued = high_yes["summary"]["val_macro_f1"] - high_no["summary"]["val_macro_f1"] > noise_2sigma
        harmless = normal_clip["summary"]["val_macro_f1"] >= baseline_f1 - noise_2sigma
        if rescued and harmless:
            combined["clip_norm"] = normal_clip["cfg"]["clip_norm"]

    def clean(cfg, label):
        cfg = {**DEFAULT_CFG, **cfg}
        cfg.update({"exp_id": label, "group": "final", "description": label.replace("-", " ").title()})
        return cfg

    candidate_a = clean(candidate_a, "candidate-a-s1")
    candidate_b = clean(combined, "candidate-b-s1")
    comparable_keys = [k for k in DEFAULT_CFG if k not in {"exp_id", "group", "description", "seed"}]
    if all(_jsonable(candidate_a[k]) == _jsonable(candidate_b[k]) for k in comparable_keys):
        fallback = ranked[1] if len(ranked) > 1 else valid[0]
        candidate_b = clean(fallback["cfg"], "candidate-b-s1")
    return candidate_a, candidate_b


def expand_candidate_seeds(candidate_cfg: dict, letter: str) -> list[dict]:
    return [{
        **candidate_cfg, "exp_id": f"candidate-{letter}-s{seed}", "group": "final",
        "description": f"Candidate {letter.upper()}, seed {seed}", "seed": seed,
    } for seed in (1, 2, 3)]


def choose_final_candidate(a_results: list[dict], b_results: list[dict],
                           baseline_noise: float) -> tuple[str, list[dict], dict]:
    stats_a, stats_b = seed_statistics(a_results), seed_statistics(b_results)
    threshold = max(float(baseline_noise), 0.002)
    delta = stats_a["f1_mean"] - stats_b["f1_mean"]
    if abs(delta) > threshold:
        winner = "a" if delta > 0 else "b"
    elif stats_a["f1_std"] != stats_b["f1_std"]:
        winner = "a" if stats_a["f1_std"] < stats_b["f1_std"] else "b"
    else:
        def complexity(r):
            c = r[0]["cfg"]
            deviations = sum(c.get(k) != DEFAULT_CFG.get(k) for k in DEFAULT_CFG if k not in {"exp_id", "group", "description", "lr", "seed"})
            return deviations, r[0]["summary"].get("time_per_epoch_s") or math.inf
        winner = "a" if complexity(a_results) <= complexity(b_results) else "b"
    chosen = a_results if winner == "a" else b_results
    return winner, chosen, {"candidate_a": stats_a, "candidate_b": stats_b, "threshold": threshold}


def run_health_checks(data: dict, figure_path: str | Path, seed: int = 42) -> dict:
    """Required shape/step-0/gradient/small-batch-overfit checks."""
    set_seed(seed)
    device = data["X_tr"].device
    model = MLP().to(device)
    param_count = count_params(model)
    assert param_count == EXPECTED_PARAMS[(256, 128)]
    with torch.no_grad():
        shape = tuple(model(torch.randn(8, 54, device=device)).shape)
    step0 = evaluate(model, data["X_val"], data["y_val"])["loss"]
    model.train()
    logits = model(data["X_tr"][:32])
    loss = compute_loss(logits, data["y_tr"][:32], "ce")
    loss.backward()
    gradient_norms = {name: float(p.grad.norm().item()) if p.grad is not None else None for name, p in model.named_parameters()}
    if any(v is None or v == 0 for v in gradient_norms.values()):
        raise AssertionError("Có tham số không nhận gradient")

    set_seed(seed)
    tiny = MLP(dropout=0.0).to(device)
    optimizer = torch.optim.Adam(tiny.parameters(), lr=0.01)
    X20, y20 = data["X_tr"][:20], data["y_tr"][:20]
    losses = []
    final_acc = 0.0
    for _ in range(1000):
        optimizer.zero_grad(set_to_none=True)
        tiny_loss = compute_loss(tiny(X20), y20, "ce")
        tiny_loss.backward()
        optimizer.step()
        losses.append(float(tiny_loss.item()))
        with torch.no_grad():
            final_acc = float((tiny(X20).argmax(1) == y20).float().mean().item())
        if losses[-1] < 1e-3 and final_acc == 1.0:
            break
    if final_acc != 1.0 or losses[-1] >= 0.01:
        raise AssertionError(f"Không overfit được 20 mẫu: loss={losses[-1]}, acc={final_acc}")
    figure_path = Path(figure_path)
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(losses)
    ax.set(title="Overfit 20 samples", xlabel="Update", ylabel="Cross-entropy loss", yscale="log")
    ax.grid(alpha=0.2)
    fig.savefig(figure_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return {
        "parameter_count": param_count, "logits_shape": list(shape), "step0_loss": step0,
        "ln7": math.log(7), "gradient_norms": gradient_norms,
        "overfit_steps": len(losses), "overfit_final_loss": losses[-1], "overfit_final_acc": final_acc,
    }


def save_stage_archive(stage_name: str, submission_dir: str | Path,
                       checkpoint_dir: str | Path, out_dir: str | Path) -> Path:
    """Archive resumable artifacts. Checkpoints are never placed inside submission."""
    submission_dir, checkpoint_dir, out_dir = map(Path, (submission_dir, checkpoint_dir, out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    archive = out_dir / f"artifacts_{stage_name}.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
        for folder in (submission_dir / "results", submission_dir / "figures", checkpoint_dir):
            if folder.exists():
                root_name = folder.name if folder != checkpoint_dir else "runtime_checkpoints"
                for file in folder.rglob("*"):
                    if file.is_file():
                        zf.write(file, Path(root_name) / file.relative_to(folder))
        for name in ("health_checks.json", "selection_state.json"):
            file = submission_dir / name
            if file.exists():
                zf.write(file, Path(name))
    return archive
