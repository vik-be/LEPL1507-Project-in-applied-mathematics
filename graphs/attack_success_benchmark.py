from __future__ import annotations

import ast
import csv
import importlib.util
import random
import time
import types
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import datasets, transforms


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT
# OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# Update these paths to the attackers you want to compare.
ATTACKER_PATHS = {
    "attacker_S11": "attackers/attacker_group_03.py",
    "deepfool": "attackers/attacker_group_03_DeepFool.py"
}

# Update these target models as needed.
MODEL_PATHS = {
    "S11_classifier": "classifiers/classifier_S11_group_03.pt",
    "Group_A": "classifiers/classifier_S4_group_A.pt",
    "Group_B": "classifiers/classifier_S4_group_B.pt",
    "Group_C": "classifiers/classifier_S4_group_C.pt",
    "Group_D": "classifiers/classifier_S4_group_D.pt",
    "Group_E": "classifiers/classifier_S4_group_E.pt",
    "Group_F": "classifiers/classifier_S4_group_F.pt",
}

EPS_VALUES = np.linspace(0.0, 2.0, num=11)
N_TEST = 200
SEED = 1234
USE_RANDOM_SUBSET = True
REPLACE_INVALID_DELTA_WITH_ZERO = True
L2_TOL = 1e-4
BOX_TOL = 1e-5


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def candidate_devices():
    devices = []
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))
    devices.append(torch.device("cpu"))
    return devices


def flatten_output(out: torch.Tensor) -> torch.Tensor:
    out = out.float()
    if out.dim() == 0:
        return out.view(1)
    if out.dim() == 1:
        return out
    if out.dim() == 2 and out.size(0) == 1:
        return out[0]
    return out.reshape(-1)


def infer_input_mode(model, x2d: torch.Tensor) -> int:
    candidates = [
        (2, x2d),
        (3, x2d.unsqueeze(0)),
        (4, x2d.unsqueeze(0).unsqueeze(0)),
    ]

    with torch.no_grad():
        for mode, xin in candidates:
            try:
                out = model(xin)
                if flatten_output(out).numel() == 47:
                    return mode
            except Exception:
                pass

    return 4


def forward_model(model, mode, x2d):
    if mode == 2:
        out = model(x2d)
    elif mode == 3:
        out = model(x2d.unsqueeze(0))
    else:
        out = model(x2d.unsqueeze(0).unsqueeze(0))
    return flatten_output(out)


def predict_label(model, mode, x2d):
    with torch.no_grad():
        scores = forward_model(model, mode, x2d)
        return int(torch.argmax(scores).item())


def load_torchscript_model_auto(path, sample_x_cpu):
    last_error = None
    for device in candidate_devices():
        try:
            model = torch.jit.load(path, map_location=device)
            model.eval()
            x = sample_x_cpu.to(device)
            mode = infer_input_mode(model, x)
            with torch.no_grad():
                _ = predict_label(model, mode, x)
            return model, mode, device
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"Impossible de charger/utiliser le modèle {path}: {last_error}")


def load_attack_from_file(py_path):
    py_path = str(py_path)
    module_name = f"attack_module_{Path(py_path).stem}_{abs(hash(py_path)) % 10**8}"
    spec = importlib.util.spec_from_file_location(module_name, py_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Impossible de charger le module attaquant : {py_path}")

    source_path = Path(py_path)
    source_code = source_path.read_text(encoding="utf-8")
    parsed = ast.parse(source_code, filename=py_path)

    safe_body = []
    for node in parsed.body:
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            safe_body.append(node)
            continue

        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            safe_body.append(node)

    safe_module = ast.Module(body=safe_body, type_ignores=[])
    ast.fix_missing_locations(safe_module)

    module = importlib.util.module_from_spec(spec)
    exec(compile(safe_module, py_path, "exec"), module.__dict__)

    if not hasattr(module, "attack"):
        raise AttributeError(
            f"Le fichier {py_path} ne contient pas de fonction attack(x, f_string, eps)."
        )

    return module.attack


def coerce_delta(delta, x) :
    if isinstance(delta, np.ndarray):
        delta = torch.from_numpy(delta)
    if not torch.is_tensor(delta):
        delta = torch.tensor(delta, dtype=x.dtype)

    delta = delta.detach().clone().float().cpu()
    if delta.dim() == 3 and delta.size(0) == 1:
        delta = delta[0]
    return delta.reshape_as(x)


def validate_delta(x, delta, eps):
    problems = []

    if delta.shape != x.shape:
        problems.append(f"shape={tuple(delta.shape)} au lieu de {tuple(x.shape)}")

    if not torch.isfinite(delta).all():
        problems.append("NaN/Inf")

    l2 = float(torch.norm(delta.reshape(-1), p=2).item())
    if l2 > eps + L2_TOL:
        problems.append(f"L2={l2:.6f}>{eps:.6f}")

    x_adv = x + delta
    xmin = float(x_adv.min().item())
    xmax = float(x_adv.max().item())
    if xmin < -BOX_TOL or xmax > 1.0 + BOX_TOL:
        problems.append(f"x+delta hors [0,1] ({xmin:.4f},{xmax:.4f})")

    return (len(problems) == 0), problems, l2


def build_emnist_test_dataset():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: torch.rot90(x, k=3, dims=[1, 2])),
        transforms.Lambda(lambda x: torch.flip(x, dims=[2])),
    ])

    return datasets.EMNIST(
        root="dataset/",
        split="balanced",
        train=False,
        download=True,
        transform=transform,
    )


def select_indices(n_total, n_keep, seed, use_random_subset):
    n_keep = min(n_keep, n_total)
    if not use_random_subset:
        return list(range(n_keep))
    rng = np.random.default_rng(seed)
    return rng.choice(n_total, size=n_keep, replace=False).tolist()


def precompute_clean_predictions(samples, model_paths):
    clean_preds = {}
    clean_correct_mask = {}
    clean_acc = {}
    model_cache = {}

    print("\n===== CLEAN =====")
    for model_name, model_path in model_paths.items():
        if not Path(model_path).exists():
            print(f"Skipping missing model: {model_name} -> {model_path}")
            continue

        model, mode, device = load_torchscript_model_auto(model_path, samples[0][0])

        preds = []
        correct_mask = []
        for x_cpu, y_true in samples:
            pred = predict_label(model, mode, x_cpu.to(device))
            preds.append(pred)
            correct_mask.append(pred == y_true)

        acc = float(np.mean(correct_mask))
        clean_preds[model_name] = preds
        clean_correct_mask[model_name] = correct_mask
        clean_acc[model_name] = acc
        model_cache[model_name] = (model, mode, device)

        print(f"{model_name}: device={device.type} | mode={mode}D | clean_acc={acc:.4f}")

    return clean_preds, clean_correct_mask, clean_acc, model_cache


def benchmark_attackers(samples, model_paths, attacker_paths, eps_values):
    clean_preds, clean_correct_mask, clean_acc, model_cache = precompute_clean_predictions(
        samples=samples,
        model_paths=model_paths,
    )

    if not model_cache:
        raise RuntimeError("No valid models were loaded. Check files in pareto_models/.")

    attackers = {}
    print("\n===== ATTAQUANTS =====")
    for attacker_name, attacker_path in attacker_paths.items():
        attackers[attacker_name] = load_attack_from_file(attacker_path)
        print(f"{attacker_name}: {attacker_path}")

    results = []
    total_configs = len(attackers) * len(model_cache) * len(eps_values)
    config_id = 0

    print("\n===== BENCHMARK =====")
    for attacker_name, attack_fn in attackers.items():
        for model_name, model_path in model_paths.items():
            if model_name not in model_cache:
                continue

            model, mode, model_device = model_cache[model_name]
            n_clean_correct = int(sum(clean_correct_mask[model_name]))

            for eps in eps_values:
                config_id += 1
                eps = float(eps)

                adv_correct = 0
                success_on_clean_correct = 0
                invalid_outputs = 0
                total_attack_time = 0.0

                print(f"[{config_id:>2}/{total_configs}] {attacker_name}/{model_name} eps={eps:.2f}")

                for i, (x_cpu, y_true) in enumerate(samples):
                    clean_pred = clean_preds[model_name][i]

                    t0 = time.time()
                    try:
                        delta = attack_fn(x_cpu.clone(), model_path, eps)
                        delta = coerce_delta(delta, x_cpu)
                        valid, problems, _ = validate_delta(x_cpu, delta, eps)
                        if not valid:
                            invalid_outputs += 1
                            if REPLACE_INVALID_DELTA_WITH_ZERO:
                                delta = torch.zeros_like(x_cpu)
                            else:
                                raise ValueError(f"sample {i}: sortie invalide ({'; '.join(problems)})")
                    except Exception as exc:
                        invalid_outputs += 1
                        if REPLACE_INVALID_DELTA_WITH_ZERO:
                            delta = torch.zeros_like(x_cpu)
                        else:
                            raise RuntimeError(
                                f"Erreur attaquant {attacker_name} / modèle {model_name} / sample {i}"
                            ) from exc

                    total_attack_time += (time.time() - t0)

                    x_adv = torch.clamp(x_cpu + delta, 0.0, 1.0).to(model_device)
                    pred_adv = predict_label(model, mode, x_adv)

                    if pred_adv == y_true:
                        adv_correct += 1
                    if clean_pred == y_true and pred_adv != clean_pred:
                        success_on_clean_correct += 1

                adv_acc = adv_correct / len(samples)
                success_rate_clean_correct = (
                    success_on_clean_correct / n_clean_correct if n_clean_correct > 0 else 0.0
                )
                mean_attack_time = total_attack_time / len(samples)

                results.append({
                    "attacker": attacker_name,
                    "model": model_name,
                    "model_device": model_device.type,
                    "eps": eps,
                    "n_samples": len(samples),
                    "clean_accuracy": clean_acc[model_name],
                    "adv_accuracy": adv_acc,
                    "success_rate_clean_correct": success_rate_clean_correct,
                    "n_clean_correct": n_clean_correct,
                    "invalid_outputs": invalid_outputs,
                    "mean_attack_time_sec": mean_attack_time,
                })

                print(
                    f"    success={success_rate_clean_correct:.4f} | "
                    f"invalid={invalid_outputs} | t={mean_attack_time:.3f}s | dev={model_device.type}"
                )

    return results





def plot_success_vs_eps(results, output_dir: Path):
    if not results:
        return

    by_model = defaultdict(list)
    for row in results:
        by_model[row["model"]].append(row)

    for model_name, rows in by_model.items():
        plt.figure(figsize=(7, 4.8))
        attackers = sorted({r["attacker"] for r in rows})
        colors = ["#E63946", "#10009C", "#2A9D8F", "#F4A261", "#6C5CE7", "#0F766E"]

        for idx, attacker_name in enumerate(attackers):
            rr = sorted((r for r in rows if r["attacker"] == attacker_name), key=lambda z: z["eps"])
            xs = [r["eps"] for r in rr]
            ys = [r["success_rate_clean_correct"] * 100 for r in rr]
            plt.plot(
                xs,
                ys,
                marker="o",
                linewidth=2,
                markersize=5,
                color=colors[idx % len(colors)],
                label=attacker_name,
            )

        plt.xlabel("Epsilon (L2)")
        plt.ylabel("Attack success rate on clean-correct samples (%)")
        plt.title(f"Attack success vs epsilon — {model_name}")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        out_path = output_dir / f"attack_success_vs_epsilon_{model_name}.pdf"
        plt.savefig(out_path, bbox_inches="tight")
        plt.close()
        print(f"Saved figure to {out_path}")


def print_summary_table(results):
    if not results:
        return

    print("\n===== RÉSUMÉ FINAL =====")
    header = (
        f"{'Attaquant':<16} {'Modèle':<16} {'eps':>6} {'clean':>8} {'succès':>8} "
        f"{'invalides':>10} {'t/samp':>10}"
    )
    print(header)
    print("-" * len(header))

    for r in sorted(results, key=lambda z: (z["attacker"], z["model"], z["eps"])):
        print(
            f"{r['attacker']:<16} {r['model']:<16} {r['eps']:>6.2f} "
            f"{r['clean_accuracy']*100:>7.2f}% {r['success_rate_clean_correct']*100:>7.2f}% "
            f"{r['invalid_outputs']:>10d} {r['mean_attack_time_sec']:>10.4f}"
        )


def main():
    set_seed(SEED)

    print("Chargement EMNIST Balanced…")
    dataset = build_emnist_test_dataset()
    indices = select_indices(
        n_total=len(dataset),
        n_keep=N_TEST,
        seed=SEED,
        use_random_subset=USE_RANDOM_SUBSET,
    )

    samples = []
    for idx in indices:
        x, y = dataset[idx]
        samples.append((x.squeeze(0).float().cpu(), int(y)))

    print(f"Samples: {len(samples)}")
    print(f"Modèles: {len(MODEL_PATHS)}")
    print(f"Attaquants: {len(ATTACKER_PATHS)}")
    print(f"Epsilons: {list(np.round(EPS_VALUES, 3))}")

    results = benchmark_attackers(
        samples=samples,
        model_paths=MODEL_PATHS,
        attacker_paths=ATTACKER_PATHS,
        eps_values=EPS_VALUES,
    )

    print_summary_table(results)

    plot_success_vs_eps(results, OUTPUT_DIR)

    print("\n===== FICHIERS =====")
    print(f"Figures : {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
