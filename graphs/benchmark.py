import csv
import time
import random
import importlib.util
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATHS = {
    "S11_classifier": "classifiers/classifier_S11_group_03.pt",
    "Group_A": "classifiers/classifier_S4_group_A.pt",
    "Group_B": "classifiers/classifier_S4_group_B.pt",
    "Group_C": "classifiers/classifier_S4_group_C.pt",
    "Group_D": "classifiers/classifier_S4_group_D.pt",
    "Group_E": "classifiers/classifier_S4_group_E.pt",
    "Group_F": "classifiers/classifier_S4_group_F.pt",
}

ATTACKER_PATHS = {
    "attacker_S11": "attackers/attacker_group_03.py",
    "deepfool": "attackers/attacker_group_03_DeepFool.py"
}

N_TEST = 200
EPS_VALUES = np.linspace(0.0, 2.0, num=11)  # 11 valeurs de epsilon entre 0.0 et 2.0
SEED = 1234
USE_RANDOM_SUBSET = True

REPLACE_INVALID_DELTA_WITH_ZERO = True

L2_TOL = 1e-4
BOX_TOL = 1e-5

OUTPUT_DIR = Path("benchmark_outputs")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)


# ============================================================
# OUTILS GÉNÉRAUX
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def candidate_devices():
    devices = []
    if torch.cuda.is_available():
        devices.append(torch.evice("cuda"))
    devices.append(torch.device("cpu"))
    return devices


def flatten_output(out) :
    out = out.float()

    if out.dim() == 0:
        return out.view(1)
    if out.dim() == 1:
        return out
    if out.dim() == 2 and out.size(0) == 1:
        return out[0]

    return out.reshape(-1)


def infer_input_mode(model, x2d) :
    candidates = [
        (2, x2d),
        (3, x2d.unsqueeze(0)),
        (4, x2d.unsqueeze(0).unsqueeze(0)),
    ]

    with torch.no_grad():
        for mode, xin in candidates:
            try:
                out = model(xin)
                vec = flatten_output(out)
                if vec.numel() == 47:
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


def predict_label(model, mode, x2d) :
    with torch.no_grad():
        scores = forward_model(model, mode, x2d)
        return int(torch.argmax(scores).item())


def load_torchscript_model_auto(path, sample_x_cpu):
    """
    Essaie CUDA puis CPU pour CE modèle.
    Retourne (model, mode, device).
    """
    last_error = None

    for device in candidate_devices():
        try:
            model = torch.jit.load(path, map_location=device)
            model.eval()

            x = sample_x_cpu.to(device)
            mode = infer_input_mode(model, x)

            # test réel
            _ = predict_label(model, mode, x)

            return model, mode, device

        except Exception as e:
            last_error = e
            continue

    raise RuntimeError(
        f"Impossible de charger/utiliser le modèle {path} sur CUDA ou CPU. "
        f"Dernière erreur : {last_error}"
    )


def load_attack_from_file(py_path):
    py_path = str(py_path)
    module_name = f"attack_module_{Path(py_path).stem}_{abs(hash(py_path)) % 10**8}"

    spec = importlib.util.spec_from_file_location(module_name, py_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Impossible de charger le module attaquant : {py_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "attack"):
        raise AttributeError(
            f"Le fichier {py_path} ne contient pas de fonction attack(x, f_string, eps)."
        )

    return module.attack


def coerce_delta(delta, x):
    if isinstance(delta, np.ndarray):
        delta = torch.from_numpy(delta)

    if not torch.is_tensor(delta):
        delta = torch.tensor(delta, dtype=x.dtype)

    delta = delta.detach().clone().float().cpu()

    if delta.dim() == 3 and delta.size(0) == 1:
        delta = delta[0]

    delta = delta.reshape_as(x)
    return delta


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


# ============================================================
# DATASET
# ============================================================

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


def select_indices(n_total, n_keep, seed, use_random_subset: bool):
    n_keep = min(n_keep, n_total)

    if not use_random_subset:
        return list(range(n_keep))

    rng = np.random.default_rng(seed)
    return rng.choice(n_total, size=n_keep, replace=False).tolist()


# ============================================================
# CLEAN
# ============================================================

def precompute_clean_predictions(samples, model_paths):
    clean_preds = {}
    clean_correct_mask = {}
    clean_acc = {}
    model_cache = {}

    print("\n===== CLEAN =====")
    for model_name, model_path in model_paths.items():
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


# ============================================================
# BENCHMARK
# ============================================================

def benchmark_attackers(samples, model_paths, attacker_paths, eps_values):
    clean_preds, clean_correct_mask, clean_acc, model_cache = precompute_clean_predictions(
        samples=samples,
        model_paths=model_paths,
    )

    attackers = {}
    print("\n===== ATTAQUANTS =====")
    for attacker_name, attacker_path in attacker_paths.items():
        attackers[attacker_name] = load_attack_from_file(attacker_path)
        print(f"{attacker_name}: {attacker_path}")

    results = []

    print("\n===== BENCHMARK =====")
    total_configs = len(attackers) * len(model_paths) * len(eps_values)
    config_id = 0

    for attacker_name, attack_fn in attackers.items():
        for model_name, model_path in model_paths.items():
            model, mode, model_device = model_cache[model_name]
            n_clean_correct = int(sum(clean_correct_mask[model_name]))

            for eps in eps_values:
                config_id += 1
                eps = float(eps)

                adv_correct = 0
                success_on_clean_correct = 0
                changed_prediction = 0
                invalid_outputs = 0
                total_attack_time = 0.0

                print(f"[{config_id:>2}/{total_configs}] {attacker_name}/{model_name} eps={eps:.2f}")

                for i, (x_cpu, y_true) in enumerate(samples):
                    clean_pred = clean_preds[model_name][i]

                    t0 = time.time()
                    try:
                        delta = attack_fn(x_cpu.clone(), model_path, eps)
                        delta = coerce_delta(delta, x_cpu)
                        valid, problems, l2 = validate_delta(x_cpu, delta, eps)

                        if not valid:
                            invalid_outputs += 1
                            if REPLACE_INVALID_DELTA_WITH_ZERO:
                                delta = torch.zeros_like(x_cpu)
                            else:
                                raise ValueError(
                                    f"sample {i}: sortie invalide ({'; '.join(problems)})"
                                )

                    except Exception as e:
                        invalid_outputs += 1
                        if REPLACE_INVALID_DELTA_WITH_ZERO:
                            delta = torch.zeros_like(x_cpu)
                        else:
                            raise RuntimeError(
                                f"Erreur attaquant {attacker_name} / modèle {model_name} / sample {i}"
                            ) from e

                    total_attack_time += (time.time() - t0)

                    x_adv = torch.clamp(x_cpu + delta, 0.0, 1.0).to(model_device)
                    pred_adv = predict_label(model, mode, x_adv)

                    if pred_adv == y_true:
                        adv_correct += 1

                    if pred_adv != clean_pred:
                        changed_prediction += 1

                    if clean_pred == y_true and pred_adv != clean_pred:
                        success_on_clean_correct += 1

                adv_acc = adv_correct / len(samples)
                fooling_rate_all = changed_prediction / len(samples)
                success_rate_clean_correct = (
                    success_on_clean_correct / n_clean_correct if n_clean_correct > 0 else 0.0
                )
                mean_attack_time = total_attack_time / len(samples)

                row = {
                    "attacker": attacker_name,
                    "model": model_name,
                    "model_device": model_device.type,
                    "eps": eps,
                    "n_samples": len(samples),
                    "clean_accuracy": clean_acc[model_name],
                    "adv_accuracy": adv_acc,
                    "fooling_rate_all": fooling_rate_all,
                    "attack_success_on_clean_correct": success_rate_clean_correct,
                    "n_clean_correct": n_clean_correct,
                    "invalid_outputs": invalid_outputs,
                    "mean_attack_time_sec": mean_attack_time,
                }
                results.append(row)

                print(
                    f"    adv_acc={adv_acc:.4f} | "
                    f"succès={success_rate_clean_correct:.4f} | "
                    f"invalides={invalid_outputs} | "
                    f"t={mean_attack_time:.3f}s | "
                    f"dev={model_device.type}"
                )

    return results



# ============================================================
# PLOTS
# ============================================================

def plot_results(results, output_dir: Path):
    if not results:
        return

    # Palette rouge et bleu marine
    colors = ["#E63946", "#10009C"]  # Rouge vif et bleu marine
    
    by_attacker = defaultdict(list)
    for row in results:
        by_attacker[row["attacker"]].append(row)

    for attacker_name, rows in by_attacker.items():
        plt.figure(figsize=(6, 4))
        models = sorted({r["model"] for r in rows})

        for idx, model_name in enumerate(models):
            rr = [r for r in rows if r["model"] == model_name]
            rr = sorted(rr, key=lambda z: z["eps"])

            xs = [r["eps"] for r in rr]
            ys = [r["adv_accuracy"] for r in rr]
            color = colors[idx % len(colors)]

            plt.plot(xs, ys, marker="o", label=f"{model_name}", color=color, linewidth=2, markersize=6)

        plt.xlabel("Epsilon (L2)")
        plt.ylabel("Accuracy adversariale")
        plt.title(f"Accuracy vs Epsilon — {attacker_name}")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"benchmark_{attacker_name}.pdf")
        plt.close()

# ============================================================
# TABLEAU FINAL
# ============================================================

def print_summary_table(results):
    if not results:
        return

    print("\n===== RÉSUMÉ FINAL =====")
    header = (
        f"{'Attaquant':<12} {'Modèle':<8} {'dev':<6} {'eps':>6} "
        f"{'adv_acc':>10} {'succès':>10} {'fooling':>10} "
        f"{'invalides':>10} {'t/samp':>10}"
    )
    print(header)
    print("-" * len(header))

    for r in sorted(results, key=lambda z: (z["attacker"], z["model"], z["eps"])):
        print(
            f"{r['attacker']:<12} {r['model']:<8} {r['model_device']:<6} {r['eps']:>6.2f} "
            f"{r['adv_accuracy']:>10.4f} "
            f"{r['attack_success_on_clean_correct']:>10.4f} "
            f"{r['fooling_rate_all']:>10.4f} "
            f"{r['invalid_outputs']:>10d} "
            f"{r['mean_attack_time_sec']:>10.4f}"
        )


# ============================================================
# MAIN
# ============================================================

def main():
    set_seed(SEED)

    print(f"CUDA dispo : {torch.cuda.is_available()}")

    test_dataset = build_emnist_test_dataset()
    indices = select_indices(
        n_total=len(test_dataset),
        n_keep=N_TEST,
        seed=SEED,
        use_random_subset=USE_RANDOM_SUBSET,
    )

    samples = []
    for idx in indices:
        x, y = test_dataset[idx]
        x = x.squeeze(0).float().cpu()   # [28, 28]
        samples.append((x, int(y)))

    print(f"Samples : {len(samples)}")
    print(f"Modèles : {len(MODEL_PATHS)}")
    print(f"Attaquants : {len(ATTACKER_PATHS)}")
    print(f"Eps : {list(np.round(EPS_VALUES, 3))}")

    results = benchmark_attackers(
        samples=samples,
        model_paths=MODEL_PATHS,
        attacker_paths=ATTACKER_PATHS,
        eps_values=EPS_VALUES,
    )

    print_summary_table(results)

    plot_results(results, OUTPUT_DIR)

    print("\n===== FICHIERS =====")

    print(f"Figures : {OUTPUT_DIR}")


if __name__ == "__main__":
    main()