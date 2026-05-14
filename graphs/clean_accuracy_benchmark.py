"""
clean_accuracy.py
=================
Évalue la clean accuracy de tous les modèles .pt dans le dossier classifiers/
sur 1000 images EMNIST.
"""

import random
from pathlib import Path

import numpy as np
import torch
from torchvision import datasets, transforms


N_TEST            = 1000
SEED              = 1234
CLASSIFIERS_DIR   = Path("classifiers")


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)

def flatten(out):
    out = out.float()
    if out.dim() <= 1:       return out.reshape(-1)
    if out.dim() == 2 and out.size(0) == 1: return out[0]
    return out.reshape(-1)

def infer_mode(model, x):
    for mode, xin in [(2, x), (3, x.unsqueeze(0)), (4, x.unsqueeze(0).unsqueeze(0))]:
        try:
            if flatten(model(xin)).numel() == 47:
                return mode
        except Exception:
            pass
    return 4

def forward(model, mode, x):
    if   mode == 2: return flatten(model(x))
    elif mode == 3: return flatten(model(x.unsqueeze(0)))
    else:           return flatten(model(x.unsqueeze(0).unsqueeze(0)))

def load_model(path, sample_x):
    devs = ([torch.device("cuda")] if torch.cuda.is_available() else []) + [torch.device("cpu")]
    for dev in devs:
        try:
            m = torch.jit.load(str(path), map_location=dev); m.eval()
            x = sample_x.to(dev)
            mode = infer_mode(m, x)
            with torch.no_grad(): forward(m, mode, x)
            return m, mode, dev
        except Exception:
            continue
    raise RuntimeError(f"Impossible de charger {path}")

def build_dataset():
    t = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: torch.rot90(x, k=3, dims=[1, 2])),
        transforms.Lambda(lambda x: torch.flip(x, dims=[2])),
    ])
    return datasets.EMNIST(root="dataset/", split="balanced",
                           train=False, download=True, transform=t)


def main():
    set_seed(SEED)

    dataset = build_dataset()
    rng     = np.random.default_rng(SEED)
    indices = rng.choice(len(dataset), size=min(N_TEST, len(dataset)), replace=False)
    samples = [(dataset[i][0].squeeze(0).float().cpu(), int(dataset[i][1])) for i in indices]

    model_files = sorted(CLASSIFIERS_DIR.glob("*.pt"))
    if not model_files:
        print(f"Aucun fichier .pt trouvé dans {CLASSIFIERS_DIR}/"); return

    print(f"\n{len(samples)} images  —  {len(model_files)} modèles\n")
    print(f"{'Modèle':<20} {'Device':<6} {'Accuracy':>10}  {'Correct':>8}")
    print("─" * 50)

    for path in model_files:
        model, mode, dev = load_model(path, samples[0][0])
        correct = 0
        with torch.no_grad():
            for x, y in samples:
                scores = forward(model, mode, x.to(dev))
                if int(torch.argmax(scores)) == y:
                    correct += 1
        acc = correct / len(samples)
        print(f"{path.stem:<20} {dev.type:<6} {acc:>10.4f}  {correct:>6}/{len(samples)}")

    print("─" * 50)


if __name__ == "__main__":
    main()