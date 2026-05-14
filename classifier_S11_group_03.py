import json
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


batch_size = 128
device = "cuda" if torch.cuda.is_available() else "cpu"
metrics_path = Path("training_metrics.json")


class ClassifierS11GroupXX(nn.Module):
    """
    Modèle robuste de classification d'images.

    Contrainte: Le classifieur génère un vecteur de probabilités en moins de 0.1 seconde pour une image, sur un ordinateur
                personnel de base (équipé par exemple d’un processeur CPU de type Core i5 et de 8 Go de mémoire RAM).

    Export: Après avoir entrainé le modèle, utilisez le code suivant afin de le sauver comme un oracle:

            model = ClassifierS11GroupXX()
            m = torch.jit.script(model)
            m.save("classifier_S11_group_XX.pt")
    """

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.gn1 = nn.GroupNorm(8, 32)

        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.gn2 = nn.GroupNorm(8, 32)

        self.conv3 = nn.Conv2d(32, 32, 3, padding=1)
        self.gn3 = nn.GroupNorm(8, 32)

        self.conv4 = nn.Conv2d(32, 64, 3, padding=1)
        self.gn4 = nn.GroupNorm(16, 64)

        self.conv5 = nn.Conv2d(64, 64, 3, padding=1)
        self.gn5 = nn.GroupNorm(16, 64)

        self.conv6 = nn.Conv2d(64, 128, 3, padding=1)
        self.gn6 = nn.GroupNorm(32, 128)

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.20)

        self.fc1 = nn.Linear(64 * 3 * 3, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 47)

    def _preprocess(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.dim() == 3:
            x = x.unsqueeze(1)
        return x

    def _features_to_logits(self, x):
        x = F.relu(self.gn1(self.conv1(x)))
        x = F.relu(self.gn2(self.conv2(x)))
        x = self.pool(x)
        x = F.relu(self.gn3(self.conv3(x)))
        x = self.pool(x)
        x = F.relu(self.gn4(self.conv4(x)))
        x = self.pool(x)
        x = x.reshape(x.size(0), -1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        return self.fc3(x)

    def forward_training(self, x):
        x = self._preprocess(x)
        return self._features_to_logits(x)

    def forward(self, x):
        """
        Inférence du réseau de neurones.

        Entrée: Une image x enregistrée comme un tensor.FloatTensor d'une des dimensions suivantes:
                    - (28, 28)
                    - (B, 28, 28)
                    - (B, 1, 28, 28)
                et dont les entrées sont comprises entre 0 et 1.

        Sortie: Un vecteur y enregistré comme un tensor.FloatTensor de dimension
                    - (47,) si B = 1
                    - (B, 47) si B > 1
                avec les probabilités de classification associées à chaque classe.

        Contrainte: y doit satisfaire à la définition de probabilités.
        """

        single_image = x.dim() == 2
        logits = self.forward_training(x)
        y = torch.softmax(logits, dim=1)
        if single_image:
            y = y.squeeze(0)
        return y


def normalize_l2(tensor):
    flat = tensor.reshape(tensor.size(0), -1)
    norms = flat.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12)
    return tensor / norms.reshape(-1, 1, 1, 1)


def project_l2(delta, eps):
    flat = delta.reshape(delta.size(0), -1)
    norms = flat.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12)
    factors = torch.clamp(float(eps) / norms, max=1.0)
    return delta * factors.reshape(-1, 1, 1, 1)


def random_l2_start(x, eps):
    delta = torch.randn_like(x)
    delta = normalize_l2(delta)
    radius = torch.rand(x.size(0), 1, 1, 1, device=x.device)
    delta = delta * radius * float(eps)
    delta = (x + delta).clamp(0.0, 1.0) - x
    return project_l2(delta, eps)


def pgd_l2_attack(model, x, y, eps, n_steps, step_size=None, restarts=1):
    if eps <= 0.0:
        return x.detach().clone()

    if step_size is None:
        step_size = 2.5 * float(eps) / max(n_steps, 1)

    was_training = model.training
    model.eval()

    best_adv = x.detach().clone()
    best_loss = torch.full((x.size(0),), -1e30, device=x.device)

    for restart_idx in range(max(restarts, 1)):
        if restart_idx == 0:
            delta = torch.zeros_like(x)
        else:
            delta = random_l2_start(x, eps)

        for step_idx in range(n_steps):
            delta.requires_grad_(True)
            x_adv = (x + delta).clamp(0.0, 1.0)
            logits = model.forward_training(x_adv)
            loss_vec = F.cross_entropy(logits, y, reduction="none")
            loss = loss_vec.sum()
            grad = torch.autograd.grad(loss, delta)[0]

            grad = normalize_l2(grad)
            step_scale = 0.5 * (1.0 + math.cos(math.pi * step_idx / max(n_steps - 1, 1)))

            delta = delta.detach() + step_size * step_scale * grad
            delta = (x + delta).clamp(0.0, 1.0) - x
            delta = project_l2(delta, eps).detach()

        with torch.no_grad():
            x_adv = (x + delta).clamp(0.0, 1.0)
            loss_vec = F.cross_entropy(model.forward_training(x_adv), y, reduction="none")
            better = loss_vec > best_loss
            best_loss[better] = loss_vec[better]
            best_adv[better] = x_adv[better]

    model.train(was_training)
    return best_adv.detach()


def evaluate_clean_accuracy(model, loader, eval_device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(eval_device)
            labels = labels.to(eval_device)
            preds = model(images).argmax(dim=1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)

    return correct / max(total, 1)


def evaluate_clean_loss_and_accuracy(model, loader, eval_device, criterion):
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(eval_device)
            labels = labels.to(eval_device)
            logits = model.forward_training(images)
            loss = criterion(logits, labels)
            total_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)

    return total_loss / max(total, 1), correct / max(total, 1)


def evaluate_robust_accuracy(model, loader, eval_device, eps, n_steps):
    model.eval()
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(eval_device)
        labels = labels.to(eval_device)

        adv_images = pgd_l2_attack(
            model=model,
            x=images,
            y=labels,
            eps=eps,
            n_steps=n_steps,
            restarts=1,
        )

        with torch.no_grad():
            preds = model(adv_images).argmax(dim=1)

        correct += preds.eq(labels).sum().item()
        total += labels.size(0)

    return correct / max(total, 1)


def save_training_metrics(train_losses, val_losses, accuracy):
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics = {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "accuracy": accuracy,
    }

    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


def main():
    filename = input("Enter the filename to save the model (without .pt extension): ") + ".pt"

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: torch.rot90(x, k=3, dims=[1, 2])),
        transforms.Lambda(lambda x: torch.flip(x, dims=[2])),
    ])

    train_dataset = datasets.EMNIST(
        root="dataset/",
        split="balanced",
        download=True,
        train=True,
        transform=transform,
    )
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device == "cuda"),
    )

    test_dataset = datasets.EMNIST(
        root="dataset/",
        split="balanced",
        download=True,
        train=False,
        transform=transform,
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device == "cuda"),
    )

    robust_eval_subset = Subset(test_dataset, range(min(2000, len(test_dataset))))
    robust_eval_loader = DataLoader(
        dataset=robust_eval_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device == "cuda"),
    )

    model = ClassifierS11GroupXX().to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=6, gamma=0.5)

    train_losses = []
    val_losses = []
    accuracy = []

    num_epochs = 20
    warmup_epochs = 2
    train_eps = 1.5
    train_steps = 6
    train_restarts = 1
    adv_loss_weight = 0.6

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_clean_correct = 0
        running_adv_correct = 0
        seen = 0

        if epoch < warmup_epochs:
            current_eps = 0.0
        else:
            ramp = min(1.0, float(epoch - warmup_epochs + 1) / 4.0)
            current_eps = train_eps * ramp

        print(f"Epoch [{epoch + 1}/{num_epochs}] | train_eps={current_eps:.3f}")

        for batch_index, (data, targets) in enumerate(train_loader):
            data = data.to(device, non_blocking=(device == "cuda"))
            targets = targets.to(device, non_blocking=(device == "cuda"))

            clean_logits = model.forward_training(data)
            clean_loss = criterion(clean_logits, targets)

            if current_eps > 0.0:
                adv_data = pgd_l2_attack(
                    model=model,
                    x=data,
                    y=targets,
                    eps=current_eps,
                    n_steps=train_steps,
                    restarts=train_restarts,
                )
                adv_logits = model.forward_training(adv_data)
                adv_loss = criterion(adv_logits, targets)
                loss = (1.0 - adv_loss_weight) * clean_loss + adv_loss_weight * adv_loss

                with torch.no_grad():
                    running_adv_correct += adv_logits.argmax(dim=1).eq(targets).sum().item()
            else:
                loss = clean_loss
                with torch.no_grad():
                    running_adv_correct += clean_logits.argmax(dim=1).eq(targets).sum().item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                running_clean_correct += clean_logits.argmax(dim=1).eq(targets).sum().item()
                running_loss += loss.item() * data.size(0)
                seen += data.size(0)

            if batch_index % 100 == 0:
                mean_loss = running_loss / max(seen, 1)
                clean_acc = running_clean_correct / max(seen, 1)
                adv_acc = running_adv_correct / max(seen, 1)
                print(
                    f"Batch {batch_index:04d} | loss={mean_loss:.4f} | clean_acc={clean_acc:.4f} | adv_acc={adv_acc:.4f}"
                )

        scheduler.step()

        train_losses.append(running_loss / max(seen, 1))
        val_loss, clean_acc = evaluate_clean_loss_and_accuracy(model, test_loader, device, criterion)
        val_losses.append(val_loss)
        accuracy.append(clean_acc)
        save_training_metrics(train_losses, val_losses, accuracy)

        robust_acc = evaluate_robust_accuracy(model, robust_eval_loader, device, eps=train_eps, n_steps=8)
        print(f"Test clean accuracy: {clean_acc:.4f}")
        print(f"Robust accuracy @ eps={train_eps:.2f}: {robust_acc:.4f}")
        print(f"Saved epoch metrics to {metrics_path}")

    cpu_model = model.to("cpu")
    cpu_model.eval()

    scripted_model = torch.jit.script(cpu_model)
    scripted_model.save(f"{filename}")
    print(f"Model saved to classifiers/{filename}")


if __name__ == "__main__":
    main()