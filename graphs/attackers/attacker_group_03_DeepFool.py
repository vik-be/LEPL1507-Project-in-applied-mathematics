import torch
import time


def attack(x, f_string, eps):
    max_iter = 20
    start = time.time()
    eps = float(eps)

    model = torch.jit.load(f_string)
    model.eval()

    x_orig = x.clone().detach()  # 28x28

    # Détecter le mode d'entrée du modèle (2D, 3D ou 4D)
    def get_mode():
        with torch.no_grad():
            for mode, xin in [
                (2, x_orig),
                (3, x_orig.unsqueeze(0)),
                (4, x_orig.unsqueeze(0).unsqueeze(0)),
            ]:
                try:
                    out = model(xin)
                    out = out.reshape(-1)
                    if out.numel() == 47:
                        return mode
                except Exception:
                    pass
        return 4

    def to_model_input(x2d):
        if mode == 2:
            return x2d
        elif mode == 3:
            return x2d.unsqueeze(0)
        else:
            return x2d.unsqueeze(0).unsqueeze(0)

    mode = get_mode()

    with torch.no_grad():
        y_pred = model(to_model_input(x_orig)).reshape(-1)
        label_idx = int(torch.argmax(y_pred).item())

    delta = torch.zeros_like(x_orig)  # 28x28 comme x

    for _ in range(max_iter):
        x_adv = (x_orig + delta).clone().detach()
        x_adv.requires_grad_(True)

        outputs = model(to_model_input(x_adv)).reshape(-1)  # shape: (47,)
        current_idx = int(torch.argmax(outputs).item())

        if current_idx != label_idx:
            break

        grad_orig = torch.autograd.grad(
            outputs[label_idx], x_adv, retain_graph=True
        )[0]
        # Ramener le gradient en 28x28
        grad_orig_2d = grad_orig.reshape(x_orig.shape)
        if not torch.isfinite(grad_orig_2d).all():
            break

        min_pert = float("inf")
        w_best = None

        for k in range(outputs.shape[0]):
            if k == label_idx:
                continue

            grad_k = torch.autograd.grad(
                outputs[k], x_adv, retain_graph=True
            )[0].reshape(x_orig.shape)

            if not torch.isfinite(grad_k).all():
                continue

            w_k = grad_k - grad_orig_2d
            f_k = outputs[k] - outputs[label_idx]

            if not torch.isfinite(w_k).all() or not torch.isfinite(f_k):
                continue

            pert_k = torch.abs(f_k) / (torch.norm(w_k.view(-1)) + 1e-8)
            if not torch.isfinite(pert_k):
                continue

            pert_value = float(pert_k.item())
            if pert_value < min_pert:
                min_pert = pert_value
                w_best = w_k

        if w_best is None:
            break

        direction = w_best / (torch.norm(w_best.view(-1)) + 1e-8)
        if not torch.isfinite(direction).all():
            break

        delta = delta + (min_pert * direction)
        if not torch.isfinite(delta).all():
            delta = torch.zeros_like(x_orig)
            break

        # Projection L2
        norm = torch.norm(delta.view(-1))
        if not torch.isfinite(norm) or norm == 0:
            delta = torch.zeros_like(x_orig)
            break
        if norm > eps:
            delta = eps * delta / norm

        # Projection boîte
        delta = torch.clamp(x_orig + delta, 0.0, 1.0) - x_orig

        if time.time() - start > 9.5:
            break

    # --- Finalisation ---
    delta = torch.clamp(x_orig + delta, 0.0, 1.0) - x_orig

    norm = torch.norm(delta.view(-1))
    if norm > eps:
        delta = (eps - 1e-6) * delta / (norm + 1e-12)

    delta = torch.clamp(delta, -1.0, 1.0)

    if not torch.isfinite(delta).all():
        delta = torch.zeros_like(x_orig)

    return delta.detach()