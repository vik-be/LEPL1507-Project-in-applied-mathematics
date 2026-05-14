import time
import torch
import numpy as np

# ============================================================
# Cache global : un cache par chemin de modèle
# ============================================================
# _MODEL_CACHE[f_string] = {
#     "model": modèle TorchScript,
#     "mode": 2 / 3 / 4,
#     "device": torch.device(...)
# }
_MODEL_CACHE = {}


# ============================================================
# Outils internes
# ============================================================

def _candidate_devices():
    devices = []
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))
    devices.append(torch.device("cpu"))
    return devices


def _sync_if_needed(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _elapsed(start_time, device):
    _sync_if_needed(device)
    return time.time() - start_time


def _flatten_output(out):
    out = out.float()

    if out.dim() == 0:
        return out.view(1)
    if out.dim() == 1:
        return out
    if out.dim() == 2 and out.size(0) == 1:
        return out[0]

    return out.reshape(-1)


def _infer_input_mode(model, x2d):
    """
    Essaie plusieurs formats d'entrée :
    - 2D : [28, 28]
    - 3D : [1, 28, 28]
    - 4D : [1, 1, 28, 28]
    """
    candidates = [
        (2, x2d),
        (3, x2d.unsqueeze(0)),
        (4, x2d.unsqueeze(0).unsqueeze(0)),
    ]

    with torch.no_grad():
        for mode, xin in candidates:
            try:
                out = model(xin)
                vec = _flatten_output(out)
                if vec.numel() == 47:
                    return mode
            except Exception:
                pass

    return 4


def _forward_raw(model, mode, x2d):
    if mode == 2:
        out = model(x2d)
    elif mode == 3:
        out = model(x2d.unsqueeze(0))
    else:
        out = model(x2d.unsqueeze(0).unsqueeze(0))
    return _flatten_output(out)


def _to_probabilities(raw_scores):
    raw_scores = raw_scores.float()

    looks_like_prob = (
        torch.isfinite(raw_scores).all()
        and (raw_scores.min() >= -1e-6)
        and (torch.abs(raw_scores.sum() - 1.0) <= 1e-3)
    )

    if looks_like_prob:
        probs = raw_scores.clamp(min=1e-12)
        probs = probs / probs.sum()
    else:
        probs = torch.softmax(raw_scores, dim=0).clamp(min=1e-12)

    return probs


def _predict_probs(model, mode, x2d):
    raw = _forward_raw(model, mode, x2d)
    return _to_probabilities(raw)


def _margin_untargeted(probs, y_ref):
    p_true = probs[y_ref]
    tmp = probs.clone()
    tmp[y_ref] = -1.0
    p_other = tmp.max()
    return torch.log(p_other.clamp(min=1e-12)) - torch.log(p_true.clamp(min=1e-12))


def _margin_targeted(probs, y_ref, y_target):
    p_true = probs[y_ref]
    p_target = probs[y_target]
    return torch.log(p_target.clamp(min=1e-12)) - torch.log(p_true.clamp(min=1e-12))


def _project_positive_l2(delta, eps, upper):
    """
    Projection dans :
    - delta >= 0
    - delta <= upper = 1 - x
    - ||delta||_2 <= eps
    """
    delta = torch.clamp(delta, min=0.0)
    delta = torch.minimum(delta, upper)

    norm = torch.norm(delta.reshape(-1), p=2)
    if norm > eps and norm > 0.0:
        delta = delta * (eps / norm)
        delta = torch.minimum(delta, upper)

    return delta


def _random_positive_start(upper, eps):
    delta = torch.rand_like(upper) * upper
    norm = torch.norm(delta.reshape(-1), p=2)
    if norm > eps and norm > 0.0:
        delta = delta * (eps / norm)
    return delta


def _successful(model, mode, x2d, delta, y_ref):
    with torch.no_grad():
        probs = _predict_probs(model, mode, torch.clamp(x2d + delta, 0.0, 1.0))
        y_adv = torch.argmax(probs)
    return int(y_adv.item()) != int(y_ref.item())


def _load_model_if_needed(f_string, x2d_cpu):
    """
    Essaie CUDA d'abord, puis CPU.
    Le choix est mémorisé par chemin de modèle.
    """
    global _MODEL_CACHE

    if f_string in _MODEL_CACHE:
        cached = _MODEL_CACHE[f_string]
        return cached["model"], cached["mode"], cached["device"]

    last_error = None

    for device in _candidate_devices():
        try:
            model = torch.jit.load(f_string, map_location=device)
            model.eval()

            x2d = x2d_cpu.to(device)
            mode = _infer_input_mode(model, x2d)

            # test réel
            with torch.no_grad():
                out = _forward_raw(model, mode, x2d)
            if out.numel() != 47:
                raise RuntimeError(f"Sortie inattendue de taille {out.numel()}")

            _MODEL_CACHE[f_string] = {
                "model": model,
                "mode": mode,
                "device": device,
            }
            return model, mode, device

        except Exception as e:
            last_error = e
            continue

    raise RuntimeError(
        f"Impossible de charger/utiliser le modèle {f_string} sur CUDA ou CPU. "
        f"Dernière erreur : {last_error}"
    )


def _optimize_phase(
    model, mode, x2d, y_ref, eps, target_idx, n_steps,
    step_scale, random_start, start_time, max_time, device
):
    """
    Petite attaque PGD-L2 positive avec momentum.
    target_idx = None  -> non ciblée
    target_idx = entier -> ciblée
    """
    upper = 1.0 - x2d

    if random_start:
        delta = _random_positive_start(upper, eps)
    else:
        delta = torch.zeros_like(x2d)

    delta = _project_positive_l2(delta, eps, upper)
    best_delta = delta.clone()

    with torch.no_grad():
        probs0 = _predict_probs(model, mode, torch.clamp(x2d + delta, 0.0, 1.0))
        if target_idx is None:
            best_score = _margin_untargeted(probs0, y_ref)
        else:
            best_score = _margin_targeted(probs0, y_ref, target_idx)

    velocity = torch.zeros_like(x2d)
    base_step = step_scale * eps / max(n_steps, 1)

    for k in range(n_steps):
        if _elapsed(start_time, device) > max_time:
            break

        delta.requires_grad_(True)
        x_adv = torch.clamp(x2d + delta, 0.0, 1.0)
        probs = _predict_probs(model, mode, x_adv)

        if target_idx is None:
            score = _margin_untargeted(probs, y_ref)
        else:
            score = _margin_targeted(probs, y_ref, target_idx)

        grad = torch.autograd.grad(score, delta)[0]

        # lecture stricte : delta doit rester positive
        grad = torch.clamp(grad, min=0.0)
        grad_norm = torch.norm(grad.reshape(-1), p=2)

        if grad_norm <= 1e-12:
            delta = delta.detach()
            continue

        grad = grad / grad_norm
        velocity = 0.85 * velocity + grad
        vel_norm = torch.norm(velocity.reshape(-1), p=2)
        if vel_norm > 1e-12:
            velocity = velocity / vel_norm

        step = base_step * (1.0 - 0.5 * float(k) / max(n_steps - 1, 1))

        delta = delta.detach() + step * velocity
        delta = _project_positive_l2(delta, eps, upper)

        with torch.no_grad():
            probs_now = _predict_probs(model, mode, torch.clamp(x2d + delta, 0.0, 1.0))
            if target_idx is None:
                score_now = _margin_untargeted(probs_now, y_ref)
            else:
                score_now = _margin_targeted(probs_now, y_ref, target_idx)

            if score_now > best_score:
                best_score = score_now
                best_delta = delta.clone()

            y_adv = torch.argmax(probs_now)
            if int(y_adv.item()) != int(y_ref.item()):
                return delta.detach(), True

    return best_delta.detach(), _successful(model, mode, x2d, best_delta, y_ref)


# ============================================================
# Fonction demandée
# ============================================================

def attack(x, f_string, eps):
    """
    Inputs :
    - x : tensor.FloatTensor 28x28 dans [0, 1]
    - f_string : chemin vers le classifieur .pt
    - eps : np.float dans [0, 2]

    Output :
    - delta : tensor.FloatTensor 28x28 sur CPU
    """

    start_time = time.time()
    
    eps = float(eps)
    if eps <= 0.0:
        x0 = x.detach().clone().float()
        if x0.dim() == 3 and x0.size(0) == 1:
            x0 = x0[0]
        return torch.zeros_like(x0.reshape(28, 28))

    if eps > 2.0:
        eps = 2.0

    # x d'abord sur CPU pour le choix du device
    x_cpu = x.detach().clone().float().cpu()
    if x_cpu.dim() == 3 and x_cpu.size(0) == 1:
        x_cpu = x_cpu[0]
    x_cpu = torch.clamp(x_cpu.reshape(28, 28), 0.0, 1.0)

    model, mode, device = _load_model_if_needed(f_string, x_cpu)

    max_time = 8.8

    # bascule réelle sur le device du modèle
    x2d = x_cpu.to(device)

    with torch.no_grad():
        probs_clean = _predict_probs(model, mode, x2d)
        y_ref = torch.argmax(probs_clean)

    best_delta = torch.zeros_like(x2d)

    # Phase 1 : non ciblée, départ zéro
    delta, success = _optimize_phase(
        model=model,
        mode=mode,
        x2d=x2d,
        y_ref=y_ref,
        eps=eps,
        target_idx=None,
        n_steps=14,
        step_scale=2.6,
        random_start=False,
        start_time=start_time,
        max_time=max_time,
        device=device,
    )
    best_delta = delta
    if success:
        _sync_if_needed(device)
        return best_delta.detach().cpu()

    # Phase 2 : non ciblée, départ aléatoire
    if _elapsed(start_time, device) <= max_time:
        delta, success = _optimize_phase(
            model=model,
            mode=mode,
            x2d=x2d,
            y_ref=y_ref,
            eps=eps,
            target_idx=None,
            n_steps=12,
            step_scale=2.2,
            random_start=True,
            start_time=start_time,
            max_time=max_time,
            device=device,
        )
        best_delta = delta
        if success:
            _sync_if_needed(device)
            return best_delta.detach().cpu()

    # Phase 3 : ciblée sur quelques classes probables
    if _elapsed(start_time, device) <= max_time:
        with torch.no_grad():
            order = torch.argsort(probs_clean, descending=True)
            targets = []
            for idx in order:
                if int(idx.item()) != int(y_ref.item()):
                    targets.append(int(idx.item()))
                if len(targets) >= 3:
                    break

            p0 = _predict_probs(model, mode, torch.clamp(x2d + best_delta, 0.0, 1.0))
            best_score = float(_margin_untargeted(p0, y_ref).item())

        for t in targets:
            if _elapsed(start_time, device) > max_time:
                break

            delta, success = _optimize_phase(
                model=model,
                mode=mode,
                x2d=x2d,
                y_ref=y_ref,
                eps=eps,
                target_idx=t,
                n_steps=8,
                step_scale=2.0,
                random_start=True,
                start_time=start_time,
                max_time=max_time,
                device=device,
            )

            with torch.no_grad():
                p_now = _predict_probs(model, mode, torch.clamp(x2d + delta, 0.0, 1.0))
                score_now = float(_margin_untargeted(p_now, y_ref).item())

            if score_now > best_score:
                best_score = score_now
                best_delta = delta

            if success:
                _sync_if_needed(device)
                return delta.detach().cpu()

    best_delta = _project_positive_l2(best_delta, eps, 1.0 - x2d)
    best_delta = torch.clamp(best_delta, 0.0, 1.0)
    _sync_if_needed(device)
    return best_delta.detach().cpu()
