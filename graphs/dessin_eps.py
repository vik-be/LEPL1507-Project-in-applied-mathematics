import torch
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, Slider, TextBox
from pathlib import Path

# ============================================================
# Import de l'attaquant externe
# ============================================================

from attackers.attacker_group_03 import attack as external_attack

# ============================================================
# Mapping EMNIST Balanced
# ============================================================
emnist_map = {
    0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
    10: 'A', 11: 'B', 12: 'C', 13: 'D', 14: 'E', 15: 'F', 16: 'G', 17: 'H', 18: 'I', 19: 'J',
    20: 'K', 21: 'L', 22: 'M', 23: 'N', 24: 'O', 25: 'P', 26: 'Q', 27: 'R', 28: 'S', 29: 'T',
    30: 'U', 31: 'V', 32: 'W', 33: 'X', 34: 'Y', 35: 'Z', 36: 'a', 37: 'b', 38: 'd', 39: 'e',
    40: 'f', 41: 'g', 42: 'h', 43: 'n', 44: 'q', 45: 'r', 46: 't'
}
char_to_idx = {v: k for k, v in emnist_map.items()}


# ============================================================
# Utilitaires modèles
# ============================================================
def prepare_input(x2d: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "nchw":
        return x2d.unsqueeze(0).unsqueeze(0)
    elif mode == "nhw":
        return x2d.unsqueeze(0)
    else:
        raise ValueError(f"Mode d'entrée inconnu : {mode}")


def normalize_logits_shape(logits: torch.Tensor) -> torch.Tensor:
    if logits.ndim == 1:
        logits = logits.unsqueeze(0)
    return logits


def resolve_input_mode(model, x2d: torch.Tensor) -> str:
    for mode in ("nchw", "nhw"):
        try:
            with torch.no_grad():
                logits = model(prepare_input(x2d, mode))
                logits = normalize_logits_shape(logits)
            return mode
        except Exception:
            pass

    raise RuntimeError(
        "Impossible de déterminer le format d'entrée attendu par le modèle. "
        "Il n'accepte ni (1,1,28,28) ni (1,28,28)."
    )


def forward_logits(model, x2d: torch.Tensor, mode: str) -> torch.Tensor:
    logits = model(prepare_input(x2d, mode))
    logits = normalize_logits_shape(logits)
    return logits


def predict_one(model, x2d: torch.Tensor, mode: str) -> int:
    with torch.no_grad():
        logits = forward_logits(model, x2d, mode)
        pred_idx = logits.argmax(dim=1).item()
    return int(pred_idx)


def safe_model_device(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except Exception:
        pass

    try:
        return next(model.buffers()).device
    except Exception:
        pass

    return torch.device("cpu")


def parse_user_label(text: str):
    """
    Retourne :
    - None si vide
    - (idx, char) si valide
    - "invalid" si non reconnu
    """
    s = text.strip()
    if s == "":
        return None
    if s in char_to_idx:
        return char_to_idx[s], s
    return "invalid"


# ============================================================
# Attaque utilitaire
# ============================================================
def run_external_attack(x_clean_cpu: torch.Tensor, model_path: str, eps: float, size=28):
    delta = external_attack(
        x=x_clean_cpu,
        f_string=model_path,
        eps=float(eps)
    )

    delta = delta.float().cpu().reshape(size, size)
    x_adv = torch.clamp(x_clean_cpu + delta, 0.0, 1.0)
    delta_norm = torch.norm(delta.reshape(-1), p=2).item()

    return delta, x_adv, delta_norm


def find_minimal_eps_for_failure(
    model,
    mode,
    x_clean_cpu: torch.Tensor,
    model_path: str,
    reference_idx: int,
    clean_idx: int,
    size=28,
    eps_max=2.0,
    coarse_step=0.2,
    refine_steps=8
):
    """
    Cherche le plus petit eps pour lequel adv_idx != reference_idx.

    - Si l'utilisateur a fourni un label : reference_idx = vrai label supposé
    - Sinon : reference_idx = clean_idx, donc on cherche le premier changement
      par rapport à la prédiction propre
    """
    device = safe_model_device(model)

    tested_curve = [{
        "eps": 0.0,
        "adv_idx": int(clean_idx),
        "changed": bool(clean_idx != reference_idx),
        "delta_norm": 0.0
    }]

    # Si déjà faux à eps=0 par rapport au label saisi
    if clean_idx != reference_idx:
        zero_delta = torch.zeros_like(x_clean_cpu)
        return {
            "found": True,
            "eps_star": 0.0,
            "delta_star": zero_delta,
            "x_adv_star": x_clean_cpu.clone(),
            "adv_idx": int(clean_idx),
            "delta_norm": 0.0,
            "clean_idx": int(clean_idx),
            "reference_idx": int(reference_idx),
            "curve": tested_curve,
            "already_wrong_at_zero": True
        }

    coarse_eps = np.arange(coarse_step, eps_max + 1e-12, coarse_step, dtype=np.float64)

    last_fail_eps = 0.0
    first_success_eps = None
    first_success_adv = None
    first_success_delta = None
    first_success_pred = clean_idx
    first_success_norm = 0.0

    for eps in coarse_eps:
        delta, x_adv, delta_norm = run_external_attack(
            x_clean_cpu=x_clean_cpu,
            model_path=model_path,
            eps=float(eps),
            size=size
        )

        adv_idx = predict_one(model, x_adv.to(device), mode)
        changed = (adv_idx != reference_idx)

        tested_curve.append({
            "eps": float(eps),
            "adv_idx": int(adv_idx),
            "changed": bool(changed),
            "delta_norm": float(delta_norm)
        })

        if changed:
            first_success_eps = float(eps)
            first_success_adv = x_adv
            first_success_delta = delta
            first_success_pred = adv_idx
            first_success_norm = delta_norm
            break
        else:
            last_fail_eps = float(eps)

    if first_success_eps is None:
        return {
            "found": False,
            "clean_idx": int(clean_idx),
            "reference_idx": int(reference_idx),
            "curve": tested_curve
        }

    lo = last_fail_eps
    hi = first_success_eps

    best_eps = first_success_eps
    best_adv = first_success_adv
    best_delta = first_success_delta
    best_pred = first_success_pred
    best_norm = first_success_norm

    for _ in range(refine_steps):
        mid = 0.5 * (lo + hi)

        delta, x_adv, delta_norm = run_external_attack(
            x_clean_cpu=x_clean_cpu,
            model_path=model_path,
            eps=float(mid),
            size=size
        )

        adv_idx = predict_one(model, x_adv.to(device), mode)
        changed = (adv_idx != reference_idx)

        tested_curve.append({
            "eps": float(mid),
            "adv_idx": int(adv_idx),
            "changed": bool(changed),
            "delta_norm": float(delta_norm)
        })

        if changed:
            hi = mid
            best_eps = mid
            best_adv = x_adv
            best_delta = delta
            best_pred = adv_idx
            best_norm = delta_norm
        else:
            lo = mid

    return {
        "found": True,
        "eps_star": float(best_eps),
        "delta_star": best_delta,
        "x_adv_star": best_adv,
        "adv_idx": int(best_pred),
        "delta_norm": float(best_norm),
        "clean_idx": int(clean_idx),
        "reference_idx": int(reference_idx),
        "curve": tested_curve,
        "already_wrong_at_zero": False
    }


# ============================================================
# Interface de dessin
# ============================================================
class DrawEMNIST:
    def __init__(self, size=28, brush_radius=2):
        self.size = size
        self.canvas = np.zeros((size, size), dtype=np.float32)

        self.device = torch.device("cpu")
        self.models = {}
        self.model_paths = {}
        self.input_modes = {}

        self.drawing = False
        self.last_point = None

        self.brush_radius = int(brush_radius)
        self.brush_mask = self.make_brush(self.brush_radius)

        self.fig, self.ax = plt.subplots(figsize=(8.6, 8.8))
        plt.subplots_adjust(bottom=0.38)

        self.im = self.ax.imshow(
            self.canvas,
            cmap="gray",
            vmin=0.0,
            vmax=1.0,
            origin="upper",
            interpolation="nearest",
            extent=(0, self.size, self.size, 0)
        )

        self.ax.set_title("Dessine un chiffre ou une lettre", pad=12)
        self.ax.set_xlim(0, self.size)
        self.ax.set_ylim(self.size, 0)
        self.ax.set_xticks([])
        self.ax.set_yticks([])

        self.status_text = self.fig.text(
            0.50, 0.320,
            "Prédictions : -",
            fontsize=10,
            va="center",
            ha="center",
            family="monospace"
        )

        self.params_text = self.fig.text(
            0.79, 0.11,
            "",
            fontsize=9,
            va="center",
            ha="center",
            family="monospace"
        )

        self.cid_press = self.fig.canvas.mpl_connect("button_press_event", self.on_press)
        self.cid_release = self.fig.canvas.mpl_connect("button_release_event", self.on_release)
        self.cid_move = self.fig.canvas.mpl_connect("motion_notify_event", self.on_move)

    # ----------------------------
    # Gestion du pinceau
    # ----------------------------
    def make_brush(self, radius):
        radius = int(radius)
        if radius <= 0:
            return np.ones((1, 1), dtype=np.float32)

        y, x = np.ogrid[-radius:radius + 1, -radius:radius + 1]
        dist2 = x * x + y * y

        sigma = max(radius / 2.0, 0.8)
        mask = np.exp(-dist2 / (2 * sigma * sigma)).astype(np.float32)
        mask[dist2 > radius * radius] *= 0.25
        mask /= mask.max()
        return mask

    def set_brush_radius(self, radius):
        self.brush_radius = int(radius)
        self.brush_mask = self.make_brush(self.brush_radius)

    # ----------------------------
    # Gestion des modèles
    # ----------------------------
    def add_models(self, models, model_paths=None, device=None):
        self.models = models
        self.model_paths = model_paths if model_paths is not None else {}

        if device is not None:
            self.device = device

        self.input_modes = {}
        x_dummy = torch.zeros((self.size, self.size), dtype=torch.float32, device=self.device)

        for name, model in self.models.items():
            model.to(self.device)
            model.eval()

            mode = resolve_input_mode(model, x_dummy)
            self.input_modes[name] = mode
            print(f"Modèle {name} chargé (format détecté : {mode})")

    # ----------------------------
    # Conversion du canvas en tenseur
    # ----------------------------
    def get_tensor(self) -> torch.Tensor:
        return torch.from_numpy(self.canvas.copy()).float().to(self.device)

    def get_tensor_cpu(self) -> torch.Tensor:
        return torch.from_numpy(self.canvas.copy()).float().cpu()

    # ----------------------------
    # Outils de dessin
    # ----------------------------
    def event_to_pixel(self, event):
        if event.inaxes != self.ax:
            return None
        if event.xdata is None or event.ydata is None:
            return None

        x = int(np.clip(np.floor(event.xdata), 0, self.size - 1))
        y = int(np.clip(np.floor(event.ydata), 0, self.size - 1))
        return x, y

    def stamp_brush(self, x_center, y_center):
        r = self.brush_radius
        mask = self.brush_mask

        x0 = max(0, x_center - r)
        x1 = min(self.size, x_center + r + 1)
        y0 = max(0, y_center - r)
        y1 = min(self.size, y_center + r + 1)

        bx0 = x0 - (x_center - r)
        bx1 = bx0 + (x1 - x0)
        by0 = y0 - (y_center - r)
        by1 = by0 + (y1 - y0)

        current = self.canvas[y0:y1, x0:x1]
        brush_part = mask[by0:by1, bx0:bx1]

        self.canvas[y0:y1, x0:x1] = np.maximum(current, brush_part)

    def draw_segment(self, p0, p1):
        x0, y0 = p0
        x1, y1 = p1

        n = max(abs(x1 - x0), abs(y1 - y0), 1) + 1
        xs = np.linspace(x0, x1, n)
        ys = np.linspace(y0, y1, n)

        for x, y in zip(xs, ys):
            self.stamp_brush(int(round(x)), int(round(y)))

    def refresh(self):
        self.im.set_data(self.canvas)
        self.fig.canvas.draw_idle()

    def set_params_text(self, coarse_step: float, brush: int, label_text: str):
        shown_label = label_text.strip() if label_text.strip() else "-"
        self.params_text.set_text(
            f"eps max = 2.00\n"
            f"pas = {coarse_step:.2f}\n"
            f"brush = {brush}\n"
            f"label = {shown_label}"
        )
        self.fig.canvas.draw_idle()

    # ----------------------------
    # Événements souris
    # ----------------------------
    def on_press(self, event):
        if event.inaxes != self.ax or event.button != 1:
            return

        p = self.event_to_pixel(event)
        if p is None:
            return

        self.drawing = True
        self.last_point = p
        self.stamp_brush(*p)
        self.refresh()

    def on_move(self, event):
        if not self.drawing:
            return
        if event.inaxes != self.ax:
            return

        p = self.event_to_pixel(event)
        if p is None:
            return

        if self.last_point is None:
            self.stamp_brush(*p)
        else:
            self.draw_segment(self.last_point, p)

        self.last_point = p
        self.refresh()

    def on_release(self, event):
        if event.button == 1:
            self.drawing = False
            self.last_point = None

    # ----------------------------
    # Boutons
    # ----------------------------
    def clear(self, event=None):
        self.canvas.fill(0.0)
        self.status_text.set_text("Prédictions : -")
        self.refresh()

    def predict(self, event=None):
        if not self.models:
            self.status_text.set_text("Prédictions : aucun modèle")
            self.fig.canvas.draw_idle()
            print("Pas de modèles chargés.")
            return

        if np.max(self.canvas) <= 0:
            self.status_text.set_text("Prédictions : canvas vide")
            self.fig.canvas.draw_idle()
            print("Canvas vide.")
            return

        x2d = self.get_tensor()

        lines = []
        print("\n=== Prédictions ===")
        for name, model in self.models.items():
            pred_idx = predict_one(model, x2d, self.input_modes[name])
            pred_char = emnist_map.get(pred_idx, str(pred_idx))
            line = f"{name}: {pred_char}"
            lines.append(line)
            print(line)

        self.status_text.set_text("Prédictions : " + " | ".join(lines))
        self.fig.canvas.draw_idle()
        print()

    def find_min_delta_and_plot(
        self,
        eps_max=2.0,
        coarse_step=0.2,
        refine_steps=8,
        user_label_idx=None,
        user_label_char=None
    ):
        if not self.models:
            self.status_text.set_text("Recherche : aucun modèle")
            self.fig.canvas.draw_idle()
            print("Pas de modèles chargés.")
            return

        if np.max(self.canvas) <= 0:
            self.status_text.set_text("Recherche : canvas vide")
            self.fig.canvas.draw_idle()
            print("Canvas vide.")
            return

        x_clean_cpu = self.get_tensor_cpu()
        names = list(self.models.keys())
        n_models = len(names)

        results = []
        status_lines = []

        print("\n=== Recherche du plus petit eps qui casse la référence ===")
        print(f"(eps_max={eps_max}, coarse_step={coarse_step}, refine_steps={refine_steps})")

        if user_label_idx is None:
            print("Critère : changement par rapport à la prédiction propre.")
        else:
            print(f"Critère : première prédiction différente du label saisi '{user_label_char}'.")

        for name in names:
            model = self.models[name]
            mode = self.input_modes[name]
            model_path = self.model_paths.get(name, None)

            clean_idx = predict_one(model, x_clean_cpu.to(self.device), mode)
            clean_char = emnist_map.get(clean_idx, str(clean_idx))

            if user_label_idx is None:
                reference_idx = clean_idx
                reference_char = clean_char
            else:
                reference_idx = user_label_idx
                reference_char = user_label_char

            if model_path is None:
                print(f"[WARN] Pas de chemin pour le modèle {name}, recherche ignorée.")
                results.append({
                    "name": name,
                    "found": False,
                    "clean_idx": clean_idx,
                    "clean_char": clean_char,
                    "reference_idx": reference_idx,
                    "reference_char": reference_char,
                    "reason": "pas de chemin modèle"
                })
                status_lines.append(f"{name}: chemin manquant")
                continue

            res = find_minimal_eps_for_failure(
                model=model,
                mode=mode,
                x_clean_cpu=x_clean_cpu,
                model_path=model_path,
                reference_idx=reference_idx,
                clean_idx=clean_idx,
                size=self.size,
                eps_max=float(eps_max),
                coarse_step=float(coarse_step),
                refine_steps=int(refine_steps)
            )

            res["name"] = name
            res["clean_char"] = clean_char
            res["reference_char"] = reference_char

            if res["found"]:
                adv_char = emnist_map.get(res["adv_idx"], str(res["adv_idx"]))
                res["adv_char"] = adv_char

                if res.get("already_wrong_at_zero", False):
                    print(
                        f"{name}: déjà faux à eps=0 | "
                        f"label={reference_char}, prédiction propre={clean_char}"
                    )
                else:
                    print(
                        f"{name}: ref={reference_char}, clean={clean_char}, adv={adv_char} | "
                        f"eps*={res['eps_star']:.4f} | "
                        f"||delta||₂={res['delta_norm']:.4f}"
                    )

                status_lines.append(f"{name}: eps*={res['eps_star']:.3f}")
            else:
                print(f"{name}: correct jusqu'à eps={eps_max:.3f} selon la référence choisie")
                status_lines.append(f"{name}: pas de bascule")

            results.append(res)

        # ----------------------------------------------------
        # Figure 1 : images adversariales minimales
        # ----------------------------------------------------
        fig_adv, axes = plt.subplots(1, n_models, figsize=(4 * n_models, 4))
        if n_models == 1:
            axes = [axes]

        for ax, res in zip(axes, results):
            name = res["name"]
            clean_char = res["clean_char"]
            reference_char = res["reference_char"]

            if res.get("found", False):
                adv_char = res["adv_char"]
                x_adv = res["x_adv_star"].detach().cpu().numpy()
                ax.imshow(x_adv, cmap="gray", vmin=0.0, vmax=1.0, origin="upper")

                if user_label_idx is None:
                    title = (
                        f"{name}\n"
                        f"{clean_char} -> {adv_char}\n"
                        f"eps*={res['eps_star']:.4f}\n"
                        f"||delta||₂={res['delta_norm']:.4f}"
                    )
                else:
                    title = (
                        f"{name}\n"
                        f"label={reference_char} | clean={clean_char}\n"
                        f"adv={adv_char} | eps*={res['eps_star']:.4f}\n"
                        f"||delta||₂={res['delta_norm']:.4f}"
                    )
                ax.set_title(title)
            else:
                ax.imshow(x_clean_cpu.numpy(), cmap="gray", vmin=0.0, vmax=1.0, origin="upper")
                if user_label_idx is None:
                    ax.set_title(f"{name}\n{clean_char}\nPas de bascule")
                else:
                    ax.set_title(f"{name}\nlabel={reference_char} | clean={clean_char}\nPas de bascule")

            ax.axis("off")

        if user_label_idx is None:
            fig_adv.suptitle("Image adversariale minimale trouvée pour chaque modèle")
        else:
            fig_adv.suptitle(f"Image adversariale minimale pour casser le label '{user_label_char}'")
        fig_adv.tight_layout()

        # ----------------------------------------------------
        # Figure 2 : barplot des eps minimaux
        # ----------------------------------------------------
        found_names = [res["name"] for res in results if res.get("found", False)]
        found_eps = [res["eps_star"] for res in results if res.get("found", False)]

        plt.figure(figsize=(8, 4))
        if len(found_names) > 0:
            plt.bar(found_names, found_eps)
            plt.ylabel("eps minimal")
            plt.xlabel("Modèle")
            if user_label_idx is None:
                plt.title("Premier changement par rapport à la prédiction propre")
            else:
                plt.title(f"Premier eps où la prédiction n'est plus '{user_label_char}'")
            plt.ylim(0.0, 2.0)
        else:
            plt.text(0.5, 0.5, "Aucun basculement trouvé", ha="center", va="center")
            plt.xticks([])
            plt.yticks([])
            plt.title("Robustesse relative des modèles")
        plt.tight_layout()

        # ----------------------------------------------------
        # Figure 3 : courbe de basculement
        # ----------------------------------------------------
        plt.figure(figsize=(9, 5))
        any_curve = False

        for res in results:
            curve = res.get("curve", [])
            if not curve:
                continue

            curve_sorted = sorted(curve, key=lambda t: t["eps"])
            xs = [t["eps"] for t in curve_sorted]
            ys = [1.0 if t["changed"] else 0.0 for t in curve_sorted]

            plt.plot(xs, ys, marker='o', label=res["name"])
            any_curve = True

        if any_curve:
            plt.xlabel("eps")
            plt.ylabel("état")
            plt.xlim(0.0, 2.0)

            if user_label_idx is None:
                plt.yticks([0, 1], ["même classe", "classe changée"])
                plt.title("Basculement de la prédiction en fonction de eps")
            else:
                plt.yticks([0, 1], ["encore correct", "incorrect"])
                plt.title(f"Perte de correction par rapport au label '{user_label_char}'")

            plt.legend()
        else:
            plt.text(0.5, 0.5, "Pas de données à tracer", ha="center", va="center")
            plt.xticks([])
            plt.yticks([])

        plt.tight_layout()
        plt.show()

        self.status_text.set_text("Recherche : " + " | ".join(status_lines))
        self.fig.canvas.draw_idle()
        print()


# ============================================================
# Chargement des modèles
# ============================================================
def load_models(device):
    model_paths = {
        "A": "classifiers/classifier_S4_group_A.pt",
        "B": "classifiers/classifier_S4_group_B.pt",
        "C": "classifiers/classifier_S4_group_C.pt",
        "D": "classifiers/classifier_S4_group_D.pt",
        "E": "classifiers/classifier_S4_group_E.pt",
        "F": "classifiers/classifier_S4_group_F.pt",
        "S11": "classifiers/classifier_S11_group_03.pt"
    }

    models = {}
    existing_paths = {}

    for name, path_str in model_paths.items():
        path = Path(path_str)
        if path.exists():
            models[name] = torch.jit.load(str(path), map_location=device)
            existing_paths[name] = str(path)
        else:
            print(f"[INFO] Modèle {name} absent : {path}")

    if not models:
        raise FileNotFoundError(
            "Aucun modèle .pt trouvé dans other_classif/. "
            "Vérifie les chemins des classifieurs."
        )

    return models, existing_paths


# ============================================================
# Programme principal
# ============================================================
if __name__ == "__main__":
    device = torch.device("cpu")
    print(f"Device utilisé : {device}")

    models, model_paths = load_models(device)

    drawer = DrawEMNIST(size=28, brush_radius=2)
    drawer.add_models(models, model_paths=model_paths, device=device)

    # --------------------------------------------------------
    # Contrôles : TextBox label
    # --------------------------------------------------------
    ax_label_box = drawer.fig.add_axes([0.08, 0.215, 0.30, 0.055])
    txt_label = TextBox(ax_label_box, "label attendu ", initial="")

    # --------------------------------------------------------
    # Contrôles : sliders
    # --------------------------------------------------------
    ax_coarse = drawer.fig.add_axes([0.08, 0.125, 0.32, 0.03])
    slider_coarse = Slider(
        ax=ax_coarse,
        label="pas",
        valmin=0.05,
        valmax=1.0,
        valinit=0.2,
        valstep=0.05
    )

    ax_brush = drawer.fig.add_axes([0.08, 0.055, 0.32, 0.03])
    slider_brush = Slider(
        ax=ax_brush,
        label="brush",
        valmin=1,
        valmax=6,
        valinit=2,
        valstep=1
    )

    # --------------------------------------------------------
    # Contrôles : boutons
    # --------------------------------------------------------
    ax_clear = drawer.fig.add_axes([0.56, 0.215, 0.11, 0.055])
    btn_clear = Button(ax_clear, "Effacer")
    btn_clear.on_clicked(drawer.clear)

    ax_pred = drawer.fig.add_axes([0.70, 0.215, 0.11, 0.055])
    btn_pred = Button(ax_pred, "Prédire")
    btn_pred.on_clicked(drawer.predict)

    ax_find = drawer.fig.add_axes([0.84, 0.215, 0.12, 0.055])
    btn_find = Button(ax_find, "Chercher eps*")

    # --------------------------------------------------------
    # Synchronisation texte paramètres
    # --------------------------------------------------------
    def refresh_params_text():
        drawer.set_params_text(
            coarse_step=float(slider_coarse.val),
            brush=int(slider_brush.val),
            label_text=txt_label.text
        )

    def update_brush(_):
        drawer.set_brush_radius(int(slider_brush.val))
        refresh_params_text()

    def update_coarse(_):
        refresh_params_text()

    def update_label(_):
        refresh_params_text()

    slider_brush.on_changed(update_brush)
    slider_coarse.on_changed(update_coarse)
    txt_label.on_submit(update_label)

    refresh_params_text()

    # --------------------------------------------------------
    # Lancement de la recherche
    # --------------------------------------------------------
    def launch_find(_):
        label_info = parse_user_label(txt_label.text)

        if label_info == "invalid":
            print(
                "\n[WARN] Label invalide. Exemples valides : "
                "0, 7, A, Z, a, b, d, e, f, g, h, n, q, r, t"
            )
            print("      Je retombe sur le critère par défaut : changement de prédiction propre.")
            user_label_idx = None
            user_label_char = None
        elif label_info is None:
            user_label_idx = None
            user_label_char = None
        else:
            user_label_idx, user_label_char = label_info

        eps_max = 2.0
        coarse_step = float(slider_coarse.val)

        print(
            f"\n[INFO] Recherche lancée avec eps_max={eps_max:.3f}, "
            f"coarse_step={coarse_step:.3f}"
        )

        drawer.find_min_delta_and_plot(
            eps_max=eps_max,
            coarse_step=coarse_step,
            refine_steps=8,
            user_label_idx=user_label_idx,
            user_label_char=user_label_char
        )

    btn_find.on_clicked(launch_find)

    plt.show()