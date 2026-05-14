import torch
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, Slider
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


# ============================================================
# Utilitaires modèles
# ============================================================
def prepare_input(x2d: torch.Tensor, mode: str) -> torch.Tensor:
    """
    x2d : tenseur de forme (28, 28)
    mode='nchw' -> (1, 1, 28, 28)
    mode='nhw'  -> (1, 28, 28)
    """
    if mode == "nchw":
        return x2d.unsqueeze(0).unsqueeze(0)
    elif mode == "nhw":
        return x2d.unsqueeze(0)
    else:
        raise ValueError(f"Mode d'entrée inconnu : {mode}")


def normalize_logits_shape(logits: torch.Tensor) -> torch.Tensor:
    """
    Certains modèles peuvent renvoyer (C,) au lieu de (1, C).
    On force ici une forme compatible.
    """
    if logits.ndim == 1:
        logits = logits.unsqueeze(0)
    return logits


def resolve_input_mode(model, x2d: torch.Tensor) -> str:
    """
    Détecte automatiquement si le modèle attend :
    - (1, 1, 28, 28)  -> mode 'nchw'
    - (1, 28, 28)     -> mode 'nhw'

    On n'applique aucune transformation au dessin lui-même.
    """
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


def predict_one(model, x2d: torch.Tensor, mode: str):
    with torch.no_grad():
        logits = forward_logits(model, x2d, mode)
        pred_idx = logits.argmax(dim=1).item()
    return pred_idx


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

        # Figure principale
        self.fig, self.ax = plt.subplots(figsize=(7, 7))
        plt.subplots_adjust(bottom=0.24)

        self.im = self.ax.imshow(
            self.canvas,
            cmap="gray",
            vmin=0.0,
            vmax=1.0,
            origin="upper",
            interpolation="nearest",
            extent=(0, self.size, self.size, 0)
        )

        self.ax.set_title("Dessine un chiffre ou une lettre")
        self.ax.set_xlim(0, self.size)
        self.ax.set_ylim(self.size, 0)
        self.ax.set_xticks([])
        self.ax.set_yticks([])

        # Texte de statut dans la figure
        self.status_text = self.fig.text(
            0.34, 0.06,
            "Prédictions : -",
            fontsize=10,
            va="center",
            family="monospace"
        )

        # Connexions souris
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
            print("Pas de modèles chargés.")
            self.status_text.set_text("Prédictions : aucun modèle")
            self.fig.canvas.draw_idle()
            return

        if np.max(self.canvas) <= 0:
            print("Canvas vide.")
            self.status_text.set_text("Prédictions : canvas vide")
            self.fig.canvas.draw_idle()
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

    def attack_and_show(self, eps=2.0):
        if not self.models:
            print("Pas de modèles chargés.")
            self.status_text.set_text("Attaque : aucun modèle")
            self.fig.canvas.draw_idle()
            return

        if np.max(self.canvas) <= 0:
            print("Canvas vide.")
            self.status_text.set_text("Attaque : canvas vide")
            self.fig.canvas.draw_idle()
            return

        x_clean_cpu = self.get_tensor_cpu()

        names = list(self.models.keys())
        n_models = len(names)

        fig_adv, axes = plt.subplots(1, n_models, figsize=(4 * n_models, 4))
        if n_models == 1:
            axes = [axes]

        result_lines = []
        print(f"\n=== Attaque externe (eps={eps}) ===")

        for ax, name in zip(axes, names):
            model = self.models[name]
            mode = self.input_modes[name]
            model_path = self.model_paths.get(name, None)

            clean_idx = predict_one(model, x_clean_cpu.to(self.device), mode)
            clean_char = emnist_map.get(clean_idx, str(clean_idx))

            if model_path is None:
                print(f"[WARN] Pas de chemin enregistré pour le modèle {name}, attaque ignorée.")
                x_adv = x_clean_cpu.clone()
                adv_char = clean_char
                delta_norm = 0.0
            else:
                delta = external_attack(
                    x=x_clean_cpu,
                    f_string=model_path,
                    eps=float(eps)
                )

                delta = delta.float().cpu().reshape(self.size, self.size)
                x_adv = torch.clamp(x_clean_cpu + delta, 0.0, 1.0)

                adv_idx = predict_one(model, x_adv.to(self.device), mode)
                adv_char = emnist_map.get(adv_idx, str(adv_idx))
                delta_norm = torch.norm(delta.reshape(-1), p=2).item()

            ax.imshow(x_adv.detach().cpu().numpy(), cmap="gray", vmin=0.0, vmax=1.0, origin="upper")
            ax.set_title(f"{name}\n{clean_char} -> {adv_char}\n||delta||₂={delta_norm:.3f}")
            ax.axis("off")

            line = f"{name}: {clean_char} -> {adv_char} (||delta||₂={delta_norm:.3f})"
            result_lines.append(line)
            print(line)

        fig_adv.suptitle(f"Dessins adversariaux - attaquant externe (eps={eps})")
        fig_adv.tight_layout()
        plt.show()

        self.status_text.set_text("Attaque : " + " | ".join(result_lines))
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

    # ----------------------------
    # Boutons
    # ----------------------------
    ax_clear = drawer.fig.add_axes([0.62, 0.12, 0.10, 0.06])
    btn_clear = Button(ax_clear, "Effacer")
    btn_clear.on_clicked(drawer.clear)

    ax_pred = drawer.fig.add_axes([0.74, 0.12, 0.10, 0.06])
    btn_pred = Button(ax_pred, "Prédire")
    btn_pred.on_clicked(drawer.predict)

    ax_adv = drawer.fig.add_axes([0.86, 0.12, 0.10, 0.06])
    btn_adv = Button(ax_adv, "Attaquer")

    # ----------------------------
    # Sliders
    # ----------------------------
    ax_eps = drawer.fig.add_axes([0.08, 0.10, 0.18, 0.03])
    slider_eps = Slider(ax_eps, "Epsilon", 0.0, 2.0, valinit=1.0, valstep=0.05)

    ax_brush = drawer.fig.add_axes([0.08, 0.04, 0.18, 0.03])
    slider_brush = Slider(ax_brush, "Brush", 1, 6, valinit=2, valstep=1)

    def update_brush(_):
        drawer.set_brush_radius(int(slider_brush.val))

    slider_brush.on_changed(update_brush)

    def launch_attack(_):
        drawer.attack_and_show(
            eps=float(slider_eps.val)
        )

    btn_adv.on_clicked(launch_attack)

    plt.show()