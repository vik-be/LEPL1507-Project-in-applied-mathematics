import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# -----------------------
# CONFIG
# -----------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

# Change here

# filename = "classifier_S4_group_A"
# filename = "classifier_S4_group_B"
# filename = "classifier_S4_group_C"
# filename = "classifier_S4_group_D"
# filename = "classifier_S4_group_E"
# filename = "classifier_S4_group_F"
filename = "classifier_S11_group_03"

K = 10                                     # top-K most confused classes
# Stop change

use_rotation_and_flip = True               # True = code1 transform, False = code2 transform
batch_size = 256

# -----------------------
# LOAD MODEL
# -----------------------
model = torch.jit.load(f"classifiers/{filename}.pt", map_location=device)
model.eval()

# -----------------------
# DATASET + TRANSFORM
# -----------------------
if use_rotation_and_flip:
    transform = transforms.Compose([
        transforms.ToTensor(),  # (1,28,28)
        transforms.Lambda(lambda x: torch.rot90(x, k=3, dims=[1, 2])),  # -90°
        transforms.Lambda(lambda x: torch.flip(x, dims=[2]))           # horizontal mirror
    ])
else:
    transform = transforms.ToTensor()

test_dataset = datasets.EMNIST(
    root="dataset/",
    split="balanced",
    train=False,
    download=True,
    transform=transform
)

test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

classes = test_dataset.classes
num_classes = len(classes)

# -----------------------
# FULL CONFUSION MATRIX
# -----------------------
confusion = torch.zeros(num_classes, num_classes)

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        preds = outputs.argmax(dim=1)

        for t, p in zip(labels.cpu(), preds.cpu()):
            confusion[t, p] += 1

conf_np = confusion.numpy()

# Normalize full confusion by row
row_sums_full = conf_np.sum(axis=1, keepdims=True) + 1e-9
conf_norm = conf_np / row_sums_full

plt.figure(figsize=(7, 6))
plt.imshow(conf_norm, cmap="Blues")
plt.colorbar()
plt.xticks(np.arange(num_classes), classes, rotation=90)
plt.yticks(np.arange(num_classes), classes)
plt.xlabel("Predicted character")
plt.ylabel("True character")
plt.title("EMNIST Balanced – Confusion Matrix (normalized)")
plt.tight_layout()
plt.savefig(f"confusion_{filename}.pdf")
plt.show()

# -----------------------
# TOP-K MISCLASSIFICATIONS (NO DIAGONAL)
# -----------------------
conf_no_diag = conf_np.copy()
np.fill_diagonal(conf_no_diag, 0)

errors_per_class = conf_no_diag.sum(axis=1)
top_classes_idx = np.argsort(errors_per_class)[::-1][:K]

reduced_conf = conf_no_diag[np.ix_(top_classes_idx, top_classes_idx)]

row_sums_red = reduced_conf.sum(axis=1, keepdims=True) + 1e-9
reduced_conf_norm = reduced_conf / row_sums_red

reduced_labels = [classes[i] for i in top_classes_idx]

plt.figure(figsize=(8, 7))
plt.imshow(reduced_conf_norm, cmap="Blues")
plt.colorbar()
plt.xticks(np.arange(K), reduced_labels, rotation=45)
plt.yticks(np.arange(K), reduced_labels)
plt.xlabel("Predicted character")
plt.ylabel("True character")
plt.title("Top Confused Characters – EMNIST")
plt.tight_layout()
plt.savefig(f"{filename}-misclassification.pdf")
plt.show()
