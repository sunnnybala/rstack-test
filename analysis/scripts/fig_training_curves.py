"""Training loss and validation accuracy curves for all conditions (averaged over seeds)."""
import json
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

plt.rcParams.update({
    'font.size': 11,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 150,
})

with open("output/full_results.json") as f:
    results = json.load(f)

# Group histories by condition
grouped = defaultdict(list)
order = ["no_dropout", "standard_p0.3", "standard_p0.5",
         "spatial_p0.3", "spatial_p0.5", "dropblock_p0.3", "dropblock_p0.5"]

labels = {
    "no_dropout": "No Dropout",
    "standard_p0.3": "Standard p=0.3",
    "standard_p0.5": "Standard p=0.5",
    "spatial_p0.3": "Spatial p=0.3",
    "spatial_p0.5": "Spatial p=0.5",
    "dropblock_p0.3": "DropBlock p=0.3",
    "dropblock_p0.5": "DropBlock p=0.5",
}

colors = {
    "no_dropout": "#888888",
    "standard_p0.3": "#4C72B0",
    "standard_p0.5": "#7BA3D4",
    "spatial_p0.3": "#DD8452",
    "spatial_p0.5": "#EEAA77",
    "dropblock_p0.3": "#55A868",
    "dropblock_p0.5": "#88CC99",
}

linestyles = {k: ("-" if "0.3" in k or k == "no_dropout" else "--") for k in order}

for r in results:
    key = f"{r['dropout_type']}_p{r['dropout_rate']}" if r["dropout_type"] != "none" else "no_dropout"
    grouped[key].append(r["history"])

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Training Loss
ax = axes[0]
for key in order:
    histories = grouped[key]
    losses = np.array([h["train_loss"] for h in histories])
    mean_loss = losses.mean(axis=0)
    std_loss = losses.std(axis=0)
    epochs = np.arange(1, len(mean_loss) + 1)
    ax.plot(epochs, mean_loss, label=labels[key], color=colors[key], linestyle=linestyles[key], linewidth=1.5)
    ax.fill_between(epochs, mean_loss - std_loss, mean_loss + std_loss, color=colors[key], alpha=0.1)

ax.set_xlabel('Epoch')
ax.set_ylabel('Training Loss')
ax.set_title('Training Loss')
ax.legend(fontsize=8, loc='upper right')

# Plot 2: Validation Accuracy
ax = axes[1]
for key in order:
    histories = grouped[key]
    accs = np.array([h["val_acc"] for h in histories])
    mean_acc = accs.mean(axis=0)
    std_acc = accs.std(axis=0)
    epochs = np.arange(1, len(mean_acc) + 1)
    ax.plot(epochs, mean_acc, label=labels[key], color=colors[key], linestyle=linestyles[key], linewidth=1.5)
    ax.fill_between(epochs, mean_acc - std_acc, mean_acc + std_acc, color=colors[key], alpha=0.1)

ax.set_xlabel('Epoch')
ax.set_ylabel('Validation Accuracy')
ax.set_title('Validation Accuracy')
ax.legend(fontsize=8, loc='lower right')

plt.suptitle('Training Dynamics by Dropout Type and Rate (CIFAR-10)', fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig("analysis/figures/training_curves.png", dpi=300, bbox_inches='tight')
plt.savefig("analysis/figures/training_curves.pdf", bbox_inches='tight')
print("Saved training_curves.png/pdf")
