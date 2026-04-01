"""Generalization gap (train_acc - val_acc) over epochs for all conditions."""
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

fig, ax = plt.subplots(figsize=(10, 6))

for key in order:
    histories = grouped[key]
    gaps = np.array([np.array(h["train_acc"]) - np.array(h["val_acc"]) for h in histories])
    mean_gap = gaps.mean(axis=0)
    epochs = np.arange(1, len(mean_gap) + 1)
    ax.plot(epochs, mean_gap, label=labels[key], color=colors[key], linestyle=linestyles[key], linewidth=1.5)

ax.axhline(y=0, color='black', linestyle=':', alpha=0.3)
ax.set_xlabel('Epoch')
ax.set_ylabel('Generalization Gap (Train Acc - Val Acc)')
ax.set_title('Generalization Gap by Dropout Type and Rate')
ax.legend(fontsize=9, loc='upper left')

plt.tight_layout()
plt.savefig("analysis/figures/generalization_gap.png", dpi=300, bbox_inches='tight')
plt.savefig("analysis/figures/generalization_gap.pdf", bbox_inches='tight')
print("Saved generalization_gap.png/pdf")
