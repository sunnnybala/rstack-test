"""Heatmap: dropout type vs rate showing test accuracy (the key figure)."""
import json
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

plt.rcParams.update({
    'font.size': 12,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 150,
})

with open("output/full_results.json") as f:
    results = json.load(f)

grouped = defaultdict(list)
for r in results:
    key = (r["dropout_type"], r["dropout_rate"])
    grouped[key].append(r["test_acc"])

types = ["standard", "spatial", "dropblock"]
rates = [0.3, 0.5]
type_labels = ["Standard\nDropout", "Spatial\nDropout", "DropBlock"]

# Build matrix
matrix = np.zeros((len(types), len(rates)))
for i, t in enumerate(types):
    for j, r in enumerate(rates):
        matrix[i, j] = np.mean(grouped[(t, r)])

fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=0.5, vmax=0.85)

# Labels
ax.set_xticks(range(len(rates)))
ax.set_xticklabels([f'p = {r}' for r in rates])
ax.set_yticks(range(len(types)))
ax.set_yticklabels(type_labels)
ax.set_xlabel('Dropout Rate')
ax.set_title('Test Accuracy: Dropout Type vs Rate')

# Annotate cells
for i in range(len(types)):
    for j in range(len(rates)):
        stds = np.std(grouped[(types[i], rates[j])])
        ax.text(j, i, f'{matrix[i,j]:.3f}\n±{stds:.3f}',
                ha='center', va='center', fontsize=13, fontweight='bold',
                color='white' if matrix[i,j] < 0.65 else 'black')

plt.colorbar(im, label='Test Accuracy')
plt.tight_layout()
plt.savefig("analysis/figures/type_vs_rate_heatmap.png", dpi=300, bbox_inches='tight')
plt.savefig("analysis/figures/type_vs_rate_heatmap.pdf", bbox_inches='tight')
print("Saved type_vs_rate_heatmap.png/pdf")
