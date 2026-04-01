"""Bar chart comparing test accuracy across all conditions with error bars."""
import json
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

plt.rcParams.update({
    'font.size': 12,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.figsize': (10, 6),
    'figure.dpi': 150,
})

with open("output/full_results.json") as f:
    results = json.load(f)

# Group by condition
grouped = defaultdict(list)
order = []
for r in results:
    key = f"{r['dropout_type']}_p{r['dropout_rate']}" if r["dropout_type"] != "none" else "no_dropout"
    if key not in order:
        order.append(key)
    grouped[key].append(r["test_acc"])

labels = {
    "no_dropout": "No Dropout",
    "standard_p0.3": "Standard\np=0.3",
    "standard_p0.5": "Standard\np=0.5",
    "spatial_p0.3": "Spatial\np=0.3",
    "spatial_p0.5": "Spatial\np=0.5",
    "dropblock_p0.3": "DropBlock\np=0.3",
    "dropblock_p0.5": "DropBlock\np=0.5",
}

colors = {
    "no_dropout": "#888888",
    "standard_p0.3": "#4C72B0",
    "standard_p0.5": "#4C72B0",
    "spatial_p0.3": "#DD8452",
    "spatial_p0.5": "#DD8452",
    "dropblock_p0.3": "#55A868",
    "dropblock_p0.5": "#55A868",
}

alphas = {k: (1.0 if "0.3" in k or k == "no_dropout" else 0.6) for k in order}

means = [np.mean(grouped[k]) for k in order]
stds = [np.std(grouped[k]) for k in order]
x = np.arange(len(order))

fig, ax = plt.subplots()
bar_colors = [colors[k] for k in order]
bar_alphas = [alphas[k] for k in order]
bars = ax.bar(x, means, yerr=stds, capsize=4,
              color=bar_colors, edgecolor='black', linewidth=0.5)
for bar, a in zip(bars, bar_alphas):
    bar.set_alpha(a)

ax.set_ylabel('Test Accuracy')
ax.set_title('Test Accuracy by Dropout Type and Rate (CIFAR-10, 50 epochs)')
ax.set_xticks(x)
ax.set_xticklabels([labels[k] for k in order])
ax.set_ylim(0.4, 0.92)
ax.axhline(y=means[0], color='gray', linestyle='--', alpha=0.5, label='No dropout baseline')

# Add value labels on bars
for i, (m, s) in enumerate(zip(means, stds)):
    ax.text(i, m + s + 0.01, f'{m:.3f}', ha='center', va='bottom', fontsize=9)

ax.legend(loc='lower right')
plt.tight_layout()
plt.savefig("analysis/figures/test_accuracy_comparison.png", dpi=300, bbox_inches='tight')
plt.savefig("analysis/figures/test_accuracy_comparison.pdf", bbox_inches='tight')
print("Saved test_accuracy_comparison.png/pdf")
