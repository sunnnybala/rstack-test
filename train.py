"""
Dropout Type vs Rate Comparison in Simple CNNs on CIFAR-10.
Runs on Modal cloud GPUs with parallel execution.

Factorial design: 3 dropout types (Standard, Spatial, DropBlock) x 2 rates (0.3, 0.5) + baseline.
"""

import json
import os
import random
import time

import modal

# ---------------------------------------------------------------------------
# Modal setup
# ---------------------------------------------------------------------------
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch", "torchvision", "numpy", "matplotlib"
)
app = modal.App("dropout-comparison")
vol = modal.Volume.from_name("dropout-results", create_if_missing=True)


# ---------------------------------------------------------------------------
# Single experiment runner (runs on cloud GPU)
# ---------------------------------------------------------------------------
@app.function(gpu="T4", image=image, timeout=1800, volumes={"/output": vol})
def run_single(dropout_type: str, dropout_rate: float, seed: int, epochs: int, smoke_test: bool):
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, Subset
    import torchvision
    import torchvision.transforms as transforms

    # --- DropBlock2D ---
    class DropBlock2D(nn.Module):
        def __init__(self, drop_prob=0.1, block_size=3):
            super().__init__()
            self.drop_prob = drop_prob
            self.block_size = block_size

        def forward(self, x):
            if not self.training or self.drop_prob == 0.0:
                return x
            _, _, h, w = x.shape
            gamma = (self.drop_prob * h * w) / (
                self.block_size ** 2 * (h - self.block_size + 1) * (w - self.block_size + 1)
            )
            mask = (torch.rand(x.shape[0], *x.shape[2:], device=x.device) < gamma).float()
            mask = mask.unsqueeze(1)
            mask = nn.functional.max_pool2d(
                mask, kernel_size=self.block_size, stride=1, padding=self.block_size // 2
            )
            mask = 1.0 - mask
            mask = mask.expand_as(x)
            count = mask.numel()
            count_ones = mask.sum()
            if count_ones == 0:
                return torch.zeros_like(x)
            return x * mask * (count / count_ones)

    # --- SimpleCNN ---
    class SimpleCNN(nn.Module):
        def __init__(self, d_type, d_rate, block_size=3):
            super().__init__()

            def make_conv_drop():
                if d_type == "none" or d_rate == 0.0:
                    return nn.Identity()
                elif d_type == "standard":
                    return nn.Dropout(p=d_rate)  # element-wise
                elif d_type == "spatial":
                    return nn.Dropout2d(p=d_rate)  # channel-wise
                elif d_type == "dropblock":
                    return DropBlock2D(drop_prob=d_rate, block_size=block_size)

            self.block1 = nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
                make_conv_drop(), nn.MaxPool2d(2),
            )
            self.block2 = nn.Sequential(
                nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                make_conv_drop(), nn.MaxPool2d(2),
            )
            self.block3 = nn.Sequential(
                nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
                make_conv_drop(), nn.MaxPool2d(2),
            )
            fc_drop = nn.Dropout(p=d_rate) if d_rate > 0 else nn.Identity()
            self.classifier = nn.Sequential(
                nn.Flatten(), nn.Linear(128 * 4 * 4, 256), nn.ReLU(inplace=True),
                fc_drop, nn.Linear(256, 10),
            )

        def forward(self, x):
            return self.classifier(self.block3(self.block2(self.block1(x))))

    # --- Seed ---
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    label = f"{dropout_type}_p{dropout_rate}" if dropout_type != "none" else "no_dropout"
    print(f"[{label} seed={seed}] Device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu'})")

    # --- Data ---
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    data_dir = "/tmp/cifar10"
    full_train = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform_train)
    test_set = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform_test)

    indices = list(range(len(full_train)))
    np.random.shuffle(indices)
    if smoke_test:
        train_idx, val_idx = indices[:500], indices[500:600]
        test_set = Subset(test_set, list(range(200)))
    else:
        train_idx, val_idx = indices[:45000], indices[45000:]

    train_loader = DataLoader(Subset(full_train, train_idx), batch_size=128, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(Subset(full_train, val_idx), batch_size=256, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False, num_workers=2, pin_memory=True)

    # --- Train ---
    model = SimpleCNN(dropout_type, dropout_rate).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "lr": []}
    best_val_acc = 0.0
    epochs_to_80 = None
    start = time.time()

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        t_loss, t_correct, t_total = 0.0, 0, 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            t_loss += loss.item() * inputs.size(0)
            t_correct += outputs.max(1)[1].eq(targets).sum().item()
            t_total += targets.size(0)
        train_loss, train_acc = t_loss / t_total, t_correct / t_total

        # Val
        model.eval()
        v_loss, v_correct, v_total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                v_loss += loss.item() * inputs.size(0)
                v_correct += outputs.max(1)[1].eq(targets).sum().item()
                v_total += targets.size(0)
        val_loss, val_acc = v_loss / v_total, v_correct / v_total
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["lr"].append(optimizer.param_groups[0]["lr"])

        if val_acc > best_val_acc:
            best_val_acc = val_acc
        if epochs_to_80 is None and val_acc >= 0.80:
            epochs_to_80 = epoch

        if epoch % 10 == 0 or epoch == 1:
            print(f"  [{label} s={seed}] Epoch {epoch:3d}/{epochs} | "
                  f"TrL:{train_loss:.4f} TrA:{train_acc:.4f} | VL:{val_loss:.4f} VA:{val_acc:.4f}")

    # Test
    model.eval()
    te_loss, te_correct, te_total = 0.0, 0, 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            te_loss += loss.item() * inputs.size(0)
            te_correct += outputs.max(1)[1].eq(targets).sum().item()
            te_total += targets.size(0)
    test_acc = te_correct / te_total
    test_loss = te_loss / te_total
    elapsed = time.time() - start

    result = {
        "dropout_type": dropout_type,
        "dropout_rate": dropout_rate,
        "seed": seed,
        "test_acc": test_acc,
        "test_loss": test_loss,
        "best_val_acc": best_val_acc,
        "final_train_acc": history["train_acc"][-1],
        "final_val_acc": history["val_acc"][-1],
        "generalization_gap": history["train_acc"][-1] - history["val_acc"][-1],
        "epochs_to_80_val": epochs_to_80,
        "elapsed_s": elapsed,
        "history": history,
    }

    print(f"  [{label} s={seed}] DONE | TestAcc={test_acc:.4f} GenGap={result['generalization_gap']:.4f} Time={elapsed:.1f}s")

    # Save to volume
    run_name = f"{label}_s{seed}"
    os.makedirs(f"/output/{run_name}", exist_ok=True)
    with open(f"/output/{run_name}/result.json", "w") as f:
        json.dump(result, f, indent=2)
    vol.commit()

    return result


# ---------------------------------------------------------------------------
# Orchestrator (runs locally, fans out to cloud)
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def main(smoke_test: bool = False, epochs: int = 50):
    import numpy as np

    if smoke_test:
        epochs = 2
        seeds = [42]
        print("=== SMOKE TEST MODE ===")
    else:
        seeds = [42, 123, 456]

    conditions = [
        ("none", 0.0),
        ("standard", 0.3),
        ("standard", 0.5),
        ("spatial", 0.3),
        ("spatial", 0.5),
        ("dropblock", 0.3),
        ("dropblock", 0.5),
    ]

    # Build argument lists for parallel execution
    args_list = []
    for d_type, d_rate in conditions:
        for seed in seeds:
            args_list.append((d_type, d_rate, seed, epochs, smoke_test))

    print(f"Launching {len(args_list)} runs in parallel on Modal...")

    # Fan out with starmap for parallel cloud execution
    all_results = []
    for result in run_single.starmap(args_list):
        label = f"{result['dropout_type']}_p{result['dropout_rate']}" if result["dropout_type"] != "none" else "no_dropout"
        print(f"  Completed: {label} s={result['seed']} | TestAcc={result['test_acc']:.4f}")
        all_results.append(result)

    # Save combined results locally
    os.makedirs("output", exist_ok=True)

    summary = [{k: v for k, v in r.items() if k != "history"} for r in all_results]
    with open("output/metrics.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open("output/full_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # Summary table
    from collections import defaultdict
    grouped = defaultdict(list)
    for r in all_results:
        key = f"{r['dropout_type']}_p{r['dropout_rate']}" if r["dropout_type"] != "none" else "no_dropout"
        grouped[key].append(r)

    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"{'Condition':<25} {'TestAcc':>14} {'ValAcc':>10} {'GenGap':>10} {'Ep→80%':>8}")
    print("-" * 70)

    for key, runs in grouped.items():
        ta = [r["test_acc"] for r in runs]
        va = [r["final_val_acc"] for r in runs]
        gg = [r["generalization_gap"] for r in runs]
        e8 = [r["epochs_to_80_val"] for r in runs if r["epochs_to_80_val"] is not None]
        print(f"{key:<25} {np.mean(ta):.4f}±{np.std(ta):.4f} {np.mean(va):.4f}     {np.mean(gg):+.4f}     {np.mean(e8) if e8 else float('nan'):>5.1f}")

    # CSV
    import csv
    with open("output/summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["condition", "dropout_type", "dropout_rate", "seed",
                     "test_acc", "val_acc", "train_acc", "gen_gap", "epochs_to_80", "elapsed_s"])
        for r in all_results:
            lbl = f"{r['dropout_type']}_p{r['dropout_rate']}" if r["dropout_type"] != "none" else "no_dropout"
            w.writerow([lbl, r["dropout_type"], r["dropout_rate"], r["seed"],
                        f"{r['test_acc']:.4f}", f"{r['final_val_acc']:.4f}",
                        f"{r['final_train_acc']:.4f}", f"{r['generalization_gap']:.4f}",
                        r["epochs_to_80_val"], f"{r['elapsed_s']:.1f}"])

    print(f"\nResults saved to output/")
