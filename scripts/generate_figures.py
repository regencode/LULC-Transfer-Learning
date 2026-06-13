#!/usr/bin/env python3
"""Generate publication-quality figures for the paper.

Reads from the CSV files produced by parse_results.py.
Outputs PNG and PDF figures into the figures/ directory.
"""

import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

CLASSES = [
    "impervious_surface",
    "building",
    "low_vegetation",
    "tree",
    "car",
    "clutter",
]

CLASS_DISPLAY = {
    "impervious_surface": "Impervious",
    "building": "Building",
    "low_vegetation": "Low Veg.",
    "tree": "Tree",
    "car": "Car",
    "clutter": "Clutter",
}

BACKBONE_DISPLAY = {
    "resnet50": "ResNet-50",
    "resnet101": "ResNet-101",
    "convnext_s": "ConvNeXt-S",
    "convnext_b": "ConvNeXt-B",
    "vmamba_s": "VMamba-S",
    "vmamba_b": "VMamba-B",
    "mambavision_s": "MambaVis-S",
    "mambavision_b": "MambaVis-B",
}

BACKBONE_ORDER = [
    "resnet50",
    "resnet101",
    "convnext_s",
    "convnext_b",
    "vmamba_s",
    "vmamba_b",
    "mambavision_s",
    "mambavision_b",
]

FAMILY_COLORS = {
    "resnet": "#7B97A3",
    "convnext": "#B5B36E",
    "vmamba": "#D4885E",
    "mambavision": "#C05A7C",
}

MARKER_HEAD = {
    "deeplabv3plus": "o",
    "upernet": "s",
}

HEAD_DISPLAY = {
    "deeplabv3plus": "DeepLabV3+",
    "upernet": "UPerNet",
}


def load_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def family_of(backbone):
    if "resnet" in backbone:
        return "resnet"
    if "convnext" in backbone:
        return "convnext"
    if "vmamba" in backbone:
        return "vmamba"
    if "mambavision" in backbone:
        return "mambavision"
    return "other"


def fig_dir(base):
    d = os.path.join(base, "figures")
    os.makedirs(d, exist_ok=True)
    return d


def save_fig(fig, outdir, name):
    for ext in ("pdf", "png"):
        path = os.path.join(outdir, f"{name}.{ext}")
        fig.savefig(path, bbox_inches="tight", dpi=300)
    print(f"  Saved {name}")
    plt.close(fig)


def plot_miou_bars(main_rows, outdir):
    fig, ax = plt.subplots(figsize=(7, 3.2))

    x = np.arange(len(BACKBONE_ORDER))
    width = 0.35

    dl3_rows = {r["backbone"]: r for r in main_rows if r["head"] == "deeplabv3plus"}
    up_rows = {r["backbone"]: r for r in main_rows if r["head"] == "upernet"}

    vals_dl3 = [float(dl3_rows[bb]["mIoU"]) for bb in BACKBONE_ORDER]
    vals_up = [float(up_rows[bb]["mIoU"]) for bb in BACKBONE_ORDER]

    bars1 = ax.bar(x - width / 2, vals_dl3, width, label="DeepLabV3+", color="#5B8DB8", edgecolor="white", linewidth=0.5)
    bars2 = ax.bar(x + width / 2, vals_up, width, label="UPerNet", color="#E8915F", edgecolor="white", linewidth=0.5)

    ax.set_ylabel("mIoU (%)")
    ax.set_xticks(x)
    ax.set_xticklabels([BACKBONE_DISPLAY[bb] for bb in BACKBONE_ORDER], rotation=30, ha="right")
    ax.legend(loc="lower right", framealpha=0.9)
    ax.set_ylim(70.0, 73.5)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)

    best = max(max(vals_dl3), max(vals_up))
    ax.axhline(y=best, color="gray", linestyle=":", linewidth=0.8, alpha=0.6)

    fig.tight_layout()
    save_fig(fig, outdir, "miou_bar_comparison")


def plot_efficiency_scatter(main_rows, outdir):
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.2))

    for ax_idx, (metric_x, xlabel) in enumerate([
        ("params", "Parameters (M)"),
        ("flops", "FLOPs (G)"),
    ]):
        ax = axes[ax_idx]
        for head in ("deeplabv3plus", "upernet"):
            head_rows = [r for r in main_rows if r["head"] == head]
            for r in head_rows:
                bb = r["backbone"]
                fam = family_of(bb)
                x_val = float(r[metric_x])
                if metric_x == "params":
                    x_val /= 1e6
                elif metric_x == "flops":
                    x_val /= 1e9
                y_val = float(r["mIoU"])
                ax.scatter(
                    x_val, y_val,
                    c=FAMILY_COLORS[fam],
                    marker=MARKER_HEAD[head],
                    s=60,
                    edgecolors="white",
                    linewidths=0.5,
                    zorder=3,
                    label=f"{BACKBONE_DISPLAY[bb]} ({HEAD_DISPLAY[head]})" if ax_idx == 0 and head == "deeplabv3plus" else None,
                )

        ax.set_xlabel(xlabel)
        if ax_idx == 0:
            ax.set_ylabel("mIoU (%)")
        ax.set_ylim(70.5, 73.2)
        ax.grid(True, alpha=0.3)
        ax.set_axisbelow(True)

    family_patches = []
    seen_families = set()
    for fam in ("resnet", "convnext", "vmamba", "mambavision"):
        family_patches.append(plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=FAMILY_COLORS[fam], markersize=8, label=fam.capitalize().replace("Mambavision", "MambaVision").replace("Vmamba", "VMamba")))
    head_patches = [
        plt.Line2D([0], [0], marker=MARKER_HEAD["deeplabv3plus"], color="w", markerfacecolor="gray", markersize=8, label="DeepLabV3+"),
        plt.Line2D([0], [0], marker=MARKER_HEAD["upernet"], color="w", markerfacecolor="gray", markersize=8, label="UPerNet"),
    ]
    axes[1].legend(handles=family_patches + head_patches, loc="lower right", fontsize=7, framealpha=0.9)

    fig.tight_layout()
    save_fig(fig, outdir, "efficiency_scatter")


def plot_perclass_heatmap(main_rows, outdir):
    fig, ax = plt.subplots(figsize=(7, 4.5))

    labels = [f"{BACKBONE_DISPLAY[r['backbone']]}\n{HEAD_DISPLAY[r['head']]}" for r in main_rows]

    data = np.zeros((len(main_rows), len(CLASSES)))
    for i, r in enumerate(main_rows):
        for j, cls in enumerate(CLASSES):
            val = r.get(f"{cls}_IoU", "")
            data[i, j] = float(val) if val else np.nan

    im = ax.imshow(data, aspect="auto", cmap="YlOrRd", vmin=35, vmax=92)
    ax.set_xticks(range(len(CLASSES)))
    ax.set_xticklabels([CLASS_DISPLAY[c] for c in CLASSES])
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=7)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=6, color="white" if v > 65 else "black")

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("IoU (%)")
    fig.tight_layout()
    save_fig(fig, outdir, "perclass_heatmap")


def plot_perclass_grouped_bars(main_rows, outdir):
    n_backbones = len(BACKBONE_ORDER)
    n_classes = len(CLASSES)

    dl3 = {r["backbone"]: r for r in main_rows if r["head"] == "deeplabv3plus"}
    up = {r["backbone"]: r for r in main_rows if r["head"] == "upernet"}

    fig, axes = plt.subplots(2, 3, figsize=(10, 5.5), sharey=True)
    axes = axes.flatten()

    for cls_idx, cls in enumerate(CLASSES):
        ax = axes[cls_idx]
        x = np.arange(n_backbones)
        width = 0.35
        vals_dl3 = [float(dl3[bb][f"{cls}_IoU"]) for bb in BACKBONE_ORDER]
        vals_up = [float(up[bb][f"{cls}_IoU"]) for bb in BACKBONE_ORDER]

        ax.bar(x - width / 2, vals_dl3, width, color="#5B8DB8", edgecolor="white", linewidth=0.3)
        ax.bar(x + width / 2, vals_up, width, color="#E8915F", edgecolor="white", linewidth=0.3)

        ax.set_title(CLASS_DISPLAY[cls], fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels([BACKBONE_DISPLAY[bb].replace("MambaVision", "MVis").replace("ConvNeXt", "CNX") for bb in BACKBONE_ORDER], rotation=45, ha="right", fontsize=6)
        ax.grid(axis="y", alpha=0.3)
        ax.set_axisbelow(True)

    axes[0].set_ylabel("IoU (%)")
    axes[3].set_ylabel("IoU (%)")
    axes[0].legend(["DeepLabV3+", "UPerNet"], fontsize=7, loc="lower right")

    fig.tight_layout()
    save_fig(fig, outdir, "perclass_grouped_bars")


def plot_latency_miou(main_rows, outdir):
    fig, ax = plt.subplots(figsize=(5, 3.5))

    for r in main_rows:
        bb = r["backbone"]
        fam = family_of(bb)
        head = r["head"]
        lat = float(r["latency_ms"])
        miou = float(r["mIoU"])
        ax.scatter(
            lat, miou,
            c=FAMILY_COLORS[fam],
            marker=MARKER_HEAD[head],
            s=70,
            edgecolors="white",
            linewidths=0.5,
            zorder=3,
        )

    ax.set_xlabel("Inference Latency (ms)")
    ax.set_ylabel("mIoU (%)")
    ax.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

    family_patches = []
    for fam in ("resnet", "convnext", "vmamba", "mambavision"):
        family_patches.append(plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=FAMILY_COLORS[fam], markersize=8, label=fam.capitalize().replace("Mambavision", "MambaVision").replace("Vmamba", "VMamba")))
    head_patches = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="gray", markersize=8, label="DeepLabV3+"),
        plt.Line2D([0], [0], marker="s", color="w", markerfacecolor="gray", markersize=8, label="UPerNet"),
    ]
    ax.legend(handles=family_patches + head_patches, loc="lower right", fontsize=7, framealpha=0.9)

    fig.tight_layout()
    save_fig(fig, outdir, "latency_vs_miou")


def plot_ablation_comparison(main_rows, ablation_rows, outdir):
    ablation_bb = {"convnext_b", "vmamba_b", "mambavision_b"}
    ablation_lookup = {(r["backbone"], r["head"]): r for r in ablation_rows}

    bbs = ["convnext_b", "vmamba_b", "mambavision_b"]
    fig, axes = plt.subplots(1, 2, figsize=(6, 3))

    for ax_idx, head in enumerate(("deeplabv3plus", "upernet")):
        ax = axes[ax_idx]
        x = np.arange(len(bbs))
        width = 0.35

        vals_aux = []
        vals_noaux = []
        for bb in bbs:
            key_aux = next((r for r in main_rows if r["backbone"] == bb and r["head"] == head), None)
            vals_aux.append(float(key_aux["mIoU"]) if key_aux else 0)

            key_noaux = ablation_lookup.get((bb, head))
            vals_noaux.append(float(key_noaux["mIoU"]) if key_noaux else 0)

        ax.bar(x - width / 2, vals_aux, width, label="aux=True", color="#5B8DB8", edgecolor="white", linewidth=0.5)
        ax.bar(x + width / 2, vals_noaux, width, label="aux=False", color="#B5B5B5", edgecolor="white", linewidth=0.5)

        ax.set_xticks(x)
        ax.set_xticklabels([BACKBONE_DISPLAY[bb] for bb in bbs], fontsize=8)
        ax.set_title(HEAD_DISPLAY[head], fontsize=10)
        ax.set_ylabel("mIoU (%)")
        ax.set_ylim(71.5, 73.2)
        ax.grid(axis="y", alpha=0.3)
        ax.set_axisbelow(True)
        ax.legend(fontsize=7)

    fig.tight_layout()
    save_fig(fig, outdir, "ablation_aux")


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    outdir = fig_dir(base_dir)

    main_rows = load_csv(os.path.join(base_dir, "results_main.csv"))
    ablation_rows = load_csv(os.path.join(base_dir, "results_ablation.csv"))
    perclass_main = load_csv(os.path.join(base_dir, "results_perclass_main.csv"))

    sorted_main = sorted(main_rows, key=lambda r: (BACKBONE_ORDER.index(r["backbone"]), 0 if r["head"] == "deeplabv3plus" else 1))
    sorted_perclass = sorted(perclass_main, key=lambda r: (BACKBONE_ORDER.index(r["backbone"]), 0 if r["head"] == "deeplabv3plus" else 1))

    print("Generating figures...")
    plot_miou_bars(sorted_main, outdir)
    plot_efficiency_scatter(sorted_main, outdir)
    plot_perclass_heatmap(sorted_perclass, outdir)
    plot_perclass_grouped_bars(sorted_perclass, outdir)
    plot_latency_miou(sorted_main, outdir)
    plot_ablation_comparison(sorted_main, ablation_rows, outdir)

    print(f"\nAll figures saved to {outdir}")


if __name__ == "__main__":
    main()
