#!/usr/bin/env python3
"""Parse results.txt, ablation_results.md, and profiling.md into CSV files.

Produces:
    results_main.csv           - 16 main experiments (aux=True), aggregate + profiling
    results_ablation.csv       - 3 ablation experiments (aux=False), aggregate + profiling
    results_perclass_main.csv  - per-class metrics for 16 main experiments
    results_perclass_ablation.csv - per-class metrics for 3 ablation experiments
"""

import csv
import re
import sys

CLASSES = [
    "impervious_surface",
    "building",
    "low_vegetation",
    "tree",
    "car",
    "clutter",
]

METRICS = ["IoU", "Acc", "Dice", "Fscore", "Precision", "Recall"]

AGGREGATE_KEYS = ["mIoU", "aAcc", "mAcc", "mDice", "mFscore", "mPrecision", "mRecall"]

BACKBONE_RENAME = {
    "convnextbase": "convnext_b",
    "convnextsmall": "convnext_s",
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


def parse_checkpoint_path(line):
    m = re.search(r"outputs/([^/]+)/best\.ckpt", line)
    if not m:
        return None
    name = m.group(1)
    parts = name.split("_")
    if "deeplabv3plus" in name:
        head = "deeplabv3plus"
        backbone = name.replace("_deeplabv3plus", "").replace("_upernet", "")
    elif "upernet" in name:
        head = "upernet"
        backbone = name.replace("_upernet", "").replace("_deeplabv3plus", "")
    else:
        return None
    return backbone, head


def parse_perclass_table(lines, start_idx):
    rows = {}
    class_metrics = ["IoU", "Acc", "Dice", "Fscore", "Precision", "Recall"]
    idx = start_idx
    while idx < len(lines):
        line = lines[idx].strip()
        if "Epoch(test)" in line:
            break
        m = re.match(r"\|\s+(\S+)\s+\|\s+([\d.]+)\s+\|\s+([\d.]+)\s+\|\s+([\d.]+)\s+\|\s+([\d.]+)\s+\|\s+([\d.]+)\s+\|\s+([\d.]+)\s+\|", line)
        if m:
            cls_name = m.group(1)
            values = [float(m.group(i)) for i in range(2, 8)]
            rows[cls_name] = dict(zip(class_metrics, values))
        idx += 1
    return rows


def parse_epoch_line(line):
    result = {}
    for key in AGGREGATE_KEYS:
        m = re.search(rf"{key}:\s+([\d.]+)", line)
        if m:
            result[key] = float(m.group(1))
    return result


def parse_results_file(path, aux=True):
    with open(path) as f:
        lines = f.readlines()

    results = []
    i = 0
    while i < len(lines):
        line = lines[i].rstrip()

        if "Load checkpoint from outputs/" not in line:
            i += 1
            continue

        parsed = parse_checkpoint_path(line)
        if not parsed:
            i += 1
            continue
        backbone, head = parsed

        class_header_idx = None
        epoch_data = None
        j = i + 1
        while j < len(lines) and j < i + 50:
            if "|       Class" in lines[j] and class_header_idx is None:
                class_header_idx = j
            if "Epoch(test)" in lines[j]:
                epoch_data = parse_epoch_line(lines[j])
                break
            j += 1

        if class_header_idx is None or epoch_data is None:
            i += 1
            continue

        perclass = parse_perclass_table(lines, class_header_idx)

        row = {
            "backbone": backbone,
            "head": head,
            "aux": aux,
        }
        row.update(epoch_data)

        for cls in CLASSES:
            if cls in perclass:
                for metric in METRICS:
                    row[f"{cls}_{metric}"] = perclass[cls][metric]

        results.append(row)
        i = j + 1

    return results


def parse_profiling_file(path):
    with open(path) as f:
        lines = f.readlines()

    results = []
    for line in lines:
        line = line.strip()
        if line.startswith("-") or line.startswith("Backbone"):
            continue
        parts = line.split()
        if len(parts) < 8:
            continue

        bb_raw = parts[0]
        head = parts[1]
        aux_str = parts[2]
        params = parts[3]
        flops = parts[4]
        latency_str = parts[5]
        latency_std_str = parts[7] if len(parts) > 7 else "0"
        throughput = parts[8] if len(parts) > 8 else parts[6]
        gpu_mem = parts[9] if len(parts) > 9 else parts[7]

        try:
            params_val = float(params.replace("M", "").replace("K", ""))
            if "K" in params:
                params_val *= 1000
            elif "M" in params:
                params_val *= 1e6
        except ValueError:
            params_val = None

        flops_val = None
        try:
            if "G" in flops:
                flops_val = float(flops.replace("G", "")) * 1e9
            elif "M" in flops:
                flops_val = float(flops.replace("M", "")) * 1e6
        except ValueError:
            pass

        try:
            lat_val = float(latency_str)
        except ValueError:
            lat_val = None

        try:
            lat_std = float(latency_std_str)
        except ValueError:
            lat_std = None

        try:
            tp_val = float(throughput)
        except ValueError:
            tp_val = None

        try:
            mem_val = float(gpu_mem)
        except ValueError:
            mem_val = None

        bb_normalized = BACKBONE_RENAME.get(bb_raw, bb_raw)

        results.append({
            "backbone": bb_normalized,
            "head": head,
            "aux": aux_str == "True",
            "params": params_val,
            "flops": flops_val,
            "latency_ms": lat_val,
            "latency_std_ms": lat_std,
            "throughput": tp_val,
            "gpu_mem_mb": mem_val,
        })

    return results


def merge_profiling(metrics_rows, profiling_rows):
    prof_lookup = {}
    for p in profiling_rows:
        key = (p["backbone"], p["head"], p["aux"])
        prof_lookup[key] = p

    for row in metrics_rows:
        key = (row["backbone"], row["head"], row["aux"])
        prof = prof_lookup.get(key, {})
        row["params"] = prof.get("params")
        row["flops"] = prof.get("flops")
        row["latency_ms"] = prof.get("latency_ms")
        row["latency_std_ms"] = prof.get("latency_std_ms")
        row["throughput"] = prof.get("throughput")
        row["gpu_mem_mb"] = prof.get("gpu_mem_mb")


def sort_key(row):
    bb = row["backbone"]
    try:
        bb_idx = BACKBONE_ORDER.index(bb)
    except ValueError:
        bb_idx = len(BACKBONE_ORDER)
    head_idx = 0 if row["head"] == "deeplabv3plus" else 1
    return (bb_idx, head_idx)


def write_csv(rows, path, fieldnames):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Wrote {len(rows)} rows to {path}")


def main():
    base_dir = __file__.replace("scripts/parse_results.py", "").rstrip("/")
    if not base_dir:
        base_dir = "."

    print("Parsing results.txt ...")
    main_results = parse_results_file(f"{base_dir}/results.txt", aux=True)
    print(f"  Found {len(main_results)} main experiments")

    print("Parsing ablation_results.md ...")
    ablation_results = parse_results_file(f"{base_dir}/ablation_results.md", aux=False)
    print(f"  Found {len(ablation_results)} ablation experiments")

    print("Parsing profiling.md ...")
    profiling = parse_profiling_file(f"{base_dir}/profiling.md")
    print(f"  Found {len(profiling)} profiling entries")

    merge_profiling(main_results, profiling)
    merge_profiling(ablation_results, profiling)

    main_results.sort(key=sort_key)
    ablation_results.sort(key=sort_key)

    id_cols = ["backbone", "head", "aux"]
    agg_cols = AGGREGATE_KEYS
    prof_cols = ["params", "flops", "latency_ms", "latency_std_ms", "throughput", "gpu_mem_mb"]
    perclass_cols = [f"{cls}_{m}" for cls in CLASSES for m in METRICS]

    write_csv(main_results, f"{base_dir}/results_main.csv", id_cols + agg_cols + prof_cols)
    write_csv(ablation_results, f"{base_dir}/results_ablation.csv", id_cols + agg_cols + prof_cols)
    write_csv(main_results, f"{base_dir}/results_perclass_main.csv", id_cols + perclass_cols)
    write_csv(ablation_results, f"{base_dir}/results_perclass_ablation.csv", id_cols + perclass_cols)

    print("\nDone.")


if __name__ == "__main__":
    main()
