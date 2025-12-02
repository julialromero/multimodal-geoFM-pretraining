import json
import re
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt

# Root directory that holds all unified evaluation outputs.
UNIFIED_EVAL_ROOT = Path(__file__).resolve().parent

# Fractions of the dataset we want to visualize in bar/table plots.
TARGET_FRACTIONS: List[float] = [0.01, 0.1, 1.0]
# Fractions used for the line plot (keep early/late splits visible).
LINE_PLOT_FRACTIONS: List[float] = [0.01, 0.1, 1.0]
# Collect at least the union so all downstream plots have the values they need.
FRACTIONS_FOR_COLLECTION: List[float] = sorted(set(TARGET_FRACTIONS) | set(LINE_PLOT_FRACTIONS))

# Filenames that correspond to the backbone features (non-batchnorm).
LINEAR_FILENAME = "eurosat_backbone_metrics.json"
KNN_FILENAME = "eurosat_backbone_knn_metrics.json"

##### TEMPORARY SET OF MODELS #####
# PREFERRED_MODEL_ORDER: List[str] = [
#     "random rcf_13ch",
#     "croma",
#     "dofa_base_s2_13ch",
#     "scalemae_large_rgb",
#     "moco",
#     "dino",
#     "resnet50_s2_rgb_moco",
#     "ciip_resnet50_epoch20",
# ]

# ALLOWED_MODELS: List[str] = [
#     "random rcf_13ch",
#     "croma",
#     "dofa_base_s2_13ch",
#     "scalemae_large_rgb",
#     "moco",
#     "dino",
#     "resnet50_s2_rgb_moco",
#     "ciip_resnet50_epoch20",
# ]


##### FULL SET OF MODELS #####
# ALLOWED_MODELS: List[str] = [
#  "rcf_13ch",
#     "croma",
#     "dofa_base_s2_13ch",
#     "vitsmall16_s2_all_moco",
#     "scalemae_large_rgb",
#     "resnet18_s2_all_moco",
#     "moco",  # resnet50 all moco
#     "dino",
#     "resnet18_s2_rgb_moco",
#     "resnet50_s2_rgb_moco",
#     "resnet152_imagenet_rgb",
#     "random_resnet50_s2_12",
#     "ciip_resnet50_epoch20",
# ]

# PREFERRED_MODEL_ORDER: List[str] = [
#     "rcf_13ch",
#     "croma",
#     "dofa_base_s2_13ch",
#     "vitsmall16_s2_all_moco",
#     "scalemae_large_rgb",
#     "resnet18_s2_all_moco",
#     "moco",  # resnet50 all moco
#     "dino",
#     "resnet18_s2_rgb_moco",
#     "resnet50_s2_rgb_moco",
#     "resnet152_imagenet_rgb",
#     "random_resnet50_s2_12",
#     "ciip_resnet50_epoch20",
# ]

##### JUST CIIP AND RCF #####
PREFERRED_MODEL_ORDER: List[str] = [
    "rcf_13ch",
    "Vanilla CIIP (S2A)",
    "Hyperbolic CIIP (S2A)",
]

ALLOWED_MODELS: List[str] = [
    "rcf_13ch",
    "Vanilla CIIP (S2A)",
    "Hyperbolic CIIP (S2A)",
]

RUN_LABELS: Dict[str, str] = {
    "2025_11_22-08_31_28-model_resnet50-lr_0.001-b_6-j_6-p_amp_bfloat16": "Vanilla CIIP (S2A)",
    "2025_11_29-20_57_06-model_resnet50-lr_0.001-b_6-j_6-p_amp_bfloat16": "Hyperbolic CIIP (S2A)",
}

def parse_fraction_key(key: str) -> Optional[float]:
    """Convert JSON fraction keys to floats, handling comma decimals."""
    try:
        return float(key.replace(",", "."))
    except ValueError:
        return None


def select_fraction_accuracies(path: Path) -> Dict[float, Optional[float]]:
    """Load a metrics file and pull test_accuracy for the needed fractions."""
    with path.open() as f:
        raw = json.load(f)

    fraction_map: Dict[float, float] = {}
    for key, metrics in raw.items():
        frac = parse_fraction_key(key)
        if frac is None:
            continue
        if not isinstance(metrics, dict):
            continue
        test_acc = metrics.get("test_accuracy")
        if test_acc is None:
            continue
        fraction_map[round(frac, 3)] = test_acc

    selected: Dict[float, Optional[float]] = {}
    for target in FRACTIONS_FOR_COLLECTION:
        selected[target] = fraction_map.get(round(target, 3))
    return selected


def extract_model_label(path: Path) -> Optional[str]:
    """Create a concise model label from the path, applying ciip rules."""
    try:
        idx = path.parts.index("unified_eval")
        after_root = path.parts[idx + 1 :]
    except ValueError:
        after_root = path.parts

    if not after_root:
        return None

    model_root = after_root[0]
    if model_root == "ciip_checkpoint":
        if "epoch_20" not in after_root:
            return None
        run_dir = after_root[2] if len(after_root) > 2 else ""
        return RUN_LABELS.get(run_dir)

    return model_root


def collect_results() -> Dict[str, Dict[str, Dict[float, Optional[float]]]]:
    """Gather knn and linear probe accuracies for all models."""
    results: Dict[str, Dict[str, Dict[float, Optional[float]]]] = {}
    target_files = {LINEAR_FILENAME: "linear", KNN_FILENAME: "knn"}

    for path in UNIFIED_EVAL_ROOT.rglob("eurosat_backbone*_metrics.json"):
        if path.name not in target_files:
            continue
        if "batchnorm" in path.name:
            continue

        model_label = extract_model_label(path)
        if model_label is None:
            continue
        if model_label not in ALLOWED_MODELS:
            continue

        method = target_files[path.name]
        fraction_accs = select_fraction_accuracies(path)
        model_entry = results.setdefault(model_label, {})
        model_entry[method] = fraction_accs

    return results


def sorted_models(models: List[str]) -> List[str]:
    """Sort models according to the preferred order, then alphabetically."""
    order_index = {name: idx for idx, name in enumerate(PREFERRED_MODEL_ORDER)}
    return sorted(models, key=lambda m: (order_index.get(m, len(order_index)), m))


def compute_model_colors(models: List[str]) -> Dict[str, str]:
    """Assign a stable color per model."""
    color_cycle = plt.cm.tab20.colors
    return {model: color_cycle[idx % len(color_cycle)] for idx, model in enumerate(models)}


def plot_results(results: Dict[str, Dict[str, Dict[float, Optional[float]]]]) -> Path:
    """Create subplots comparing knn vs linear for each fraction."""
    if not results:
        raise RuntimeError("No results found for the requested criteria.")

    models = sorted_models(list(results.keys()))
    fig, axes = plt.subplots(1, len(TARGET_FRACTIONS), figsize=(16, 6), sharey=True)

    bar_width = 0.35
    colors = {"linear": "#1f77b4", "knn": "#ff7f0e"}

    for ax, frac in zip(axes, TARGET_FRACTIONS):
        rows = []
        for model in models:
            model_data = results[model]
            lin = model_data.get("linear", {}).get(frac)
            knn = model_data.get("knn", {}).get(frac)
            if lin is None and knn is None:
                continue
            rows.append((model, lin, knn))

        if not rows:
            ax.set_title(f"Fraction {frac}")
            ax.set_xticks([])
            ax.set_ylabel("Test Accuracy")
            continue

        x = list(range(len(rows)))
        linear_vals = [r[1] for r in rows]
        knn_vals = [r[2] for r in rows]

        ax.bar(
            [v - bar_width / 2 for v in x],
            linear_vals,
            width=bar_width,
            label="Linear probe",
            color=colors["linear"],
        )
        ax.bar(
            [v + bar_width / 2 for v in x],
            knn_vals,
            width=bar_width,
            label="k-NN",
            color=colors["knn"],
        )

        ax.set_title(f"Fraction {frac}")
        ax.set_xticks(x)
        ax.set_xticklabels([r[0] for r in rows], rotation=45, ha="right")
        ax.set_ylim(0, 1)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        if ax is axes[0]:
            ax.set_ylabel("Test Accuracy")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.05))
    fig.tight_layout()

    output_path = UNIFIED_EVAL_ROOT / "eurosat_knn_vs_linear_probe.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    return output_path


def render_table(
    method: str,
    results: Dict[str, Dict[str, Dict[float, Optional[float]]]],
    output_name: str,
) -> Path:
    """Render a LaTeX-style table for a given method and save as PNG."""
    models = sorted_models(list(results.keys()))

    headers = ["Model"] + [str(f) for f in TARGET_FRACTIONS]
    rows: List[List[str]] = []
    best_by_frac: Dict[float, float] = {}

    for frac in TARGET_FRACTIONS:
        vals = [
            results.get(model, {}).get(method, {}).get(frac)
            for model in models
            if results.get(model, {}).get(method, {}).get(frac) is not None
        ]
        if vals:
            best_by_frac[frac] = max(vals)

    for model in models:
        row = [model]
        for frac in TARGET_FRACTIONS:
            val = results.get(model, {}).get(method, {}).get(frac)
            row.append(f"{val:.3f}" if val is not None else "N/A")
        rows.append(row)

    fig, ax = plt.subplots(figsize=(10, 0.6 * len(rows) + 1.5))
    ax.axis("off")

    table = ax.table(cellText=rows, colLabels=headers, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.3)

    # Style the header row
    num_cols = len(headers)
    for col in range(num_cols):
        header_cell = table[(0, col)]  # Header row is row 0
        header_cell._text.set_fontweight("bold")
        header_cell.set_facecolor("#E6E6E6")
        header_cell._text.set_color("black")
        header_cell.set_linewidth(2)

    # Bold the best value per fraction column (data rows only)
    for col_idx, frac in enumerate(TARGET_FRACTIONS, start=1):
        best = best_by_frac.get(frac)
        if best is None:
            continue
        for row_idx, model in enumerate(models, start=1):  # +1 to skip header row
            val = results.get(model, {}).get(method, {}).get(frac)
            if val is not None and abs(val - best) < 1e-6:
                table[(row_idx, col_idx)]._text.set_fontweight("bold")

    plt.tight_layout()
    output_path = UNIFIED_EVAL_ROOT / output_name
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def render_bump_chart(
    results: Dict[str, Dict[str, Dict[float, Optional[float]]]],
    model_colors: Dict[str, str],
) -> Path:
    """Render a bump chart ranking models across linear/kNN fractions."""
    models = sorted_models(list(results.keys()))
    conditions = [
        ("linear", 0.01, "LP 0.01"),
        ("linear", 0.1, "LP 0.1"),
        ("linear", 1.0, "LP 1.0"),
        ("knn", 0.01, "kNN 0.01"),
        ("knn", 0.1, "kNN 0.1"),
        ("knn", 1.0, "kNN 1.0"),
    ]

    # Compute ranks per condition.
    ranks: Dict[str, Dict[str, int]] = {m: {} for m in models}
    for method, frac, label in conditions:
        vals = []
        for model in models:
            val = results.get(model, {}).get(method, {}).get(frac)
            if val is not None:
                vals.append((model, val))
        vals.sort(key=lambda x: x[1], reverse=True)
        for rank, (model, _) in enumerate(vals, start=1):
            ranks[model][label] = rank

    fig, ax = plt.subplots(figsize=(13, 8))
    x_positions = list(range(len(conditions)))
    x_labels = [label for _, _, label in conditions]

    for model in models:
        y = []
        x = []
        for idx, (_, _, label) in enumerate(conditions):
            rank = ranks[model].get(label)
            if rank is None:
                continue
            x.append(idx)
            y.append(rank)
        if not x:
            continue
        color = model_colors[model]
        ax.plot(
            x,
            y,
            marker="o",
            linewidth=2,
            markersize=6,
            label=model,
            color=color,
            alpha=0.85,
        )
        # Annotate at the first and last available points to avoid misalignment.
        ax.text(
            x[0] - 0.12,
            y[0],
            model,
            ha="right",
            va="center",
            fontsize=11,
            color=color,
        )
        ax.text(
            x[-1] + 0.12,
            y[-1],
            model,
            ha="left",
            va="center",
            fontsize=11,
            color=color,
        )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, rotation=25, ha="right")
    ax.set_ylim(len(models) + 1, 0.5)  # invert so rank 1 is at the top
    ax.set_ylabel("") #Rank (top to bottom, top is Rank 1)")
    ax.tick_params(axis="y", left=False, labelleft=False)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_title("Model rank (Top is best) across Linear Probe (LP) and k-NN on EuroSAT", fontsize=14)

    fig.tight_layout()
    output_path = UNIFIED_EVAL_ROOT / "eurosat_bump_chart.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def render_line_plots(
    results: Dict[str, Dict[str, Dict[float, Optional[float]]]],
    model_colors: Dict[str, str],
) -> Path:
    """Render per-model line plots across fractions for linear and kNN."""
    models = sorted_models(list(results.keys()))
    x_positions = list(range(len(LINE_PLOT_FRACTIONS)))
    x_labels = [str(f) for f in LINE_PLOT_FRACTIONS]

    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    method_specs = [
        ("linear", "Linear probe", axes[0]),
        ("knn", "k-NN", axes[1]),
    ]

    for method, title, ax in method_specs:
        for model in models:
            xs = []
            ys = []
            for idx, frac in enumerate(LINE_PLOT_FRACTIONS):
                val = results.get(model, {}).get(method, {}).get(frac)
                if val is None:
                    continue
                xs.append(idx)
                ys.append(val)
            if not xs:
                continue
            ax.plot(
                xs,
                ys,
                marker="o",
                linewidth=2,
                markersize=6,
                color=model_colors[model],
                label=model,
                alpha=0.9,
            )
        ax.set_title(f"{title} test accuracy")
        ax.set_ylabel("Accuracy")
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    axes[-1].set_xticks(x_positions)
    axes[-1].set_xticklabels(x_labels)
    axes[-1].set_xlabel("Training fraction")

    # Single legend for clarity.
    axes[0].legend(bbox_to_anchor=(1.02, 1), loc="upper left")

    fig.tight_layout()
    output_path = UNIFIED_EVAL_ROOT / "eurosat_fraction_lineplots.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    results = collect_results()
    output_path = plot_results(results)
    linear_table_path = render_table("linear", results, "eurosat_linear_probe_table.png")
    knn_table_path = render_table("knn", results, "eurosat_knn_table.png")
    model_list = sorted_models(list(results.keys()))
    model_colors = compute_model_colors(model_list)
    bump_chart_path = render_bump_chart(results, model_colors)
    line_plot_path = render_line_plots(results, model_colors)

    print("Collected results (test_accuracy):")
    for model in sorted_models(list(results.keys())):
        methods = results[model]
        print(f"- {model}")
        for method in ("linear", "knn"):
            frac_values = methods.get(method)
            if not frac_values:
                continue
            formatted = ", ".join(
                f"{frac}: {acc:.3f}" if acc is not None else f"{frac}: N/A"
                for frac, acc in sorted(frac_values.items())
            )
            print(f"  {method}: {formatted}")

    print(f"\nPlot saved to: {output_path}")
    print(f"Linear probe table saved to: {linear_table_path}")
    print(f"k-NN table saved to: {knn_table_path}")
    print(f"Bump chart saved to: {bump_chart_path}")
    print(f"Line plots saved to: {line_plot_path}")


if __name__ == "__main__":
    main()
