"""
Compare model-derived soundscape composition (soundscape_comparison_table.csv)
against a manual/expert characterization (soundscape_comparison_table_manual.csv)
for the sites where manual data is available.

Expects both CSVs to share a 'deploymentID' column and comparable columns for
biophony/anthrophony/geophony proportions. Adjust MANUAL_COLS below to match
the actual column names in your manual file.
"""

import pandas as pd
from scipy.stats import pearsonr, spearmanr

# --- File paths ---
model_path = "../../data/output/panns/soundscape_comparison_table.csv"
manual_path = "soundscape_comparison_table_manual.csv"

# --- Column mapping: model column -> manual column ---
# Adjust the right-hand side to match your manual file's actual column names.
COLUMN_MAP = {
    "pct_biophony": "biophony",
    "pct_anthropophony": "anthropophony",
    "pct_geophony": "geophony",
    "pct_silence": "silence",
}


def load_and_align(model_path, manual_path, column_map):
    model_df = pd.read_csv(model_path)
    manual_df = pd.read_csv(manual_path)

    merged = pd.merge(
        model_df, manual_df, on="deploymentID", how="inner", suffixes=("", "_manual")
    )

    # Keep only rows where ALL relevant manual columns are filled in (not NaN)
    manual_cols = list(column_map.values())
    complete_mask = merged[manual_cols].notna().all(axis=1)
    complete = merged[complete_mask].copy()

    dropped = len(merged) - len(complete)
    print(f"Matched sites (model + manual): {len(merged)}")
    print(f"Sites with complete manual data: {len(complete)}  (dropped {dropped} incomplete/missing)\n")

    return complete


def compute_correlations(df, column_map):
    results = []
    for model_col, manual_col in column_map.items():
        x = df[model_col]
        y = df[manual_col]

        pearson_r, pearson_p = pearsonr(x, y)
        spearman_r, spearman_p = spearmanr(x, y)
        mae = (x - y).abs().mean()
        bias = (x - y).mean()  # positive = model overestimates vs manual

        results.append({
            "category": model_col.replace("pct_", ""),
            "n_sites": len(df),
            "pearson_r": round(pearson_r, 3),
            "pearson_p": round(pearson_p, 4),
            "spearman_r": round(spearman_r, 3),
            "spearman_p": round(spearman_p, 4),
            "mean_abs_error_pct": round(mae, 2),
            "mean_bias_pct": round(bias, 2),
        })

    return pd.DataFrame(results)


def plot_model_vs_manual(df, plot_columns, output_path="model_vs_manual_scatter.png"):
    """
    One figure with one subplot per (model_col, manual_col, display_name) triplet
    in plot_columns. Each subplot: model (x) vs manual (y) scatter, 1:1 reference
    line, and the deploymentID printed above each point to flag which sites
    disagree most.
 
    plot_columns: list of tuples (model_col, manual_col, display_name)
    """
    import matplotlib.pyplot as plt
 
    n = len(plot_columns)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 6))
    if n == 1:
        axes = [axes]
 
    for ax, (model_col, manual_col, display_name) in zip(axes, plot_columns):
        x = df[model_col]
        y = df[manual_col]
 
        ax.scatter(x, y, s=60, color="steelblue", zorder=3, edgecolor="white", linewidth=0.5)
 
        # 1:1 reference line (perfect agreement)
        lo = min(x.min(), y.min())
        hi = max(x.max(), y.max())
        pad = (hi - lo) * 0.1 if hi > lo else 1
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad],
                linestyle="--", color="gray", linewidth=1, zorder=1, label="1:1 agreement")
 
        # Annotate each point with its deploymentID, above the dot
        for _, row in df.iterrows():
            ax.annotate(
                row["deploymentID"],
                (row[model_col], row[manual_col]),
                textcoords="offset points",
                xytext=(0, 8),
                ha="center",
                fontsize=8,
                color="black",
            )
 
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_xlabel(f"Model ({model_col})")
        ax.set_ylabel(f"Manual ({manual_col})")
        ax.set_title(display_name)
        ax.set_aspect("equal", adjustable="box")
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(alpha=0.3)
 
    fig.suptitle("Model vs. manual soundscape characterization", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot: {output_path}")
    return fig

if __name__ == "__main__":
    complete_df = load_and_align(model_path, manual_path, COLUMN_MAP)

    if complete_df.empty:
        print("No sites with complete matching data between model and manual tables.")
    else:
        results_df = compute_correlations(complete_df, COLUMN_MAP)
        print(results_df.to_string(index=False))

        output_path = "model_vs_manual_correlation.csv"
        results_df.to_csv(output_path, index=False)
        print(f"\nSaved: {output_path}")

        # Also save the aligned, complete-case comparison table itself for inspection
        detail_path = "model_vs_manual_matched_sites.csv"
        complete_df.to_csv(detail_path, index=False)
        print(f"Saved: {detail_path}")

        # --- Scatter plot: biophony, anthrophony, silence (not geophony) ---
        # Adjust manual column name for silence to match your manual file.
        plot_columns = [
            ("pct_biophony", COLUMN_MAP["pct_biophony"], "Biophony"),
            ("pct_anthropophony", COLUMN_MAP["pct_anthropophony"], "Anthropophony"),
            ("pct_silence", COLUMN_MAP["pct_silence"], "Silence"),
        ]
        plot_path = "model_vs_manual_scatter.png"
        plot_model_vs_manual(complete_df, plot_columns, output_path=plot_path)