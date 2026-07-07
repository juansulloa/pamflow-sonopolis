"""
Batch soundscape comparison across all deployments.

Scans a directory of {deploymentID}_observations.csv files (as produced by
01_analyze_tagging.py), runs the biophony/anthropophony/geophony profile on
each, and builds one comparison table:

    deploymentID | n_windows | %biophony | %anthropophony | %geophony | top_biophony | top_anthropophony | top_geophony

Run directly (adjust `save_path` below), or import `build_comparison_table`
elsewhere.
"""

import os
import re
from pathlib import Path

import pandas as pd
from label_category_crosswalk import LABEL_CATEGORY

CATEGORIES = ["biophony", "anthropophony", "geophony", "silence"]


def categorize(label):
    return LABEL_CATEGORY.get(label, "unclassified")


def soundscape_profile(csv_path, top_n=5, include_unclassified_in_total=True):
    df = pd.read_csv(csv_path)
    n_windows = df.groupby(["mediaID", "eventStart"]).ngroups

    df["category"] = df["label"].apply(categorize)
    
    # remove observations with confidence below 0.1
    df = df[df["classificationProbability"] >= 0.1]

    label_weight = (
        df.groupby(["category", "label"])["classificationProbability"]
        .sum().div(n_windows).reset_index(name="weighted_proportion")
    )

    # Drop explicitly excluded labels
    label_weight = label_weight[label_weight["category"].notna()]

    if not include_unclassified_in_total:
        label_weight = label_weight[label_weight["category"] != "unclassified"]

    cat_totals = label_weight.groupby("category")["weighted_proportion"].sum()
    cat_totals_norm = (cat_totals / cat_totals.sum()).sort_values(ascending=False)

    top_labels = (
        label_weight.sort_values(["category", "weighted_proportion"], ascending=[True, False])
        .groupby("category").head(top_n)
    )

    return cat_totals_norm, top_labels, n_windows


def extract_deployment_id(filename, suffix="_observations.csv"):
    """Recover deploymentID from a file named '{deploymentID}_observations.csv'."""
    name = Path(filename).name
    if name.endswith(suffix):
        return name[: -len(suffix)]
    return Path(filename).stem


def build_comparison_table(save_path, top_n=3, include_unclassified_in_total=True,
                            file_pattern=r".*_observations\.csv$"):
    save_path = Path(save_path)
    files = sorted([f for f in save_path.glob("*.csv") if re.match(file_pattern, f.name)])

    if not files:
        print(f"No observation CSVs found in {save_path}")
        return pd.DataFrame()

    rows = []
    for f in files:
        deployment_id = extract_deployment_id(f.name)
        try:
            cat_props, top_labels, n_windows = soundscape_profile(
                f, top_n=top_n, include_unclassified_in_total=include_unclassified_in_total
            )
        except Exception as e:
            print(f"  [skipped] {f.name}: {e}")
            continue

        # Ensure all three main categories are represented (0 if absent)
        row = {
            "deploymentID": deployment_id,
            "n_windows": n_windows,
        }
        for cat in CATEGORIES:
            row[f"pct_{cat}"] = round(cat_props.get(cat, 0.0) * 100, 2)

        # Add unclassified % if present and requested
        if include_unclassified_in_total and "unclassified" in cat_props.index:
            row["pct_unclassified"] = round(cat_props.get("unclassified", 0.0) * 100, 2)
        else:
            row["pct_unclassified"] = 0.0

        # Top-N dominant labels per category, as a compact string
        for cat in CATEGORIES:
            subset = top_labels[top_labels["category"] == cat]
            labels_str = ", ".join(subset["label"].tolist())
            row[f"top_{cat}_labels"] = labels_str

        rows.append(row)
        print(f"  Processed: {deployment_id} "
              f"(bio={row['pct_biophony']}%, anthro={row['pct_anthropophony']}%, "
              f"geo={row['pct_geophony']}%)")

    comparison_df = pd.DataFrame(rows)
    comparison_df = comparison_df.sort_values("deploymentID").reset_index(drop=True)
    return comparison_df


if __name__ == "__main__":
    # --- Variables ---
    save_path = "../../data/output/panns/"     # folder with {deploymentID}_observations.csv files
    output_path = "../../data/output/panns/soundscape_comparison_table.csv"
    top_n = 3
    include_unclassified_in_total = True  # keep visible rather than silently redistributed

    print(f"Scanning: {save_path}\n")
    comparison_df = build_comparison_table(
        save_path, top_n=top_n, include_unclassified_in_total=include_unclassified_in_total
    )

    if comparison_df.empty:
        print("No results generated.")
    else:
        comparison_df.to_csv(output_path, index=False)
        print(f"\nSaved comparison table: {output_path}")
        print(f"({len(comparison_df)} deployments)\n")
        print(comparison_df[["deploymentID", "n_windows", "pct_biophony",
                              "pct_anthropophony", "pct_geophony"]].to_string(index=False))
