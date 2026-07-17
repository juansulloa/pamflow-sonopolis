#!/usr/bin/env python3
"""
Walks a root folder containing one subfolder per species. Inside each species
folder, .wav segments may sit directly in the species folder (not yet
classified -> NA), or inside a "Positive" / "Negative" subfolder.

Produces a DataFrame with columns:
    segmentName, scientificName, classificationProbability, validationResult

classificationProbability is parsed from the leading numeric field of the
filename, e.g. "0.464_PAH20_20260427_140000_54.0_57.0.wav" -> 0.464.

Usage:
    python get_validation_status.py /path/to/root_folder [-o output.csv]
"""

import argparse
from pathlib import Path
import pandas as pd


def parse_classification_probability(filename: str):
    """Extract the leading float from a segment filename, e.g.
    '0.464_PAH20_20260427_140000_54.0_57.0.wav' -> 0.464.
    Returns None if the leading field isn't a valid number.
    """
    leading = filename.split("_", 1)[0]
    try:
        return float(leading)
    except ValueError:
        return None


def get_validation_status(root_folder: str) -> pd.DataFrame:
    root = Path(root_folder)
    if not root.is_dir():
        raise NotADirectoryError(f"{root_folder} is not a valid directory")

    records = []

    # Each direct subfolder of root is treated as a species
    for species_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        scientific_name = species_dir.name.replace("_", " ")

        for wav_path in species_dir.rglob("*.wav"):
            # Determine validation based on immediate parent folder name
            parent_name = wav_path.parent.name

            if parent_name.lower() == "positive":
                validation = "Positive"
            elif parent_name.lower() == "negative":
                validation = "Negative"
            elif wav_path.parent == species_dir:
                validation = "NA"
            else:
                # segment inside an unexpected nested folder -> treat as NA
                validation = "NA"

            records.append({
                "segmentName": wav_path.name,
                "scientificName": scientific_name,
                "classificationProbability": parse_classification_probability(wav_path.name),
                "validationResult": validation,
            })

    df = pd.DataFrame(
        records,
        columns=["segmentName", "scientificName", "classificationProbability", "validationResult"],
    )
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Build a DataFrame of segment validation status from a species folder structure."
    )
    parser.add_argument("root_folder", help="Path to the root folder containing species subfolders")
    parser.add_argument("-o", "--output", help="Optional path to save the result as CSV")
    args = parser.parse_args()

    df = get_validation_status(args.root_folder)

    # Print result summary as pivot table: species, Positive, Negative, NA counts
    pivot = df.pivot_table(
        index="scientificName",
        columns="validationResult",
        aggfunc="size",
        fill_value=0,
    )
    only_negative = (pivot["Negative"] > 0) & (pivot["Positive"] == 0)
    only_positive = (pivot["Positive"] > 0) & (pivot["Negative"] == 0)
    positive_and_negative = (pivot["Positive"] > 0) & (pivot["Negative"] > 0)
    only_na = (pivot["NA"] > 0) & (pivot["Positive"] == 0) & (pivot["Negative"] == 0)

    print("\nValidation status summary:")
    print("Number of species:", len(pivot))    
    print("Number of species with only Negative segments:", only_negative.sum())
    print("Number of species with only Positive segments:", only_positive.sum())
    print("Number of species with both Positive and Negative segments:", positive_and_negative.sum())
    print("Number of species with only NA segments:", only_na.sum())
    print("Species with only NA segments:", pivot[only_na].index.tolist())
    print(f"\nTotal segments: {len(df)}")
    print(f"\nTotal segments (excluding NA): {len(df[df['validationResult'] != 'NA'])}")
    print(df["validationResult"].value_counts())

    if args.output:
        df.to_csv(args.output, index=False)
        print(f"\nSaved to {args.output}")
        print("Note: 'NA' here is a real validationResult label, not a missing value. "
              "When reading this CSV back with pandas, pass "
              "keep_default_na=False, na_values=[] or 'NA' will silently "
              "become NaN.")


if __name__ == "__main__":
    main()