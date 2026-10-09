#!/usr/bin/env python3
"""
Builds a single segment validation table from three sources under a root
segments folder:

    01_validated_segments/<Genus_species>/[Positive|Negative]/<segment>.wav
        .wav files directly under the species folder are not yet validated (NA).
    02_reassigned_segments/<Genus_species>/<Genus_species>/<segment>.wav
        First level: species originally predicted by the model. Second level:
        species assigned by the annotator. Never creates rows: each file is
        matched to its row from 01 by (original species, segmentName) and
        fills scientificNameValidated.
    03_manual_segments/<Genus_species>/**/<segment>.wav
        Manually collected segments. Always Positive; any Positive/Negative
        subfolder is ignored.

Anything that is not a .wav file, and any other folder in the root, is ignored.
The script is read-only on the segments folders.

Produces a table with columns:
    segmentName, scientificNameModel, classificationProbability,
    validationResult, scientificNameValidated
Missing values are written as NA.

classificationProbability is parsed from the leading numeric field of the
filename, e.g. "0.464_PAH20_20260427_140000_54.0_57.0.wav" -> 0.464.

Usage:
    python 01_get_validation_status.py /path/to/segments_root [-o output.csv]

    If -o/--output is omitted, results are saved to
    data/output_pa/validation/segment_validation_results.csv.
"""

import argparse
import logging
from collections import Counter
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

VALIDATED_DIR = "01_validated_segments"
REASSIGNED_DIR = "02_reassigned_segments"
MANUAL_DIR = "03_manual_segments"

COLUMNS = [
    "segmentName",
    "scientificNameModel",
    "classificationProbability",
    "validationResult",
    "scientificNameValidated",
]


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


def normalize_species(name: str) -> str:
    """Convert a folder name such as 'Genus_species' to 'Genus species'."""
    return name.replace("_", " ").strip()


def read_validated(folder: Path) -> pd.DataFrame:
    """Read 01_validated_segments: one row per .wav file."""
    records = []

    # Each direct subfolder is treated as a species
    for species_dir in sorted(p for p in folder.iterdir() if p.is_dir()):
        scientific_name = normalize_species(species_dir.name)

        for wav_path in sorted(species_dir.rglob("*.wav")):
            parent_name = wav_path.parent.name.lower()
            if parent_name == "positive":
                validation = "Positive"
            elif parent_name == "negative":
                validation = "Negative"
            else:
                # directly under the species folder, or in an unexpected
                # nested folder -> not validated
                validation = "NA"

            records.append({
                "segmentName": wav_path.name,
                "scientificNameModel": scientific_name,
                "classificationProbability": parse_classification_probability(wav_path.name),
                "validationResult": validation,
                "scientificNameValidated": scientific_name if validation == "Positive" else None,
            })

    return pd.DataFrame(records, columns=COLUMNS)


def apply_reassigned(df: pd.DataFrame, folder: Path):
    """Fill scientificNameValidated in df (in place) from 02_reassigned_segments.

    Returns the number of files read and a Counter of reassignments keyed by
    (original species, validated species).
    """
    index = {
        (row.scientificNameModel, row.segmentName): i
        for i, row in zip(df.index, df.itertuples())
    }
    assigned_to = {}
    pair_counts = Counter()
    n_files = 0

    for original_dir in sorted(p for p in folder.iterdir() if p.is_dir()):
        original = normalize_species(original_dir.name)

        for assigned_dir in sorted(p for p in original_dir.iterdir() if p.is_dir()):
            assigned = normalize_species(assigned_dir.name)

            for wav_path in sorted(assigned_dir.rglob("*.wav")):
                n_files += 1
                key = (original, wav_path.name)

                if key not in index:
                    logger.warning(
                        "Reassigned segment without counterpart in %s, skipped: %s/%s",
                        VALIDATED_DIR, original, wav_path.name,
                    )
                    continue

                if key in assigned_to:
                    logger.warning(
                        "Segment found in more than one reassigned destination "
                        "(%s and %s), keeping the first: %s/%s",
                        assigned_to[key], assigned, original, wav_path.name,
                    )
                    continue

                row = index[key]
                result = df.at[row, "validationResult"]
                if result == "NA":
                    logger.warning(
                        "Reassigned but not validated: %s -> %s: %s",
                        original, assigned, wav_path.name,
                    )
                elif result == "Positive":
                    logger.warning(
                        "Reassigned segment is Positive in %s (possible co-occurrence "
                        "or error): %s -> %s: %s",
                        VALIDATED_DIR, original, assigned, wav_path.name,
                    )

                assigned_to[key] = assigned
                df.at[row, "scientificNameValidated"] = assigned
                pair_counts[(original, assigned)] += 1

    return n_files, pair_counts


def read_manual(folder: Path, existing_names: set) -> pd.DataFrame:
    """Read 03_manual_segments: one Positive row per .wav not already present."""
    records = []

    for species_dir in sorted(p for p in folder.iterdir() if p.is_dir()):
        scientific_name = normalize_species(species_dir.name)

        for wav_path in sorted(species_dir.rglob("*.wav")):
            if wav_path.name in existing_names:
                logger.warning(
                    "Manual segment already present in %s or %s, skipped: %s",
                    VALIDATED_DIR, REASSIGNED_DIR, wav_path.name,
                )
                continue

            records.append({
                "segmentName": wav_path.name,
                "scientificNameModel": None,
                "classificationProbability": None,
                "validationResult": "Positive",
                "scientificNameValidated": scientific_name,
            })

    return pd.DataFrame(records, columns=COLUMNS)


def get_validation_status(root_folder: str):
    """Build the validation table. Returns (table, summary dict)."""
    root = Path(root_folder)
    if not root.is_dir():
        raise NotADirectoryError(f"{root_folder} is not a valid directory")
    for name in (VALIDATED_DIR, REASSIGNED_DIR, MANUAL_DIR):
        if not (root / name).is_dir():
            raise NotADirectoryError(f"Expected subfolder not found: {root / name}")

    validated = read_validated(root / VALIDATED_DIR)
    n_reassigned_files, pair_counts = apply_reassigned(validated, root / REASSIGNED_DIR)

    # Names present in 01 or 02. 02 files either match a 01 row (already
    # included) or were skipped for lacking a counterpart.
    existing_names = set(validated["segmentName"])
    for wav_path in (root / REASSIGNED_DIR).rglob("*.wav"):
        existing_names.add(wav_path.name)
    manual = read_manual(root / MANUAL_DIR, existing_names)

    df = pd.concat([validated, manual], ignore_index=True)[COLUMNS]

    duplicated = df[df.duplicated(subset="segmentName", keep=False)]
    if not duplicated.empty:
        raise ValueError(
            f"Duplicate segmentName keys in the final table: "
            f"{sorted(duplicated['segmentName'].unique())[:10]} "
            f"({len(duplicated)} rows affected)"
        )

    summary = {
        "rows_validated": len(validated),
        "rows_reassigned_files": n_reassigned_files,
        "rows_manual": len(manual),
        "reassigned_pairs": pair_counts,
    }
    return df, summary


def print_summary(df: pd.DataFrame, summary: dict):
    print("\nRows per source:")
    print(f"  {VALIDATED_DIR}: {summary['rows_validated']} rows")
    print(f"  {REASSIGNED_DIR}: {summary['rows_reassigned_files']} files (no new rows)")
    print(f"  {MANUAL_DIR}: {summary['rows_manual']} rows")
    print(f"\nTotal segments: {len(df)}")
    print(f"Total segments (excluding NA): {len(df[df['validationResult'] != 'NA'])}")

    print("\nvalidationResult counts:")
    print(df["validationResult"].value_counts().to_string())

    print("\nReassigned segments (original species -> validated species):")
    if summary["reassigned_pairs"]:
        for (original, assigned), n in sorted(summary["reassigned_pairs"].items()):
            print(f"  {original} -> {assigned}: {n}")
    else:
        print("  none")
    by_validated = Counter()
    for (_, assigned), n in summary["reassigned_pairs"].items():
        by_validated[assigned] += n
    for assigned, n in sorted(by_validated.items()):
        print(f"  total -> {assigned}: {n}")

    # Per-species summary based on the model species (excludes manual rows)
    model_rows = df[df["scientificNameModel"].notna()]
    pivot = model_rows.pivot_table(
        index="scientificNameModel",
        columns="validationResult",
        aggfunc="size",
        fill_value=0,
    ).reindex(columns=["Positive", "Negative", "NA"], fill_value=0)
    only_negative = (pivot["Negative"] > 0) & (pivot["Positive"] == 0)
    only_positive = (pivot["Positive"] > 0) & (pivot["Negative"] == 0)
    positive_and_negative = (pivot["Positive"] > 0) & (pivot["Negative"] > 0)
    only_na = (pivot["NA"] > 0) & (pivot["Positive"] == 0) & (pivot["Negative"] == 0)

    print("\nValidation status summary (model species):")
    print("Number of species:", len(pivot))
    print("Number of species with only Negative segments:", only_negative.sum())
    print("Number of species with only Positive segments:", only_positive.sum())
    print("Number of species with both Positive and Negative segments:", positive_and_negative.sum())
    print("Number of species with only NA segments:", only_na.sum())
    print("Species with only NA segments:", pivot[only_na].index.tolist())


class _WarningCounter(logging.Handler):
    """Count warnings by message prefix to report them in the final summary."""

    def __init__(self):
        super().__init__(level=logging.WARNING)
        self.counts = Counter()

    def emit(self, record):
        message = record.getMessage()
        self.counts[message.split(" (")[0].split(":")[0]] += 1


def main():
    parser = argparse.ArgumentParser(
        description="Build a single segment validation table from the validated, "
                    "reassigned and manual segment folders."
    )
    parser.add_argument(
        "root_folder",
        help=f"Path to the segments root folder containing {VALIDATED_DIR}, "
             f"{REASSIGNED_DIR} and {MANUAL_DIR}",
    )
    parser.add_argument(
        "-o", "--output",
        default="../../data/output_pa/validation/segment_validation_results.csv",
        help="Path to save the result as CSV (default: segment_validation_results.csv)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="WARNING: %(message)s")
    counter = _WarningCounter()
    logger.addHandler(counter)

    df, summary = get_validation_status(args.root_folder)
    print_summary(df, summary)

    print("\nWarnings:")
    if counter.counts:
        for message, n in counter.counts.items():
            print(f"  {message}: {n}")
    else:
        print("  none")

    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output, index=False, na_rep="NA")
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
