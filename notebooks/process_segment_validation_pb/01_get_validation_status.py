#!/usr/bin/env python3
"""
Builds a single segment validation table from a validation results CSV
(one row per validated segment), e.g. validaciones_anura_pb.csv.

Expected input columns:
    scientificNameProposed  species predicted by the model
    validationCategory      positive | negative | uncertain | reassigned
    scientificNameValidated species assigned by the annotator (reassigned only)
    fileName                path of the segment, e.g.
                            "Genus_species\\Genus_species\\0.103_PBH27_20260814_183000_30.0_33.0.wav"
                            (Windows or POSIX separators; only the basename is used)
    classifiedBy, favorite, validationComment
                            carried over unchanged to the output

Segment names follow "{classificationProbability}_{mediaID}_{eventStart}_{eventEnd}.wav",
e.g. "0.122_PAO202_20260412_013000_30.0_33.0.wav".

Mapping rules:
    positive   -> validationResult "Positive", scientificNameValidated = model species
    negative   -> validationResult "Negative", scientificNameValidated = NA
    reassigned -> validationResult "Reassigned" (the model prediction was wrong),
                  scientificNameValidated = species assigned by the annotator
    uncertain  -> validationResult "Uncertain" (reviewed, undecided),
                  scientificNameValidated = NA
    anything else / empty -> "NA" (not validated)

Produces a table with columns:
    segmentName, scientificNameModel, classificationProbability,
    validationResult, scientificNameValidated, classifiedBy, favorite,
    validationComment
Missing values are written as NA.

classificationProbability is parsed from the leading numeric field of the
segment name, e.g. "0.464_PAH20_20260427_140000_54.0_57.0.wav" -> 0.464.

Usage:
    python 01_get_validation_status.py /path/to/validation_results.csv -o output.csv

    -o/--output is required; there is no default path.
"""

import argparse
import logging
from collections import Counter
from pathlib import Path, PureWindowsPath

import pandas as pd

logger = logging.getLogger(__name__)

COLUMNS = [
    "segmentName",
    "scientificNameModel",
    "classificationProbability",
    "validationResult",
    "scientificNameValidated",
    "classifiedBy",
    "favorite",
    "validationComment",
]

REQUIRED_INPUT_COLUMNS = [
    "scientificNameProposed",
    "validationCategory",
    "scientificNameValidated",
    "fileName",
    "classifiedBy",
    "favorite",
    "validationComment",
]

# validationCategory (lowercase) -> validationResult
CATEGORY_TO_RESULT = {
    "positive": "Positive",
    "negative": "Negative",
    "reassigned": "Reassigned",
    "uncertain": "Uncertain",
}


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


def normalize_species(name) -> str:
    """Convert 'Genus_species' to 'Genus species'; return None for blanks."""
    if pd.isna(name):
        return None
    name = str(name).replace("_", " ").strip()
    return name or None


def segment_name_from_path(file_name: str) -> str:
    """Basename of a path that may use Windows or POSIX separators."""
    return PureWindowsPath(str(file_name).strip()).name


def read_validation_results(csv_path: Path) -> pd.DataFrame:
    """Read the validation CSV and return one row per segment."""
    raw = pd.read_csv(csv_path, dtype=str, keep_default_na=False, na_values=[""])

    missing = [c for c in REQUIRED_INPUT_COLUMNS if c not in raw.columns]
    if missing:
        raise ValueError(f"Missing required columns in {csv_path}: {missing}")

    records = []
    for row in raw.itertuples(index=False):
        if pd.isna(row.fileName):
            logger.warning("Row without fileName, skipped (species: %s)", row.scientificNameProposed)
            continue

        segment_name = segment_name_from_path(row.fileName)
        model_species = normalize_species(row.scientificNameProposed)
        category = str(row.validationCategory).strip().lower() if pd.notna(row.validationCategory) else ""

        if category not in CATEGORY_TO_RESULT:
            logger.warning("Unknown validationCategory %r treated as NA: %s", category, segment_name)
        validation = CATEGORY_TO_RESULT.get(category, "NA")

        if category == "positive":
            validated_species = model_species
        elif category == "reassigned":
            validated_species = normalize_species(row.scientificNameValidated)
            if validated_species is None:
                logger.warning("Reassigned segment without scientificNameValidated: %s", segment_name)
        else:
            validated_species = None

        probability = parse_classification_probability(segment_name)
        if probability is None:
            logger.warning("Could not parse classificationProbability: %s", segment_name)

        records.append({
            "segmentName": segment_name,
            "scientificNameModel": model_species,
            "classificationProbability": probability,
            "validationResult": validation,
            "scientificNameValidated": validated_species,
            "classifiedBy": row.classifiedBy,
            "favorite": row.favorite,
            "validationComment": row.validationComment,
        })

    return pd.DataFrame(records, columns=COLUMNS)


def get_validation_status(csv_path: str):
    """Build the validation table. Returns (table, summary dict)."""
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"{csv_path} is not a valid file")

    df = read_validation_results(path)

    duplicated = df[df.duplicated(subset="segmentName", keep=False)]
    if not duplicated.empty:
        raise ValueError(
            f"Duplicate segmentName keys in the final table: "
            f"{sorted(duplicated['segmentName'].unique())[:10]} "
            f"({len(duplicated)} rows affected)"
        )

    reassigned = df[df["validationResult"] == "Reassigned"]
    pair_counts = Counter(
        zip(reassigned["scientificNameModel"], reassigned["scientificNameValidated"])
    )

    summary = {"rows": len(df), "reassigned_pairs": pair_counts}
    return df, summary


def print_summary(df: pd.DataFrame, summary: dict):
    print(f"\nTotal segments: {len(df)}")
    print(f"Total segments (excluding NA): {len(df[df['validationResult'] != 'NA'])}")

    print("\nvalidationResult counts:")
    print(df["validationResult"].value_counts().to_string())

    print("\nReassigned segments (model species -> validated species):")
    if summary["reassigned_pairs"]:
        for (original, assigned), n in sorted(summary["reassigned_pairs"].items()):
            print(f"  {original} -> {assigned}: {n}")
    else:
        print("  none")

    pivot = df.pivot_table(
        index="scientificNameModel",
        columns="validationResult",
        aggfunc="size",
        fill_value=0,
    ).reindex(columns=["Positive", "Negative", "Reassigned", "Uncertain", "NA"], fill_value=0)
    only_negative = (pivot["Negative"] > 0) & (pivot["Positive"] == 0)
    only_positive = (pivot["Positive"] > 0) & (pivot["Negative"] == 0)
    positive_and_negative = (pivot["Positive"] > 0) & (pivot["Negative"] > 0)
    only_na = (pivot["NA"] > 0) & (pivot[["Positive", "Negative", "Reassigned", "Uncertain"]].sum(axis=1) == 0)

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
        description="Build a single segment validation table from a validation "
                    "results CSV."
    )
    parser.add_argument(
        "input_csv",
        help="Path to the validation results CSV (e.g. validaciones_anura_pb.csv)",
    )
    parser.add_argument(
        "-o", "--output",
        required=True,
        help="Path to save the result as CSV",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="WARNING: %(message)s")
    counter = _WarningCounter()
    logger.addHandler(counter)

    df, summary = get_validation_status(args.input_csv)
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
