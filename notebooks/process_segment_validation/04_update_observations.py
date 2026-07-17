"""
Filter observations.csv down to detections that meet each species'
appliedCutoff confidence threshold, as computed by
02_find_model_threshold.py's species_confidence_thresholds.csv.

An observation is kept only if its classificationProbability is >= the
appliedCutoff for its scientificName. Species with appliedCutoff = inf
(cutoffSource = 'exclude') are dropped entirely, since no confidence score
can meet an infinite cutoff. Species with no row at all in the thresholds
table (never went through validation) are also dropped by default, since
there's no evidence-backed cutoff to apply; --keep-unthresholded keeps them
instead.

Usage:
    python 04_update_observations.py [-i observations.csv]
        [-t ../../data/output/validation/species_confidence_thresholds.csv]
        [-o ../../data/output/validation/observations.csv]
        [--keep-unthresholded]

    -i/--observations defaults to observations.csv in the current directory.
    -t/--thresholds defaults to species_confidence_thresholds.csv produced by
        02_find_model_threshold.py.
    -o/--output is where the filtered observations are saved
        (default: ../../data/output/validation/observations.csv).
    --keep-unthresholded keeps observations for species missing from the
        thresholds table instead of dropping them.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def filter_observations(observations, thresholds, keep_unthresholded=False):
    """Keep only observations whose classificationProbability is >= the
    appliedCutoff for their species. Species absent from `thresholds` are
    dropped unless keep_unthresholded is set.
    """
    cutoffs = thresholds.set_index('scientificName')['appliedCutoff']
    applied_cutoff = observations['scientificName'].map(cutoffs)

    unthresholded = applied_cutoff.isna()
    if unthresholded.any():
        species = sorted(observations.loc[unthresholded, 'scientificName'].unique())
        action = "keeping" if keep_unthresholded else "dropping"
        print(f"{len(species)} species have no entry in the thresholds table ({action}): {species}")
        if keep_unthresholded:
            applied_cutoff = applied_cutoff.fillna(-np.inf)

    keep = observations['classificationProbability'] >= applied_cutoff
    return observations[keep]


def main():
    parser = argparse.ArgumentParser(
        description="Filter observations.csv to detections meeting each species' appliedCutoff confidence threshold."
    )
    parser.add_argument('-i', '--observations', default='../../data/output/species_detection/observations.csv',
                         help='Path to observations.csv (default: ../../data/output/species_detection/observations.csv)')
    parser.add_argument('-t', '--thresholds', default='../../data/output/validation/species_confidence_thresholds.csv',
                         help='Path to species_confidence_thresholds.csv (default: ../../data/output/validation/species_confidence_thresholds.csv)')
    parser.add_argument('-o', '--output', default='../../data/output/validation/observations_thresholded.csv',
                         help='Path to save the filtered observations CSV (default: ../../data/output/validation/observations_thresholded.csv)')
    parser.add_argument('--keep-unthresholded', action='store_true',
                         help='Keep observations for species missing from the thresholds table (default: drop them)')
    args = parser.parse_args()

    observations = pd.read_csv(args.observations)
    thresholds = pd.read_csv(args.thresholds)

    filtered = filter_observations(observations, thresholds, args.keep_unthresholded)

    print(f"\nObservations: {len(observations)} -> {len(filtered)} ({len(observations) - len(filtered)} removed)")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    filtered.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
