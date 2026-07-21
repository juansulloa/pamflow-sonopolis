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

Observations manually confirmed as Positive in segment_validation_results.csv
are included in the output regardless of whether they meet the cutoff, since
a human confirmation is stronger evidence than the statistical threshold.

Usage:
    python 04_update_observations.py [-i observations.csv]
        [-t ../../data/output/validation/species_confidence_thresholds.csv]
        [-v ../../data/output/validation/segment_validation_results.csv]
        [-o ../../data/output/validation/observations.csv]
        [--keep-unthresholded]

    -i/--observations defaults to observations.csv in the current directory.
    -t/--thresholds defaults to species_confidence_thresholds.csv produced by
        02_find_model_threshold.py.
    -v/--validations defaults to segment_validation_results.csv, used to
        recover manually confirmed positives regardless of cutoff.
    -o/--output is where the filtered observations are saved
        (default: ../../data/output/validation/observations.csv).
    --keep-unthresholded keeps observations for species missing from the
        thresholds table instead of dropping them.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

CLASSIFIEDBYNAME = "Eliana Barona"
CLASSIFICATIONDATE = "2026-07-13T00:00:00"

def get_manually_validated_observations(observations, segment_validation_results):
    """Return the subset of `observations` confirmed by manual validation.

    Segments in `segment_validation_results` with validationResult == 'Positive'
    are matched back to their source row in `observations`. Each segmentName is
    structured as "{classificationProbability}_{mediaID}_{eventStart}_{eventEnd}.wav"
    (e.g. "0.122_PAO202_20260412_013000_30.0_33.0.wav" decodes to
    classificationProbability=0.122, mediaID=PAO202_20260412_013000.WAV,
    eventStart=30.0, eventEnd=33.0). Note mediaID in `observations` carries an
    uppercase ".WAV" extension while segmentName uses a lowercase ".wav" suffix.

    A match requires equal mediaID, eventStart, eventEnd, scientificName, and
    classificationProbability between the decoded segment and an observations row.

    Returns the matching rows of `observations` (one row per matched positive
    segment); segments that don't match any observation are dropped silently.
    """
    positive = segment_validation_results[segment_validation_results['validationResult'] == 'Positive']

    stem = positive['segmentName'].str.removesuffix('.wav')
    parts = stem.str.split('_')

    decoded = pd.DataFrame({
        'classificationProbability': parts.str[0].astype(float),
        'mediaID': parts.str[1:-2].str.join('_') + '.WAV',
        'eventStart': parts.str[-2].astype(float),
        'eventEnd': parts.str[-1].astype(float),
        'scientificName': positive['scientificName'].values,
    })

    merge_cols = ['mediaID', 'eventStart', 'eventEnd', 'scientificName', 'classificationProbability']
    matched = observations.merge(decoded[merge_cols], on=merge_cols, how='inner')
    return matched

def stamp_manual_validation_metadata(manually_validated):
    """Overwrite classification fields on manually-confirmed rows to reflect
    human review instead of the original automated classification.
    """
    manually_validated = manually_validated.copy()
    manually_validated['classificationProbability'] = 1
    manually_validated['classificationMethod'] = 'human'
    manually_validated['classifiedBy'] = CLASSIFIEDBYNAME
    manually_validated['classificationTimestamp'] = CLASSIFICATIONDATE
    return manually_validated

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
    parser.add_argument('-v', '--validations', default='../../data/output/validation/segment_validation_results.csv',
                         help='Path to segment_validation_results.csv (default: ../../data/output/validation/segment_validation_results.csv)')
    parser.add_argument('-o', '--output', default='../../data/output/validation/observations_thresholded.csv',
                         help='Path to save the filtered observations CSV (default: ../../data/output/validation/observations_thresholded.csv)')
    parser.add_argument('--keep-unthresholded', action='store_true',
                         help='Keep observations for species missing from the thresholds table (default: drop them)')
    args = parser.parse_args()

    observations = pd.read_csv(args.observations)
    thresholds = pd.read_csv(args.thresholds)
    validations = pd.read_csv(args.validations)

    filtered = filter_observations(observations, thresholds, args.keep_unthresholded)
    manually_validated = get_manually_validated_observations(observations, validations)
    manually_validated = stamp_manual_validation_metadata(manually_validated)

    combined = (
        pd.concat([manually_validated, filtered])
        .drop_duplicates(subset='observationID')
        .sort_values('observationID')
    )

    added_by_manual = (~manually_validated['observationID'].isin(filtered['observationID'])).sum()
    print(
        f"\nValidated observations: {len(manually_validated)}"
        f"\nObservations: {len(observations)} -> {len(filtered)} passed cutoff "
        f"(+{added_by_manual} added via manual validation) -> {len(combined)} total"
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
