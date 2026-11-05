"""
Build the final observations table from the model observations and the manual
validation results (segment_validation_results.csv from
01_get_validation_status.py, thresholds from 02_find_model_threshold.py).

The output contains, in order of precedence (an observation is used once):
    1. Reassigned: observations whose segment was manually reassigned to another
       species. scientificName is replaced by the validated species and a
       comment records that it was a model false positive reassigned manually.
    2. Manually confirmed: observations whose segment was validated as Positive
       (classification fields stamped as human review).
    3. Thresholded: the remaining machine observations whose
       classificationProbability is >= the appliedCutoff of their species.
       Species with appliedCutoff = inf (cutoffSource = 'exclude') are dropped,
       as are species missing from the thresholds table unless
       --keep-unthresholded is set. Observations manually validated as Negative
       (and not reassigned) are dropped even if they meet the cutoff.
    4. Inferred reassignments (only with --apply-confusions): remaining
       observations of a species consistently confused with another species
       (species_confusions.csv) at or above appliedConfusionCutoff are relabeled
       as the confused species and tagged 'inferred_reassignment'.
    5. Manual annotations: segments without a model species are added as new
       observations with a new observationID.

Segment names are structured as
"{classificationProbability}_{mediaID}_{eventStart}_{eventEnd}.wav", e.g.
"0.122_PAO202_20260412_013000_30.0_33.0.wav". mediaID in the observations
carries an uppercase ".WAV" extension.

Usage:
    python 04_update_observations.py [-i observations.csv]
        [-t ../../data/output_pa/validation/species_confidence_thresholds.csv]
        [-v ../../data/output_pa/validation/segment_validation_results.csv]
        [-c ../../data/output_pa/validation/species_confusions.csv]
        [-o ../../data/output_pa/validation/observations_thresholded.csv]
        [--keep-unthresholded] [--apply-confusions]
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

CLASSIFIEDBYNAME = "Eliana Barona"
CLASSIFICATIONDATE = "2026-07-13T00:00:00"

MATCH_COLUMNS = ['mediaID', 'eventStart', 'eventEnd', 'scientificName', 'classificationProbability']
INFERRED_TAG = 'inferred_reassignment'


def decode_segments(validations):
    """Decode segmentName into classificationProbability, mediaID, eventStart
    and eventEnd, keeping the validation columns.
    """
    stem = validations['segmentName'].str.removesuffix('.wav')
    parts = stem.str.split('_')
    return pd.DataFrame({
        'segmentName': validations['segmentName'].values,
        'classificationProbability': parts.str[0].astype(float).values,
        'mediaID': (parts.str[1:-2].str.join('_') + '.WAV').values,
        'eventStart': parts.str[-2].astype(float).values,
        'eventEnd': parts.str[-1].astype(float).values,
        'scientificNameModel': validations['scientificNameModel'].values,
        'validationResult': validations['validationResult'].values,
        'scientificNameValidated': validations['scientificNameValidated'].values,
    })


def match_observations(observations, decoded, label):
    """Return the rows of `observations` matching the decoded model segments,
    with scientificNameValidated added. Unmatched segments are reported.
    """
    right = decoded.rename(columns={'scientificNameModel': 'scientificName'})[
        MATCH_COLUMNS + ['scientificNameValidated']
    ]
    matched = observations.merge(right, on=MATCH_COLUMNS, how='inner')

    # Some segments were exported with the observation window padded with
    # context (e.g. observation 24.0-27.0 -> segment 21.5-29.5). Match these on
    # media, species and score, with the observation window inside the segment.
    unmatched = right.merge(
        observations[MATCH_COLUMNS].drop_duplicates(), on=MATCH_COLUMNS, how='left', indicator=True
    )
    unmatched = unmatched[unmatched['_merge'] == 'left_only'].drop(columns='_merge')
    n_unmatched = len(unmatched)
    if len(unmatched) > 0:
        candidates = unmatched.reset_index(drop=True).rename_axis('segmentIndex').reset_index().merge(
            observations,
            on=['mediaID', 'scientificName', 'classificationProbability'],
            suffixes=('_segment', ''),
        )
        candidates = candidates[
            (candidates['eventStart_segment'] <= candidates['eventStart'])
            & (candidates['eventEnd'] <= candidates['eventEnd_segment'])
        ].copy()
        candidates['centerDistance'] = (
            (candidates['eventStart'] + candidates['eventEnd'])
            - (candidates['eventStart_segment'] + candidates['eventEnd_segment'])
        ).abs()
        n_ambiguous = int((candidates.groupby('segmentIndex').size() > 1).sum())
        if n_ambiguous > 0:
            print(f"WARNING: {n_ambiguous} {label} segments matched several observations, "
                  "using the one closest to the segment center")
        padded = candidates.sort_values('centerDistance').drop_duplicates(subset='segmentIndex')
        matched = pd.concat([matched, padded[matched.columns]])
        n_unmatched -= len(padded)

    # Several segments can point to the same observation (e.g. a padded and an
    # exact clip of one detection), so the output is deduplicated.
    matched = matched.drop_duplicates(subset='observationID')
    if n_unmatched > 0:
        print(f"WARNING: {n_unmatched} {label} segments did not match any observation")
    return matched


def stamp_human_review(rows):
    """Overwrite classification fields to reflect human review."""
    rows = rows.copy()
    rows['classificationMethod'] = 'human'
    rows['classifiedBy'] = CLASSIFIEDBYNAME
    rows['classificationTimestamp'] = CLASSIFICATIONDATE
    return rows


def build_reassigned(observations, decoded):
    """Observations manually reassigned to another species, relabeled and commented."""
    validated = decoded['scientificNameValidated']
    model = decoded['scientificNameModel']
    reassigned = decoded[model.notna() & validated.notna() & (validated != model)]
    matched = match_observations(observations, reassigned, 'reassigned')

    result = stamp_human_review(matched.drop(columns='scientificNameValidated'))
    result['observationComments'] = (
        "Model false positive (" + matched['scientificName'] + "), reassigned manually to "
        + matched['scientificNameValidated']
    ).values
    result['scientificName'] = matched['scientificNameValidated'].values
    return result


def build_confirmed(observations, decoded):
    """Observations manually validated as Positive for the model species."""
    model = decoded['scientificNameModel']
    validated = decoded['scientificNameValidated']
    positive = decoded[
        model.notna() & (decoded['validationResult'] == 'Positive') & (validated == model)
    ]
    matched = match_observations(observations, positive, 'positive')
    result = stamp_human_review(matched.drop(columns='scientificNameValidated'))
    result['classificationProbability'] = 1
    return result


def get_manually_rejected_ids(observations, decoded):
    """observationIDs validated as Negative and not reassigned."""
    model = decoded['scientificNameModel']
    negative = decoded[
        model.notna() & (decoded['validationResult'] == 'Negative')
        & decoded['scientificNameValidated'].isna()
    ]
    return set(match_observations(observations, negative, 'negative')['observationID'])


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


def apply_confusions(candidates, confusions):
    """Relabel candidate observations of consistently confused species."""
    pairs = confusions[np.isfinite(confusions['appliedConfusionCutoff'])]
    relabeled = []
    for pair in pairs.itertuples():
        rows = candidates[
            (candidates['scientificName'] == pair.modelSpecies)
            & (candidates['classificationProbability'] >= pair.appliedConfusionCutoff)
        ].copy()
        rows['observationComments'] = (
            f"Inferred reassignment: {pair.modelSpecies} detections at this confidence "
            f"were consistently {pair.confusedSpecies} in manual validation"
        )
        rows['observationTags'] = INFERRED_TAG
        rows['scientificName'] = pair.confusedSpecies
        relabeled.append(rows)
    if not relabeled:
        return candidates.iloc[0:0]
    return pd.concat(relabeled)


def build_manual_annotations(observations, decoded):
    """New observation rows for manual annotations (segments without a model species)."""
    manual = decoded[decoded['scientificNameModel'].isna()]
    media_stem = manual['mediaID'].str.removesuffix('.WAV')
    new = pd.DataFrame({
        'deploymentID': media_stem.str.split('_').str[0].values,
        'mediaID': manual['mediaID'].values,
        'eventStart': manual['eventStart'].values,
        'eventEnd': manual['eventEnd'].values,
        'scientificName': manual['scientificNameValidated'].values,
    })

    unknown = sorted(set(new['deploymentID']) - set(observations['deploymentID']))
    if unknown:
        print(f"WARNING: deploymentIDs not found in observations: {unknown}")

    existing = observations[['mediaID', 'eventStart', 'eventEnd', 'scientificName']]
    already = new.merge(existing, how='left', indicator=True)['_merge'].eq('both').values
    if already.any():
        print(f"WARNING: {already.sum()} manual annotations already exist in observations, skipped")
    new = new[~already].reset_index(drop=True)

    new['observationID'] = observations['observationID'].max() + 1 + np.arange(len(new))
    new['observationLevel'] = 'interval'
    new['observationType'] = 'animal'
    new['classificationProbability'] = 1
    new['observationComments'] = 'Manual annotation'
    new = stamp_human_review(new)
    return new.reindex(columns=observations.columns)


def main():
    parser = argparse.ArgumentParser(
        description="Build the final observations table from model observations, thresholds and manual validations."
    )
    parser.add_argument('-i', '--observations', default='../../data/output_pa/species_detection/observations.csv',
                         help='Path to observations.csv (default: ../../data/output_pa/species_detection/observations.csv)')
    parser.add_argument('-t', '--thresholds', default='../../data/output_pa/validation/species_confidence_thresholds.csv',
                         help='Path to species_confidence_thresholds.csv (default: ../../data/output_pa/validation/species_confidence_thresholds.csv)')
    parser.add_argument('-v', '--validations', default='../../data/output_pa/validation/segment_validation_results.csv',
                         help='Path to segment_validation_results.csv (default: ../../data/output_pa/validation/segment_validation_results.csv)')
    parser.add_argument('-c', '--confusions', default='../../data/output_pa/validation/species_confusions.csv',
                         help='Path to species_confusions.csv, used with --apply-confusions (default: ../../data/output_pa/validation/species_confusions.csv)')
    parser.add_argument('-o', '--output', default='../../data/output_pa/validation/observations_thresholded.csv',
                         help='Path to save the final observations CSV (default: ../../data/output_pa/validation/observations_thresholded.csv)')
    parser.add_argument('--keep-unthresholded', action='store_true',
                         help='Keep observations for species missing from the thresholds table (default: drop them)')
    parser.add_argument('--apply-confusions', action='store_true',
                         help='Relabel observations of consistently confused species using the confusions table (default: off)')
    args = parser.parse_args()

    observations = pd.read_csv(args.observations)
    for column in ['observationComments', 'observationTags']:
        observations[column] = observations[column].astype('object')
    thresholds = pd.read_csv(args.thresholds)
    decoded = decode_segments(pd.read_csv(args.validations))

    reassigned = build_reassigned(observations, decoded)
    confirmed = build_confirmed(observations, decoded)
    rejected_ids = get_manually_rejected_ids(observations, decoded)

    # Source rows already handled by manual validation never go through the threshold
    reassigned_source_ids = set(reassigned['observationID'])
    handled_ids = reassigned_source_ids | set(confirmed['observationID']) | rejected_ids
    remaining = observations[~observations['observationID'].isin(handled_ids)]

    thresholded = filter_observations(remaining, thresholds, args.keep_unthresholded)

    inferred = observations.iloc[0:0]
    if args.apply_confusions:
        confusions = pd.read_csv(args.confusions)
        candidates = remaining[~remaining['observationID'].isin(thresholded['observationID'])]
        inferred = apply_confusions(candidates, confusions)

    manual = build_manual_annotations(observations, decoded)

    combined = (
        pd.concat([reassigned, confirmed, thresholded, inferred, manual])
        .drop_duplicates(subset='observationID')
        .sort_values('observationID')
    )

    print(
        f"\nObservations: {len(observations)} -> {len(combined)} total"
        f"\n  thresholded (machine):       {len(thresholded)}"
        f"\n  manually confirmed:          {len(confirmed)}"
        f"\n  reassigned manually:         {len(reassigned)}"
        f"\n  inferred reassignments:      {len(inferred)}"
        f"\n  manual annotations added:    {len(manual)}"
        f"\n  dropped (validated Negative): {len(rejected_ids)}"
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
