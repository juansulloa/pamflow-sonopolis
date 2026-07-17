"""
Identify which species need more validated segments to be includable by
02_find_model_threshold.py, versus which ones have an inherent precision
ceiling that more of the same validation won't fix.

Reads the raw segment_validation_results.csv (produced by
01_get_validation_status.py) and, for every species, finds the confidence
threshold that maximizes a one-sided 95% Jeffreys lower-bound precision
across ALL candidate thresholds -- deliberately ignoring the minimum-sample-
size floor that 02_find_model_threshold.py enforces, so a small but clean
subset still shows up here even though it wouldn't be trusted as a real cutoff.

Two distinct situations get produced:
    - 'closeGap': the best point-estimate precision found is already >=
      targetPrecision, but the sample behind it is too small for the
      statistical lower bound to clear that bar. More validated segments in
      that same confidence range -- if they keep coming back correct -- will
      close the gap. estimatedAdditionalSegmentsNeeded gives a best-case
      count (assuming every additional validation is correct) of how many
      more it would take.
    - 'lowCeiling': even the cleanest achievable subset doesn't reach
      targetPrecision. More validation of the same kind won't help; either
      genuinely higher-confidence segments need to be found and validated,
      or the species has a real false-positive problem at this site/season.
    - 'noData': no validated (Positive/Negative) segments at all yet.

If --thresholds-csv (the output of 02_find_model_threshold.py) is supplied,
species already assigned a usable cutoff there (cutoffSource in
{'model','sweep'}) are labeled 'alreadyIncluded' instead, so this report can
be filtered down to just the actual to-do list.

Usage:
    python 04_validation_priority.py [-i segment_validation_results.csv]
        [--thresholds-csv species_confidence_thresholds.csv]
        [--target-precision 0.95]
        [--output-csv ../../data/output/validation/validation_priority.csv]
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import beta

def jeffreys_lower_bound(successes, n, confidence=0.95):
    """One-sided Jeffreys lower confidence bound for a binomial proportion."""
    alpha = 1 - confidence
    return beta.ppf(alpha, successes + 0.5, n - successes + 0.5)

def load_data(validation_results, species):
    """Filter validation results to a single species' validated (Positive/Negative)
    segments, mapping validationResult to 0/1 in 'positive'.
    """
    data = validation_results[validation_results['scientificName'] == species]
    data = data[data['validationResult'].isin(['Positive', 'Negative'])].copy()
    data = data.dropna(subset=['classificationProbability'])
    data['positive'] = data['validationResult'].map({'Negative': 0, 'Positive': 1})
    return data

def best_threshold(data, target_precision):
    """Candidate threshold (over ALL observed confidence scores, no sample-size
    floor) that maximizes the Jeffreys lower-bound precision, and the raw
    precision/support behind it.
    """
    best = {'threshold': np.nan, 'lowerBound': -np.inf, 'precision': np.nan, 'support': 0, 'correct': 0}
    for candidate in np.sort(data['classificationProbability'].unique()):
        subset = data[data['classificationProbability'] >= candidate]
        n = len(subset)
        x = int(subset['positive'].sum())
        lb = jeffreys_lower_bound(x, n)
        if lb > best['lowerBound']:
            best = {'threshold': candidate, 'lowerBound': lb, 'precision': x / n, 'support': n, 'correct': x}
    return best

def estimate_additional_segments_needed(x, n, target_precision, max_search=500):
    """Best-case count of additional correct validations (added to x/n) needed
    for the Jeffreys lower bound to reach target_precision. NaN if it wouldn't
    happen within max_search additions.
    """
    m = 0
    while jeffreys_lower_bound(x + m, n + m) < target_precision:
        m += 1
        if m > max_search:
            return np.nan
    return m

def summarize_priority(species, data, target_precision):
    n_segments = len(data)
    if n_segments == 0:
        return {
            'scientificName': species,
            'numberOfValidatedSegments': 0,
            'bestThreshold': np.nan,
            'bestAchievablePrecision': np.nan,
            'bestAchievableLowerBound': np.nan,
            'supportAtBestThreshold': 0,
            'correctAtBestThreshold': 0,
            'priorityCategory': 'noData',
            'estimatedAdditionalSegmentsNeeded': np.nan,
        }

    best = best_threshold(data, target_precision)
    if best['precision'] >= target_precision:
        category = 'closeGap'
        needed = estimate_additional_segments_needed(best['correct'], best['support'], target_precision)
    else:
        category = 'lowCeiling'
        needed = np.nan

    return {
        'scientificName': species,
        'numberOfValidatedSegments': n_segments,
        'bestThreshold': best['threshold'],
        'bestAchievablePrecision': best['precision'],
        'bestAchievableLowerBound': best['lowerBound'],
        'supportAtBestThreshold': best['support'],
        'correctAtBestThreshold': best['correct'],
        'priorityCategory': category,
        'estimatedAdditionalSegmentsNeeded': needed,
    }

def main():
    parser = argparse.ArgumentParser(
        description='Prioritize which species need more validated segments to become includable.'
    )
    parser.add_argument('-i', '--input', default='../../data/output/validation/segment_validation_results.csv',
                         help='Path to segment_validation_results.csv (default: ../../data/output/validation/segment_validation_results.csv)')
    parser.add_argument('--thresholds-csv', default=None,
                         help='Optional path to species_confidence_thresholds.csv (output of '
                              '02_find_model_threshold.py). If given, species already assigned a '
                              "usable cutoff there are labeled 'alreadyIncluded'.")
    parser.add_argument('--target-precision', type=float, default=0.95,
                         help='Target precision to evaluate against (default: 0.95)')
    parser.add_argument('--output-csv', default='../../data/output/validation/validation_priority.csv',
                         help='Path to save the priority table (default: ../../data/output/validation/validation_priority.csv)')
    args = parser.parse_args()

    validation_results = pd.read_csv(args.input)
    species_list = sorted(validation_results['scientificName'].unique())

    rows = []
    for species in species_list:
        data = load_data(validation_results, species)
        rows.append(summarize_priority(species, data, args.target_precision))

    priority = pd.DataFrame(rows)

    if args.thresholds_csv:
        thresholds = pd.read_csv(args.thresholds_csv)
        already_included = set(thresholds.loc[thresholds['cutoffSource'].isin(['model', 'sweep']), 'scientificName'])
        priority.loc[priority['scientificName'].isin(already_included), 'priorityCategory'] = 'alreadyIncluded'

    category_order = {'closeGap': 0, 'lowCeiling': 1, 'noData': 2, 'alreadyIncluded': 3}
    priority['_sortKey'] = priority['priorityCategory'].map(category_order)
    priority = priority.sort_values(
        by=['_sortKey', 'estimatedAdditionalSegmentsNeeded', 'bestAchievableLowerBound'],
        ascending=[True, True, False],
    ).drop(columns='_sortKey')

    priority = priority.round(3)
    output_csv_path = Path(args.output_csv)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    priority.to_csv(output_csv_path, index=False)

    counts = priority['priorityCategory'].value_counts().to_dict()
    print(f"Priority breakdown: {counts}")
    print(f"Saved priority report for {len(priority)} species to {output_csv_path}")

if __name__ == "__main__":
    main()
