"""
Compute BirdNET confidence score thresholds for every species in a
segment_validation_results.csv file (produced by 01_get_validation_status.py),
which has columns:
    segmentName, scientificName, classificationProbability, validationResult
where validationResult is one of "Positive", "Negative", "NA".

For each species, computes two independent candidate cutoffs from its
validated (Positive/Negative) segments:
    - model: fits a null model and a logistic confidence model, and derives
      the confidence score cutoff for a 95% true positive rate. Assumes a
      monotonic relationship between confidence score and correctness; only
      trusted (used as appliedCutoff) when there's enough data
      (>= MIN_SAMPLE_SIZE segments), the confidence score is a statistically
      significant predictor (likelihood-ratio p < SIGNIFICANCE_LEVEL), and the
      cutoff isn't extrapolated beyond the range of validated confidence scores.
    - sweep: assumption-free. Sweeps every observed confidence score as a
      candidate cutoff and picks the lowest one whose Jeffreys lower-bound
      precision (among validated segments at or above it, requiring
      >= MIN_SAMPLE_SIZE of them) reaches the target precision. Unlike the
      model, this doesn't assume the relationship is monotonic, so it still
      works for species where false positives appear at higher confidence
      scores than some true positives.

--method controls which candidate becomes appliedCutoff/cutoffSource:
    - 'model': only the model cutoff, if trustworthy.
    - 'sweep': only the sweep cutoff, if one exists.
    - 'auto' (default): try sweep first (fewer assumptions), then model.
    - If neither is usable, the species is excluded (appliedCutoff = inf).
Both candidates are evaluated at PRIMARY_TARGET_PRECISION (module constant,
default 0.95) for this decision.
All candidate columns are always computed and included in the output
regardless of --method, so results can be compared across methods.

The target precision/TPR levels reported are configured via module constants
rather than hardcoded column names:
    - MODEL_CUTOFF_PROBABILITIES (default [0.95, 0.99]) -> cutoffConfidence{p}
    - SWEEP_TARGET_PRECISIONS (default [0.95, 0.99]) -> sweepCutoff{p}
    - JEFFREYS_CONFIDENCE (default 0.95) -> precisionLowerBound{p}
    - PRIMARY_TARGET_PRECISION (default 0.95) selects which cutoffConfidence{p}
      and sweepCutoff{p} columns feed appliedCutoff/cutoffSource; it must be a
      member of both MODEL_CUTOFF_PROBABILITIES and SWEEP_TARGET_PRECISIONS.
Edit these lists to add/remove/rename levels, e.g. set
MODEL_CUTOFF_PROBABILITIES = [0.90, 0.95] to get cutoffConfidence90/95 instead
of cutoffConfidence95/99.

Usage:
    python 02_find_model_threshold.py [-i segment_validation_results.csv] [--plot]
        [--method {auto,model,sweep}]
        [-o ../../data/output/validation/figures]
        [--output-csv ../../data/output/validation/species_confidence_thresholds.csv]

    -i/--input defaults to segment_validation_results.csv in the current directory.
    --plot saves a confidence score plot per fitted species (disabled by default).
    --method selects which candidate cutoff is used as appliedCutoff (default: auto).
    -o/--output-dir is where plots are saved when --plot is set
        (default: ../../data/output/validation/figures).
    --output-csv is where the summary table is saved
        (default: ../../data/output/validation/species_confidence_thresholds.csv).
"""
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import glm
from scipy.stats import beta, chi2
import matplotlib.pyplot as plt

# Minimum validated segments required to trust a model-based cutoff, and
# minimum segments at/above a candidate threshold to trust a sweep cutoff.
MIN_SAMPLE_SIZE = 20

# Significance level for the confidence-model likelihood-ratio test.
SIGNIFICANCE_LEVEL = 0.05

# True-positive-rate levels the logistic model derives a cutoff for. Column
# names are generated as cutoffConfidence{int(p*100)}, e.g. cutoffConfidence95.
MODEL_CUTOFF_PROBABILITIES = [0.90, 0.95]

# Target precision levels the threshold sweep searches for. Column names are
# generated as sweepCutoff{int(p*100)}, e.g. sweepCutoff95.
SWEEP_TARGET_PRECISIONS = [0.90, 0.95]

# Confidence level for the Jeffreys lower bound on precision. Column name is
# generated as precisionLowerBound{int(c*100)}.
JEFFREYS_CONFIDENCE = 0.95

# Which target precision drives appliedCutoff/cutoffSource selection. Must be
# present in both MODEL_CUTOFF_PROBABILITIES and SWEEP_TARGET_PRECISIONS.
PRIMARY_TARGET_PRECISION = 0.90

assert PRIMARY_TARGET_PRECISION in MODEL_CUTOFF_PROBABILITIES
assert PRIMARY_TARGET_PRECISION in SWEEP_TARGET_PRECISIONS


def pct_label(p):
    """Format a probability as the integer-percent suffix used in column names."""
    return str(int(round(p * 100)))


def read_validation_results(file_path):
    """Read the segment_validation_results.csv file."""
    return pd.read_csv(file_path)

def load_data(validation_results, species):
    """Filter validation results to a single species' validated (Positive/Negative)
    segments, mapping validationResult to 0/1 in 'positive'.
    """
    data = validation_results[validation_results['scientificName'] == species]
    data = data[data['validationResult'].isin(['Positive', 'Negative'])].copy()
    data = data.dropna(subset=['classificationProbability'])
    data['positive'] = data['validationResult'].map({'Negative': 0, 'Positive': 1})
    return data

def fit_models(data):
    """Fit null and confidence models, and return both models."""
    null_model = glm('positive ~ 1', data=data, family=sm.families.Binomial()).fit()
    conf_model = glm('positive ~ classificationProbability', data=data, family=sm.families.Binomial()).fit()
    return null_model, conf_model

def create_aic_table(null_model, conf_model):
    """Create and return an AIC table comparing the null and confidence models."""
    aic_table = pd.DataFrame({
        'Model': ['null_model', 'conf_model'],
        'AIC': [null_model.aic, conf_model.aic]
    })
    return aic_table.sort_values(by='AIC', ascending=True)

def calculate_cutoff(conf_model, probability=0.95):
    """Calculate and return the cutoff confidence score for a given true positive rate."""
    logit_value = np.log(probability / (1 - probability))
    cutoff = (logit_value - conf_model.params['Intercept']) / conf_model.params['classificationProbability']
    return cutoff

def jeffreys_lower_bound(successes, n, confidence=JEFFREYS_CONFIDENCE):
    """One-sided Jeffreys lower confidence bound for a binomial proportion."""
    alpha = 1 - confidence
    return beta.ppf(alpha, successes + 0.5, n - successes + 0.5)

def likelihood_ratio_pvalue(null_model, conf_model):
    """One-df likelihood-ratio test p-value comparing conf_model to null_model.

    Low sample sizes can let a confidence model fit without erroring even
    though classificationProbability has no real relationship to correctness
    for that species; this quantifies whether the model is actually informative.
    """
    lr_stat = 2 * (conf_model.llf - null_model.llf)
    return chi2.sf(lr_stat, df=1)

def threshold_sweep_cutoff(data, target_precision, min_sample_size=MIN_SAMPLE_SIZE):
    """Lowest confidence threshold whose Jeffreys lower-bound precision, among
    validated segments at or above it, meets target_precision.

    Makes no assumption about the relationship between score and correctness
    (unlike the logistic model), so it still works when false positives occur
    at higher confidence scores than some true positives. Only considers
    thresholds backed by >= min_sample_size segments at or above them, so a
    threshold isn't chosen on a handful of lucky points. Returns NaN if no
    threshold clears the bar.
    """
    for candidate in np.sort(data['classificationProbability'].unique()):
        subset = data[data['classificationProbability'] >= candidate]
        n = len(subset)
        if n < min_sample_size:
            continue
        n_positive = int(subset['positive'].sum())
        if jeffreys_lower_bound(n_positive, n) >= target_precision:
            return candidate
    return np.nan

def select_cutoff(row, method):
    """Fill in appliedCutoff/cutoffSource on a summary row for the chosen method.

    Reads only already-computed diagnostic columns on `row`, so this can be
    re-run against a saved summary table to compare methods without refitting
    anything. Uses the PRIMARY_TARGET_PRECISION columns as the candidates.
    """
    sweep_col = f'sweepCutoff{pct_label(PRIMARY_TARGET_PRECISION)}'
    model_col = f'cutoffConfidence{pct_label(PRIMARY_TARGET_PRECISION)}'

    def model_is_trustworthy():
        return (
            row['numberOfValidatedSegments'] >= MIN_SAMPLE_SIZE
            and row['confModelPValue'] < SIGNIFICANCE_LEVEL
            and row['minValidatedConfidence'] <= row[model_col] <= row['maxValidatedConfidence']
        )

    def try_sweep():
        if not np.isnan(row[sweep_col]):
            row['appliedCutoff'] = row[sweep_col]
            row['cutoffSource'] = 'sweep'
            return True
        return False

    def try_model():
        if model_is_trustworthy():
            row['appliedCutoff'] = row[model_col]
            row['cutoffSource'] = 'model'
            return True
        return False

    row['appliedCutoff'] = np.inf
    row['cutoffSource'] = 'exclude'

    if method == 'sweep':
        try_sweep()
    elif method == 'model':
        try_model()
    elif method == 'auto':
        try_sweep() or try_model()
    else:
        raise ValueError(f"Unknown method: {method}")

    return row

def plot_results(
        focal_species, data, prediction_range_conf, predictions_conf, cutoff, proba, aic_table,
        output_dir):
    """Plot the scatter plot of data, model predictions, and the cutoff line."""
    plt.figure(figsize=(10, 6))

    # Scatter plot of the original data (basic matplotlib)
    plt.scatter(data['classificationProbability'], data['positive'],
                c='black', s=100, alpha=0.2)

    # Add the line for model predictions
    plt.plot(prediction_range_conf, predictions_conf,
             linewidth=4, color=(0, 0.75, 1, 0.5))

    # Add the vertical line for the cutoff at 95% confidence
    plt.axvline(x=cutoff, color='red', linewidth=2)

    # Add textbox with aic_table information at lower right corner
    aic_text = aic_table.to_string(index=False)
    plt.gcf().text(0.87, 0.12, aic_text, fontsize=10,
                   verticalalignment='bottom', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                   color='grey')

    # Customize the plot
    plt.title(f'Confidence scores - pr(tpr={proba}) = {round(cutoff, 3)}')
    plt.xlabel('Confidence score')
    plt.ylabel('pr(BirdNET prediction is correct)')
    plt.xlim([min(prediction_range_conf), max(prediction_range_conf)])

    # Save plot
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / f"{focal_species.replace(' ', '_')}_confidence_score.png", dpi=300)
    plt.close()

def summarize_species(species, data, method):
    """Compute every diagnostic value for a species (model-based and sweep-based
    cutoffs, precision lower bound), select appliedCutoff/cutoffSource per
    `method`, and return (row, null_model, conf_model).
    """
    n_segments = len(data)
    n_positive = int(data['positive'].sum()) if n_segments else 0
    tpr = data['positive'].mean() if n_segments else np.nan

    jeffreys_col = f'precisionLowerBound{pct_label(JEFFREYS_CONFIDENCE)}'
    model_cols = [f'cutoffConfidence{pct_label(p)}' for p in MODEL_CUTOFF_PROBABILITIES]
    sweep_cols = [f'sweepCutoff{pct_label(p)}' for p in SWEEP_TARGET_PRECISIONS]

    row = {
        'scientificName': species,
        'numberOfValidatedSegments': n_segments,
        'nullModelAic': np.nan,
        'confModelAic': np.nan,
        'confModelPValue': np.nan,
        **{col: np.nan for col in model_cols},
        'truePositiveRate': tpr,
        jeffreys_col: np.nan,
        'minValidatedConfidence': np.nan,
        'maxValidatedConfidence': np.nan,
        **{col: np.nan for col in sweep_cols},
        'appliedCutoff': np.inf,
        'cutoffSource': 'exclude',
    }

    if n_segments == 0:
        print(f"'{species}': no validated segments; excluding")
        return row, None, None

    row['minValidatedConfidence'] = data['classificationProbability'].min()
    row['maxValidatedConfidence'] = data['classificationProbability'].max()
    row[jeffreys_col] = jeffreys_lower_bound(n_positive, n_segments)
    for p, col in zip(SWEEP_TARGET_PRECISIONS, sweep_cols):
        row[col] = threshold_sweep_cutoff(data, p)

    null_model = conf_model = None
    if n_segments >= 2 and data['positive'].nunique() == 2:
        try:
            null_model, conf_model = fit_models(data)
        except Exception as exc:
            print(f"'{species}': model fitting failed ({exc})")

    if conf_model is not None:
        row['nullModelAic'] = null_model.aic
        row['confModelAic'] = conf_model.aic
        row['confModelPValue'] = likelihood_ratio_pvalue(null_model, conf_model)
        for p, col in zip(MODEL_CUTOFF_PROBABILITIES, model_cols):
            row[col] = calculate_cutoff(conf_model, p)

    select_cutoff(row, method)

    print(
        f"{species}: n={n_segments}, source={row['cutoffSource']}, "
        f"appliedCutoff={row['appliedCutoff']}"
    )

    return row, null_model, conf_model

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Compute BirdNET confidence thresholds for every species in a segment_validation_results.csv file.'
    )
    parser.add_argument('-i', '--input', default='../../data/output/validation/segment_validation_results.csv',
                         help='Path to segment_validation_results.csv (default: ../../data/output/validation/segment_validation_results.csv)')
    parser.add_argument('--method', choices=['auto', 'model', 'sweep'], default='auto',
                         help="Which candidate cutoff to use as appliedCutoff: 'model', 'sweep', "
                              "or 'auto' (sweep first, then model). Default: auto.")
    parser.add_argument('--plot', action='store_true',
                         help='Save a confidence score plot per fitted species (disabled by default)')
    parser.add_argument('-o', '--output-dir', default='../../data/output/validation/figures',
                         help='Directory to save plots when --plot is set (default: ../../data/output/validation/figures)')
    parser.add_argument('--output-csv', default='../../data/output/validation/species_confidence_thresholds.csv',
                         help='Path to save the summary CSV (default: ../../data/output/validation/species_confidence_thresholds.csv)')
    args = parser.parse_args()

    validation_results = read_validation_results(args.input)
    species_list = sorted(validation_results['scientificName'].unique())

    summary_rows = []
    for species in species_list:
        data = load_data(validation_results, species)
        row, null_model, conf_model = summarize_species(species, data, args.method)
        summary_rows.append(row)

        if conf_model is not None and args.plot:
            aic_table = create_aic_table(null_model, conf_model)
            prediction_range_conf = np.arange(0, 1.001, 0.001)
            predictions_conf = conf_model.predict(pd.DataFrame({'classificationProbability': prediction_range_conf}))
            plot_results(
                species,
                data,
                prediction_range_conf,
                predictions_conf,
                row[f'cutoffConfidence{pct_label(PRIMARY_TARGET_PRECISION)}'],
                PRIMARY_TARGET_PRECISION,
                aic_table,
                args.output_dir,
            )

    summary = pd.DataFrame(summary_rows)
    summary = summary.round(3)
    output_csv_path = Path(args.output_csv)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_csv_path, index=False)
    print(f"\nSaved summary for {len(summary)} species to {output_csv_path}")

# Run the main function
if __name__ == "__main__":
    main()
