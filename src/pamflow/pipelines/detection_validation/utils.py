"""Helpers for the detection_validation pipeline.

Three groups, one per node:
    1. Validation table: turn raw annotation rows into the segment validation table.
    2. Species thresholds: per-species confidence cutoffs and confusion table.
    3. Observations: apply manual validations and thresholds to the observations.
"""

import logging
from pathlib import PureWindowsPath

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import beta, chi2
from statsmodels.formula.api import glm

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 1. Validation table
# ---------------------------------------------------------------------------

VALIDATION_COLUMNS = [
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


def normalize_species(name):
    """Convert 'Genus_species' to 'Genus species'; return None for blanks."""
    if pd.isna(name):
        return None
    name = str(name).replace("_", " ").strip()
    return name or None


def segment_name_from_path(file_name: str) -> str:
    """Basename of a path that may use Windows or POSIX separators."""
    return PureWindowsPath(str(file_name).strip()).name


def build_validation_table(raw: pd.DataFrame) -> pd.DataFrame:
    """Map raw annotation rows (one per validated segment) to the segment
    validation table.

    Mapping of ``validationCategory``:
        positive   -> Positive, scientificNameValidated = model species
        negative   -> Negative
        reassigned -> Reassigned, scientificNameValidated = annotator species
        uncertain  -> Uncertain
        other      -> NA (not validated)
    """
    missing = [c for c in REQUIRED_INPUT_COLUMNS if c not in raw.columns]
    if missing:
        raise ValueError(f"Missing required columns in the validation results: {missing}")

    records = []
    for row in raw.itertuples(index=False):
        if pd.isna(row.fileName):
            logger.warning(
                "Row without fileName, skipped (species: %s)", row.scientificNameProposed
            )
            continue

        segment_name = segment_name_from_path(row.fileName)
        model_species = normalize_species(row.scientificNameProposed)
        category = (
            str(row.validationCategory).strip().lower()
            if pd.notna(row.validationCategory)
            else ""
        )

        if category not in CATEGORY_TO_RESULT:
            logger.warning(
                "Unknown validationCategory %r treated as NA: %s", category, segment_name
            )
        validation = CATEGORY_TO_RESULT.get(category, "NA")

        if category == "positive":
            validated_species = model_species
        elif category == "reassigned":
            validated_species = normalize_species(row.scientificNameValidated)
            if validated_species is None:
                logger.warning(
                    "Reassigned segment without scientificNameValidated: %s", segment_name
                )
        else:
            validated_species = None

        probability = parse_classification_probability(segment_name)
        if probability is None:
            logger.warning("Could not parse classificationProbability: %s", segment_name)

        records.append(
            {
                "segmentName": segment_name,
                "scientificNameModel": model_species,
                "classificationProbability": probability,
                "validationResult": validation,
                "scientificNameValidated": validated_species,
                "classifiedBy": row.classifiedBy,
                "favorite": row.favorite,
                "validationComment": row.validationComment,
            }
        )

    table = pd.DataFrame(records, columns=VALIDATION_COLUMNS)

    duplicated = table[table.duplicated(subset="segmentName", keep=False)]
    if not duplicated.empty:
        raise ValueError(
            f"Duplicate segmentName keys in the validation table: "
            f"{sorted(duplicated['segmentName'].unique())[:10]} "
            f"({len(duplicated)} rows affected)"
        )
    return table


def log_validation_summary(table: pd.DataFrame) -> None:
    """Log counts of the validation table."""
    logger.info("Total segments: %d", len(table))
    logger.info(
        "validationResult counts: %s", table["validationResult"].value_counts().to_dict()
    )
    reassigned = table[table["validationResult"] == "Reassigned"]
    for (original, assigned), n in sorted(
        reassigned.groupby(["scientificNameModel", "scientificNameValidated"]).size().items()
    ):
        logger.info("Reassigned %s -> %s: %d", original, assigned, n)


# ---------------------------------------------------------------------------
# 2. Species thresholds
# ---------------------------------------------------------------------------

# Significance level for the confidence-model likelihood-ratio test.
SIGNIFICANCE_LEVEL = 0.05

# Confidence level for the Jeffreys lower bound on precision. Column name is
# generated as precisionLowerBound{int(c*100)}.
JEFFREYS_CONFIDENCE = 0.95

# Secondary level always reported next to the primary true positive rate.
SECONDARY_RATE = 0.95

CONFUSION_COLUMNS = [
    "modelSpecies",
    "confusedSpecies",
    "nResolved",
    "nReassignedToConfused",
    "nPositiveOwn",
    "consistency",
    "appliedConfusionCutoff",
]


def target_levels(true_positive_rate: float) -> list:
    """Levels (true positive rate / target precision) reported per species.

    The primary rate drives appliedCutoff; a secondary 0.95 level is reported too.
    """
    if not 0 < true_positive_rate < 1:
        raise ValueError(
            f"true_positive_rate must be between 0 and 1 (exclusive), got {true_positive_rate}"
        )
    return sorted({true_positive_rate, SECONDARY_RATE})


def pct_label(p):
    """Format a probability as the integer-percent suffix used in column names."""
    return str(int(round(p * 100)))


def split_segments(validation_results):
    """Split validation results into model-detected segments, reassigned
    segments and manual annotations.

    Manual annotations have no model species (scientificNameModel is NA).
    Reassigned segments have a validated species different from the model's.
    Returns (model_segments, reassigned, n_manual).
    """
    is_manual = validation_results["scientificNameModel"].isna()
    validated = validation_results["scientificNameValidated"]
    is_reassigned = (
        ~is_manual
        & validated.notna()
        & (validated != validation_results["scientificNameModel"])
    )
    model_segments = validation_results[~is_manual & ~is_reassigned]
    return model_segments, validation_results[is_reassigned], int(is_manual.sum())


def load_data(validation_results, species):
    """Filter validation results to a single species' validated segments, mapping
    them to 0/1 in 'positive'.

    Positive segments are 1. Negative segments and segments reassigned to a
    different species are 0: a reassigned segment is a false positive of the
    model species.
    """
    data = validation_results[validation_results["scientificNameModel"] == species]
    validated = data["scientificNameValidated"]
    is_reassigned = validated.notna() & (validated != data["scientificNameModel"])
    data = data[data["validationResult"].isin(["Positive", "Negative"]) | is_reassigned].copy()
    data = data.dropna(subset=["classificationProbability"])
    data["positive"] = (data["validationResult"] == "Positive").astype(int)
    return data


def fit_models(data):
    """Fit null and confidence models, and return both models."""
    null_model = glm("positive ~ 1", data=data, family=sm.families.Binomial()).fit()
    conf_model = glm(
        "positive ~ classificationProbability", data=data, family=sm.families.Binomial()
    ).fit()
    return null_model, conf_model


def create_aic_table(null_model, conf_model):
    """Create and return an AIC table comparing the null and confidence models."""
    aic_table = pd.DataFrame(
        {"Model": ["null_model", "conf_model"], "AIC": [null_model.aic, conf_model.aic]}
    )
    return aic_table.sort_values(by="AIC", ascending=True)


def calculate_cutoff(conf_model, probability=0.95):
    """Cutoff confidence score for a given true positive rate."""
    logit_value = np.log(probability / (1 - probability))
    return (logit_value - conf_model.params["Intercept"]) / conf_model.params[
        "classificationProbability"
    ]


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


def threshold_sweep_cutoff(data, target_precision, min_sample_size):
    """Lowest confidence threshold whose Jeffreys lower-bound precision, among
    validated segments at or above it, meets target_precision.

    Makes no assumption about the relationship between score and correctness
    (unlike the logistic model), so it still works when false positives occur
    at higher confidence scores than some true positives. Only considers
    thresholds backed by >= min_sample_size segments at or above them. Returns
    NaN if no threshold clears the bar.
    """
    for candidate in np.sort(data["classificationProbability"].unique()):
        subset = data[data["classificationProbability"] >= candidate]
        n = len(subset)
        if n < min_sample_size:
            continue
        n_positive = int(subset["positive"].sum())
        if jeffreys_lower_bound(n_positive, n) >= target_precision:
            return candidate
    return np.nan


def select_cutoff(row, method, primary_rate, min_sample_size):
    """Fill in appliedCutoff/cutoffSource on a summary row for the chosen method.

    Reads only already-computed diagnostic columns on `row`. Uses the columns
    of the primary rate as the candidates.
    """
    sweep_col = f"sweepCutoff{pct_label(primary_rate)}"
    model_col = f"cutoffConfidence{pct_label(primary_rate)}"

    def model_is_trustworthy():
        return (
            row["numberOfValidatedSegments"] >= min_sample_size
            and row["confModelPValue"] < SIGNIFICANCE_LEVEL
            and row["minValidatedConfidence"]
            <= row[model_col]
            <= row["maxValidatedConfidence"]
        )

    def try_sweep():
        if not np.isnan(row[sweep_col]):
            row["appliedCutoff"] = row[sweep_col]
            row["cutoffSource"] = "sweep"
            return True
        return False

    def try_model():
        if model_is_trustworthy():
            row["appliedCutoff"] = row[model_col]
            row["cutoffSource"] = "model"
            return True
        return False

    row["appliedCutoff"] = np.inf
    row["cutoffSource"] = "exclude"

    if method == "sweep":
        try_sweep()
    elif method == "model":
        try_model()
    elif method == "auto":
        try_sweep() or try_model()
    else:
        raise ValueError(f"Unknown method: {method}")

    return row


def plot_results(
    focal_species, data, prediction_range_conf, predictions_conf, cutoff, proba, aic_table
):
    """Scatter of the data, model predictions and the cutoff line. Returns the figure."""
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.scatter(
        data["classificationProbability"], data["positive"], c="black", s=100, alpha=0.2
    )
    ax.plot(prediction_range_conf, predictions_conf, linewidth=4, color=(0, 0.75, 1, 0.5))
    ax.axvline(x=cutoff, color="red", linewidth=2)

    fig.text(
        0.87,
        0.12,
        aic_table.to_string(index=False),
        fontsize=10,
        verticalalignment="bottom",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        color="grey",
    )

    ax.set_title(
        f"{focal_species} \n Confidence scores - pr(tpr={proba}) = {round(cutoff, 3)}"
    )
    ax.set_xlabel("Confidence score")
    ax.set_ylabel("pr(BirdNET prediction is correct)")
    ax.set_xlim([min(prediction_range_conf), max(prediction_range_conf)])
    return fig


def summarize_species(species, data, method, primary_rate, levels, min_sample_size):
    """Compute every diagnostic value for a species (model-based and sweep-based
    cutoffs, precision lower bound), select appliedCutoff/cutoffSource per
    `method`, and return (row, null_model, conf_model).
    """
    n_segments = len(data)
    n_positive = int(data["positive"].sum()) if n_segments else 0
    tpr = data["positive"].mean() if n_segments else np.nan

    jeffreys_col = f"precisionLowerBound{pct_label(JEFFREYS_CONFIDENCE)}"
    model_cols = [f"cutoffConfidence{pct_label(p)}" for p in levels]
    sweep_cols = [f"sweepCutoff{pct_label(p)}" for p in levels]

    row = {
        "scientificName": species,
        "numberOfValidatedSegments": n_segments,
        "nullModelAic": np.nan,
        "confModelAic": np.nan,
        "confModelPValue": np.nan,
        **{col: np.nan for col in model_cols},
        "truePositiveRate": tpr,
        jeffreys_col: np.nan,
        "minValidatedConfidence": np.nan,
        "maxValidatedConfidence": np.nan,
        **{col: np.nan for col in sweep_cols},
        "appliedCutoff": np.inf,
        "cutoffSource": "exclude",
    }

    if n_segments == 0:
        logger.info("'%s': no validated segments; excluding", species)
        return row, None, None

    row["minValidatedConfidence"] = data["classificationProbability"].min()
    row["maxValidatedConfidence"] = data["classificationProbability"].max()
    row[jeffreys_col] = jeffreys_lower_bound(n_positive, n_segments)
    for p, col in zip(levels, sweep_cols):
        row[col] = threshold_sweep_cutoff(data, p, min_sample_size)

    null_model = conf_model = None
    if n_segments >= 2 and data["positive"].nunique() == 2:
        try:
            null_model, conf_model = fit_models(data)
        except Exception as exc:  # noqa: BLE001 - model fitting can fail in many ways
            logger.warning("'%s': model fitting failed (%s)", species, exc)

    if conf_model is not None:
        row["nullModelAic"] = null_model.aic
        row["confModelAic"] = conf_model.aic
        row["confModelPValue"] = likelihood_ratio_pvalue(null_model, conf_model)
        for p, col in zip(levels, model_cols):
            row[col] = calculate_cutoff(conf_model, p)

    select_cutoff(row, method, primary_rate, min_sample_size)

    logger.info(
        "%s: n=%d, source=%s, appliedCutoff=%s",
        species,
        n_segments,
        row["cutoffSource"],
        row["appliedCutoff"],
    )
    return row, null_model, conf_model


def compute_confusions(model_and_reassigned, thresholds, primary_rate, levels, min_sample_size):
    """Build the species confusion table from reassigned segments.

    For each model species A with segments reassigned to a species B, takes all
    of A's resolved segments (Positive/Negative, plus every reassigned one) and
    sweeps for the lowest confidence score above which they are consistently B
    (positive = reassigned to B). The pair cutoff is only applied
    (appliedConfusionCutoff) when A has no usable cutoff of its own, i.e. its
    appliedCutoff in `thresholds` is infinite or A is missing from it.
    """
    sweep_cols = [f"sweepCutoff{pct_label(p)}" for p in levels]
    primary_col = f"sweepCutoff{pct_label(primary_rate)}"
    own_cutoffs = thresholds.set_index("scientificName")["appliedCutoff"]

    validated = model_and_reassigned["scientificNameValidated"]
    model = model_and_reassigned["scientificNameModel"]
    is_reassigned = validated.notna() & (validated != model)
    resolved = model_and_reassigned[
        model_and_reassigned["validationResult"].isin(["Positive", "Negative"])
        | is_reassigned
    ].dropna(subset=["classificationProbability"])

    rows = []
    reassigned = model_and_reassigned[is_reassigned]
    for (species, confused), _ in reassigned.groupby(
        ["scientificNameModel", "scientificNameValidated"]
    ):
        data = resolved[resolved["scientificNameModel"] == species].copy()
        data["positive"] = (data["scientificNameValidated"] == confused).astype(int)
        n_to_confused = int(data["positive"].sum())
        n_positive_own = int((data["validationResult"] == "Positive").sum())

        row = {
            "modelSpecies": species,
            "confusedSpecies": confused,
            "nResolved": len(data),
            "nReassignedToConfused": n_to_confused,
            "nPositiveOwn": n_positive_own,
            "consistency": n_to_confused / len(data) if len(data) else np.nan,
        }
        for p, col in zip(levels, sweep_cols):
            row[col] = threshold_sweep_cutoff(data, p, min_sample_size)

        own_cutoff = own_cutoffs.get(species, np.inf)
        eligible = np.isinf(own_cutoff) and not np.isnan(row[primary_col])
        row["appliedConfusionCutoff"] = row[primary_col] if eligible else np.inf
        rows.append(row)

    columns = CONFUSION_COLUMNS[:6] + sweep_cols + CONFUSION_COLUMNS[6:]
    return pd.DataFrame(rows, columns=columns)


# ---------------------------------------------------------------------------
# 3. Observations
# ---------------------------------------------------------------------------

MATCH_COLUMNS = [
    "mediaID",
    "eventStart",
    "eventEnd",
    "scientificName",
    "classificationProbability",
]
INFERRED_TAG = "inferred_reassignment"
FAVORITE_TAG = "favorite"

# validation column -> temporary column carried through matching (renamed to
# avoid clashing with the observations' own classifiedBy)
REVIEW_COLUMNS = {
    "classifiedBy": "reviewer",
    "favorite": "reviewFavorite",
    "validationComment": "reviewComment",
}


def normalize_observations(observations: pd.DataFrame) -> pd.DataFrame:
    """Numeric observationID and float64 scores/times rounded to 3 decimals, so
    exact matching against segment names is reliable whatever the loader dtype.
    """
    observations = observations.copy()
    observations["observationID"] = pd.to_numeric(observations["observationID"]).astype("int64")
    for column in ["eventStart", "eventEnd", "classificationProbability"]:
        observations[column] = observations[column].astype("float64").round(3)
    for column in ["observationComments", "observationTags"]:
        observations[column] = observations[column].astype("object")
    return observations


def decode_segments(validations):
    """Decode segmentName into classificationProbability, mediaID, eventStart
    and eventEnd, keeping the validation columns.
    """
    stem = validations["segmentName"].str.removesuffix(".wav")
    parts = stem.str.split("_")
    decoded = pd.DataFrame(
        {
            "segmentName": validations["segmentName"].values,
            "classificationProbability": parts.str[0].astype(float).round(3).values,
            "mediaID": (parts.str[1:-2].str.join("_") + ".WAV").values,
            "eventStart": parts.str[-2].astype(float).round(3).values,
            "eventEnd": parts.str[-1].astype(float).round(3).values,
            "scientificNameModel": validations["scientificNameModel"].values,
            "validationResult": validations["validationResult"].values,
            "scientificNameValidated": validations["scientificNameValidated"].values,
        }
    )
    for source, target in REVIEW_COLUMNS.items():
        if source in validations.columns:
            decoded[target] = validations[source].values
    return decoded


def join_text(first, second, sep):
    """Element-wise join of two text Series, skipping empty/NA parts."""
    first = first.fillna("").astype(str)
    second = second.fillna("").astype(str)
    both = (first != "") & (second != "")
    joined = first.where(~both, first + sep + second).where(first != "", second)
    return joined.where(both | (first != "") | (second != ""), None).astype("object")


def match_observations(observations, decoded, label):
    """Return the rows of `observations` matching the decoded model segments,
    with scientificNameValidated added. Unmatched segments are reported.
    """
    right = decoded.rename(columns={"scientificNameModel": "scientificName"})[
        MATCH_COLUMNS
        + ["scientificNameValidated"]
        + [c for c in REVIEW_COLUMNS.values() if c in decoded.columns]
    ]
    matched = observations.merge(right, on=MATCH_COLUMNS, how="inner")

    # Some segments were exported with the observation window padded with
    # context (e.g. observation 24.0-27.0 -> segment 21.5-29.5). Match these on
    # media, species and score, with the observation window inside the segment.
    unmatched = right.merge(
        observations[MATCH_COLUMNS].drop_duplicates(),
        on=MATCH_COLUMNS,
        how="left",
        indicator=True,
    )
    unmatched = unmatched[unmatched["_merge"] == "left_only"].drop(columns="_merge")
    n_unmatched = len(unmatched)
    if len(unmatched) > 0:
        candidates = (
            unmatched.reset_index(drop=True)
            .rename_axis("segmentIndex")
            .reset_index()
            .merge(
                observations,
                on=["mediaID", "scientificName", "classificationProbability"],
                suffixes=("_segment", ""),
            )
        )
        candidates = candidates[
            (candidates["eventStart_segment"] <= candidates["eventStart"])
            & (candidates["eventEnd"] <= candidates["eventEnd_segment"])
        ].copy()
        candidates["centerDistance"] = (
            (candidates["eventStart"] + candidates["eventEnd"])
            - (candidates["eventStart_segment"] + candidates["eventEnd_segment"])
        ).abs()
        n_ambiguous = int((candidates.groupby("segmentIndex").size() > 1).sum())
        if n_ambiguous > 0:
            logger.warning(
                "%d %s segments matched several observations, using the one closest "
                "to the segment center",
                n_ambiguous,
                label,
            )
        padded = candidates.sort_values("centerDistance").drop_duplicates(
            subset="segmentIndex"
        )
        matched = pd.concat([matched, padded[matched.columns]])
        n_unmatched -= len(padded)

    # Several segments can point to the same observation (e.g. a padded and an
    # exact clip of one detection), so the output is deduplicated.
    matched = matched.drop_duplicates(subset="observationID")
    if n_unmatched > 0:
        logger.warning("%d %s segments did not match any observation", n_unmatched, label)
    return matched


def stamp_human_review(rows, fallback_classified_by, timestamp):
    """Overwrite classification fields to reflect human review, using the
    reviewer, comment and favorite information when available.
    """
    rows = rows.copy()
    rows["classificationMethod"] = "human"
    rows["classifiedBy"] = fallback_classified_by
    rows["classificationTimestamp"] = timestamp

    if "reviewer" in rows:
        rows["classifiedBy"] = rows["reviewer"].where(
            rows["reviewer"].notna(), fallback_classified_by
        )
    if "reviewComment" in rows:
        rows["observationComments"] = join_text(
            rows["observationComments"], rows["reviewComment"], " | "
        )
    if "reviewFavorite" in rows:
        favorite = rows["reviewFavorite"].astype(str).str.lower().eq("true")
        tag = pd.Series(
            np.where(favorite, FAVORITE_TAG, None), index=rows.index, dtype="object"
        )
        rows["observationTags"] = join_text(rows["observationTags"], tag, ", ")
    return rows.drop(columns=[c for c in REVIEW_COLUMNS.values() if c in rows.columns])


def build_reassigned(observations, decoded, fallback_classified_by, timestamp):
    """Observations manually reassigned to another species, relabeled and commented."""
    validated = decoded["scientificNameValidated"]
    model = decoded["scientificNameModel"]
    reassigned = decoded[model.notna() & validated.notna() & (validated != model)]
    matched = match_observations(observations, reassigned, "reassigned")

    result = matched.drop(columns="scientificNameValidated")
    result["observationComments"] = (
        "Model false positive ("
        + matched["scientificName"]
        + "), reassigned manually to "
        + matched["scientificNameValidated"]
    ).values
    result["scientificName"] = matched["scientificNameValidated"].values
    return stamp_human_review(result, fallback_classified_by, timestamp)


def build_confirmed(observations, decoded, fallback_classified_by, timestamp):
    """Observations manually validated as Positive for the model species."""
    model = decoded["scientificNameModel"]
    validated = decoded["scientificNameValidated"]
    positive = decoded[
        model.notna() & (decoded["validationResult"] == "Positive") & (validated == model)
    ]
    matched = match_observations(observations, positive, "positive")
    result = stamp_human_review(
        matched.drop(columns="scientificNameValidated"), fallback_classified_by, timestamp
    )
    result["classificationProbability"] = 1
    return result


def get_manually_rejected_ids(observations, decoded):
    """observationIDs validated as Negative and not reassigned."""
    model = decoded["scientificNameModel"]
    negative = decoded[
        model.notna()
        & (decoded["validationResult"] == "Negative")
        & decoded["scientificNameValidated"].isna()
    ]
    return set(match_observations(observations, negative, "negative")["observationID"])


def filter_observations(observations, thresholds, keep_unthresholded=False):
    """Keep only observations whose classificationProbability is >= the
    appliedCutoff for their species. Species absent from `thresholds` are
    dropped unless keep_unthresholded is set.
    """
    cutoffs = thresholds.set_index("scientificName")["appliedCutoff"]
    applied_cutoff = observations["scientificName"].map(cutoffs)

    unthresholded = applied_cutoff.isna()
    if unthresholded.any():
        species = sorted(observations.loc[unthresholded, "scientificName"].unique())
        action = "keeping" if keep_unthresholded else "dropping"
        logger.info(
            "%d species have no entry in the thresholds table (%s): %s",
            len(species),
            action,
            species,
        )
        if keep_unthresholded:
            applied_cutoff = applied_cutoff.fillna(-np.inf)

    keep = observations["classificationProbability"] >= applied_cutoff
    return observations[keep]


def apply_confusions(candidates, confusions):
    """Relabel candidate observations of consistently confused species."""
    pairs = confusions[np.isfinite(confusions["appliedConfusionCutoff"])]
    relabeled = []
    for pair in pairs.itertuples():
        rows = candidates[
            (candidates["scientificName"] == pair.modelSpecies)
            & (candidates["classificationProbability"] >= pair.appliedConfusionCutoff)
        ].copy()
        rows["observationComments"] = (
            f"Inferred reassignment: {pair.modelSpecies} detections at this confidence "
            f"were consistently {pair.confusedSpecies} in manual validation"
        )
        rows["observationTags"] = INFERRED_TAG
        rows["scientificName"] = pair.confusedSpecies
        relabeled.append(rows)
    if not relabeled:
        return candidates.iloc[0:0]
    return pd.concat(relabeled)


def build_manual_annotations(observations, decoded, fallback_classified_by, timestamp):
    """New observation rows for manual annotations (segments without a model species)."""
    manual = decoded[decoded["scientificNameModel"].isna()]
    media_stem = manual["mediaID"].str.removesuffix(".WAV")
    new = pd.DataFrame(
        {
            "deploymentID": media_stem.str.split("_").str[0].values,
            "mediaID": manual["mediaID"].values,
            "eventStart": manual["eventStart"].values,
            "eventEnd": manual["eventEnd"].values,
            "scientificName": manual["scientificNameValidated"].values,
        }
    )
    for column in REVIEW_COLUMNS.values():
        if column in manual.columns:
            new[column] = manual[column].values

    unknown = sorted(set(new["deploymentID"]) - set(observations["deploymentID"]))
    if unknown:
        logger.warning("deploymentIDs not found in observations: %s", unknown)

    existing = observations[["mediaID", "eventStart", "eventEnd", "scientificName"]]
    already = new.merge(existing, how="left", indicator=True)["_merge"].eq("both").values
    if already.any():
        logger.warning(
            "%d manual annotations already exist in observations, skipped", already.sum()
        )
    new = new[~already].reset_index(drop=True)

    new["observationID"] = observations["observationID"].max() + 1 + np.arange(len(new))
    new["observationLevel"] = "interval"
    new["observationType"] = "animal"
    new["classificationProbability"] = 1
    new["observationComments"] = "Manual annotation"
    new["observationTags"] = None
    new = stamp_human_review(new, fallback_classified_by, timestamp)
    return new.reindex(columns=observations.columns)
