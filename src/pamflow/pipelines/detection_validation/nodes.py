"""Nodes of the detection_validation pipeline.

Turn manual validations of detected segments into per-species confidence
thresholds and a final, validated observations table.
"""

import logging
from datetime import datetime
from typing import Callable, Dict, Tuple

import numpy as np
import pandas as pd

from . import utils

logger = logging.getLogger(__name__)


def get_validation_status(
    validation_results: Dict[str, Callable[[], pd.DataFrame]],
) -> pd.DataFrame:
    """Build the segment validation table from the annotation CSV files.

    Parameters
    ----------
    validation_results : dict
        Catalog entry `validation_results@PartitionedDataset`: one loader per
        CSV file. Each file has one row per validated segment, with columns
        scientificNameProposed, validationCategory, scientificNameValidated,
        classifiedBy, favorite, fileName and validationComment. All files are
        concatenated.

    Returns
    -------
    pandas.DataFrame
        One row per segment with columns segmentName, scientificNameModel,
        classificationProbability, validationResult, scientificNameValidated,
        classifiedBy, favorite and validationComment. validationResult is one
        of Positive, Negative, Reassigned, Uncertain or NA. Saved as
        `segment_validation_results@pandas`.
    """
    if not validation_results:
        raise ValueError("No validation result files found in the validation_results folder")

    frames = []
    for name, load in sorted(validation_results.items()):
        logger.info("Reading validation results: %s", name)
        frames.append(load())
    raw = pd.concat(frames, ignore_index=True)

    table = utils.build_validation_table(raw)
    utils.log_validation_summary(table)
    return table


def find_model_threshold(
    segment_validation_results: pd.DataFrame, parameters: dict
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    """Compute the confidence cutoff of every species and the species confusions.

    Manual annotations and reassigned segments are excluded from the
    per-species fits. Reassigned segments feed the confusion table instead.

    Parameters
    ----------
    segment_validation_results : pandas.DataFrame
        Output of `get_validation_status`.
    parameters : dict
        `detection_validation_parameters`. Used keys: true_positive_rate,
        min_sample_size, threshold_method ('auto', 'model' or 'sweep') and
        plot_thresholds.

    Returns
    -------
    tuple
        (thresholds, confusions, plots). `thresholds` has one row per species
        with the diagnostics and the appliedCutoff / cutoffSource columns.
        `confusions` has one row per (model species, confused species) pair.
        `plots` maps a file name to a matplotlib figure, and is empty unless
        plot_thresholds is true.
    """
    primary_rate = parameters["true_positive_rate"]
    min_sample_size = parameters["min_sample_size"]
    method = parameters["threshold_method"]
    levels = utils.target_levels(primary_rate)

    validation_results, reassigned, n_manual = utils.split_segments(segment_validation_results)
    logger.info(
        "Excluded from threshold fits: %d manual annotations, %d reassigned segments",
        n_manual,
        len(reassigned),
    )

    plots = {}
    summary_rows = []
    for species in sorted(validation_results["scientificNameModel"].dropna().unique()):
        data = utils.load_data(validation_results, species)
        row, null_model, conf_model = utils.summarize_species(
            species, data, method, primary_rate, levels, min_sample_size
        )
        summary_rows.append(row)

        if conf_model is not None and parameters["plot_thresholds"]:
            prediction_range = np.arange(0, 1.001, 0.001)
            predictions = conf_model.predict(
                pd.DataFrame({"classificationProbability": prediction_range})
            )
            figure = utils.plot_results(
                species,
                data,
                prediction_range,
                predictions,
                row[f"cutoffConfidence{utils.pct_label(primary_rate)}"],
                primary_rate,
                utils.create_aic_table(null_model, conf_model),
            )
            plots[f"{species.replace(' ', '_')}_confidence_score"] = figure

    thresholds = pd.DataFrame(summary_rows).round(3)
    logger.info("Computed thresholds for %d species", len(thresholds))

    confusions = utils.compute_confusions(
        segment_validation_results[segment_validation_results["scientificNameModel"].notna()],
        thresholds,
        primary_rate,
        levels,
        min_sample_size,
    ).round(3)
    logger.info("Found %d species confusion pairs", len(confusions))

    return thresholds, confusions, plots


def update_observations(
    observations: pd.DataFrame,
    thresholds: pd.DataFrame,
    confusions: pd.DataFrame,
    segment_validation_results: pd.DataFrame,
    parameters: dict,
) -> pd.DataFrame:
    """Build the final observations table from machine observations, species
    thresholds and manual validations.

    The output contains, in order of precedence (an observation is used once):
        1. Reassigned: observations whose segment was reassigned to another
           species (relabeled and commented).
        2. Confirmed: observations whose segment was validated as Positive.
        3. Thresholded: remaining machine observations at or above the
           appliedCutoff of their species. Observations validated as Negative
           are dropped even if they meet the cutoff.
        4. Inferred reassignments (apply_confusions): remaining observations of
           a consistently confused species, relabeled and tagged.
        5. Manual annotations: segments without a model species.

    Parameters
    ----------
    observations : pandas.DataFrame
        Catalog entry `observations@pamDP`.
    thresholds : pandas.DataFrame
        Output of `find_model_threshold`.
    confusions : pandas.DataFrame
        Species confusion table, used only if apply_confusions is true.
    segment_validation_results : pandas.DataFrame
        Output of `get_validation_status`.
    parameters : dict
        `detection_validation_parameters`. Used keys: keep_unthresholded,
        apply_confusions, fallback_classified_by and classification_timestamp
        (null means the run time).

    Returns
    -------
    pandas.DataFrame
        Observations in the pamDP observations columns. Human-reviewed rows
        carry the annotator in classifiedBy, comments in observationComments
        and 'favorite' in observationTags.
    """
    fallback = parameters["fallback_classified_by"]
    timestamp = parameters["classification_timestamp"] or datetime.now().isoformat(
        timespec="seconds"
    )

    observations = utils.normalize_observations(observations)
    decoded = utils.decode_segments(segment_validation_results)

    reassigned = utils.build_reassigned(observations, decoded, fallback, timestamp)
    confirmed = utils.build_confirmed(observations, decoded, fallback, timestamp)
    rejected_ids = utils.get_manually_rejected_ids(observations, decoded)

    # Source rows already handled by manual validation never go through the threshold
    handled_ids = set(reassigned["observationID"]) | set(confirmed["observationID"]) | rejected_ids
    remaining = observations[~observations["observationID"].isin(handled_ids)]

    thresholded = utils.filter_observations(
        remaining, thresholds, parameters["keep_unthresholded"]
    )

    inferred = observations.iloc[0:0]
    if parameters["apply_confusions"]:
        candidates = remaining[~remaining["observationID"].isin(thresholded["observationID"])]
        inferred = utils.apply_confusions(candidates, confusions)

    manual = utils.build_manual_annotations(observations, decoded, fallback, timestamp)

    combined = (
        pd.concat([reassigned, confirmed, thresholded, inferred, manual])
        .drop_duplicates(subset="observationID")
        .sort_values("observationID")
    )

    logger.info(
        "Observations: %d -> %d total (thresholded machine: %d, confirmed: %d, "
        "reassigned: %d, inferred reassignments: %d, manual annotations: %d, "
        "dropped as Negative: %d)",
        len(observations),
        len(combined),
        len(thresholded),
        len(confirmed),
        len(reassigned),
        len(inferred),
        len(manual),
        len(rejected_ids),
    )
    return combined
