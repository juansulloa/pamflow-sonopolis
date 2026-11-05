import numpy as np
import pandas as pd
import pytest

from pamflow.pipelines.detection_validation import nodes, utils

PARAMETERS = {
    "true_positive_rate": 0.9,
    "min_sample_size": 5,
    "threshold_method": "auto",
    "plot_thresholds": False,
    "keep_unthresholded": False,
    "apply_confusions": False,
    "fallback_classified_by": None,
    "classification_timestamp": "2026-01-01T00:00:00",
}


def raw_row(name, species, category, validated=None, by="Ana", favorite="false", comment=None):
    return {
        "scientificNameProposed": species,
        "validationCategory": category,
        "scientificNameValidated": validated,
        "classifiedBy": by,
        "favorite": favorite,
        "fileName": f"Genus_species\\Genus_species\\{name}",
        "validationComment": comment,
    }


def test_get_validation_status_maps_categories_and_concatenates_files():
    first = pd.DataFrame(
        [
            raw_row("0.500_PBH01_20260101_000000_3.0_6.0.wav", "Genus species", "positive"),
            raw_row("0.200_PBH01_20260101_000000_6.0_9.0.wav", "Genus species", "negative"),
        ]
    )
    second = pd.DataFrame(
        [
            raw_row(
                "0.300_PBH02_20260101_000000_3.0_6.0.wav",
                "Genus species",
                "reassigned",
                validated="Other_species",
                comment="check",
            ),
            raw_row("0.400_PBH02_20260101_000000_6.0_9.0.wav", "Genus species", "uncertain"),
        ]
    )
    table = nodes.get_validation_status({"a": lambda: first, "b": lambda: second})

    by_name = table.set_index("segmentName")
    assert by_name.loc["0.500_PBH01_20260101_000000_3.0_6.0.wav", "validationResult"] == "Positive"
    assert (
        by_name.loc["0.500_PBH01_20260101_000000_3.0_6.0.wav", "scientificNameValidated"]
        == "Genus species"
    )
    assert by_name.loc["0.200_PBH01_20260101_000000_6.0_9.0.wav", "validationResult"] == "Negative"
    reassigned = by_name.loc["0.300_PBH02_20260101_000000_3.0_6.0.wav"]
    assert reassigned["validationResult"] == "Reassigned"
    assert reassigned["scientificNameValidated"] == "Other species"
    assert reassigned["validationComment"] == "check"
    assert by_name.loc["0.400_PBH02_20260101_000000_6.0_9.0.wav", "validationResult"] == "Uncertain"
    assert by_name["classificationProbability"].tolist() == [0.5, 0.2, 0.3, 0.4]


def test_get_validation_status_rejects_duplicates_and_empty_input():
    row = raw_row("0.500_PBH01_20260101_000000_3.0_6.0.wav", "Genus species", "positive")
    frame = pd.DataFrame([row])
    with pytest.raises(ValueError, match="Duplicate segmentName"):
        nodes.get_validation_status({"a": lambda: frame, "b": lambda: frame})
    with pytest.raises(ValueError, match="No validation result files"):
        nodes.get_validation_status({})


def test_target_levels_validates_rate():
    assert utils.target_levels(0.9) == [0.9, 0.95]
    assert utils.target_levels(0.8) == [0.8, 0.95]
    with pytest.raises(ValueError):
        utils.target_levels(1.5)


def make_validation_table(n=30, seed=0):
    """Species A: positives at high scores. Species B: reassigned to C."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        score = round(float(rng.uniform(0.1, 0.99)), 3)
        positive = score > 0.5
        rows.append(
            {
                "segmentName": f"{score:.3f}_PBH01_20260101_{i:06d}_3.0_6.0.wav",
                "scientificNameModel": "Species a",
                "classificationProbability": score,
                "validationResult": "Positive" if positive else "Negative",
                "scientificNameValidated": "Species a" if positive else None,
                "classifiedBy": "Ana",
                "favorite": False,
                "validationComment": None,
            }
        )
    rows.append(
        {
            "segmentName": "0.600_PBH01_20260102_000000_3.0_6.0.wav",
            "scientificNameModel": "Species b",
            "classificationProbability": 0.6,
            "validationResult": "Reassigned",
            "scientificNameValidated": "Species c",
            "classifiedBy": "Ana",
            "favorite": False,
            "validationComment": None,
        }
    )
    return pd.DataFrame(rows)


@pytest.mark.parametrize("rate, label", [(0.9, "90"), (0.8, "80")])
def test_find_model_threshold_columns_follow_rate_and_min_sample_size(rate, label):
    table = make_validation_table()
    parameters = {**PARAMETERS, "true_positive_rate": rate}
    thresholds, confusions, plots = nodes.find_model_threshold(table, parameters)

    assert f"sweepCutoff{label}" in thresholds.columns
    assert f"cutoffConfidence{label}" in thresholds.columns
    assert plots == {}
    species_a = thresholds.set_index("scientificName").loc["Species a"]
    assert species_a["cutoffSource"] == "sweep"
    assert 0.4 <= species_a["appliedCutoff"] <= 0.99
    # Species b only has a reassigned segment: no validated segments, excluded
    assert "Species b" not in thresholds["scientificName"].tolist()
    assert confusions["confusedSpecies"].tolist() == ["Species c"]


def test_find_model_threshold_min_sample_size_excludes_species():
    table = make_validation_table(n=30)
    parameters = {**PARAMETERS, "min_sample_size": 1000}
    thresholds, _, _ = nodes.find_model_threshold(table, parameters)
    row = thresholds.set_index("scientificName").loc["Species a"]
    assert row["cutoffSource"] == "exclude"
    assert np.isinf(row["appliedCutoff"])


def make_observations():
    rows = []
    specs = [
        # id, media, start, end, species, probability
        (0, "PBH01_20260101_000000.WAV", 3.0, 6.0, "Species a", 0.9),  # positive
        (1, "PBH01_20260101_000000.WAV", 6.0, 9.0, "Species a", 0.95),  # negative -> dropped
        (2, "PBH01_20260101_000000.WAV", 9.0, 12.0, "Species a", 0.97),  # kept by threshold
        (3, "PBH01_20260101_000000.WAV", 12.0, 15.0, "Species a", 0.2),  # below cutoff
        (4, "PBH01_20260101_000000.WAV", 15.0, 18.0, "Species b", 0.6),  # reassigned
        (5, "PBH01_20260101_000000.WAV", 18.0, 21.0, "Species d", 0.9),  # no threshold
    ]
    for observation_id, media, start, end, species, probability in specs:
        rows.append(
            {
                "observationID": str(observation_id),
                "deploymentID": "PBH01",
                "mediaID": media,
                "eventID": None,
                "eventStart": np.float32(start),
                "eventEnd": np.float32(end),
                "frequencyLow": None,
                "frequencyHigh": None,
                "observationLevel": "interval",
                "observationType": "animal",
                "scientificName": species,
                "count": None,
                "lifeStage": None,
                "sex": None,
                "behavior": None,
                "individualID": None,
                "individualPositionRadius": None,
                "classificationMethod": "machine",
                "classifiedBy": "Birdnet 2.4",
                "classificationTimestamp": "2026-01-01T00:00:00",
                "classificationProbability": np.float32(probability),
                "observationTags": None,
                "observationComments": None,
            }
        )
    return pd.DataFrame(rows)


def make_segment_validations():
    def row(name, model, result, validated, by="Ana", favorite=False, comment=None):
        return {
            "segmentName": name,
            "scientificNameModel": model,
            "classificationProbability": np.nan,
            "validationResult": result,
            "scientificNameValidated": validated,
            "classifiedBy": by,
            "favorite": favorite,
            "validationComment": comment,
        }

    return pd.DataFrame(
        [
            row("0.900_PBH01_20260101_000000_3.0_6.0.wav", "Species a", "Positive",
                "Species a", favorite=True, comment="clear call"),
            row("0.950_PBH01_20260101_000000_6.0_9.0.wav", "Species a", "Negative", None),
            row("0.600_PBH01_20260101_000000_15.0_18.0.wav", "Species b", "Reassigned",
                "Species c", by="Beto", comment="not b"),
        ]
    )


def test_update_observations_applies_validations_and_thresholds():
    thresholds = pd.DataFrame(
        {"scientificName": ["Species a"], "appliedCutoff": [0.5]}
    )
    confusions = pd.DataFrame(columns=utils.CONFUSION_COLUMNS)
    result = nodes.update_observations(
        make_observations(), thresholds, confusions, make_segment_validations(), PARAMETERS
    ).set_index("observationID")

    # 1 negative dropped, 3 below cutoff dropped, 5 has no threshold -> dropped
    assert sorted(result.index) == [0, 2, 4]

    confirmed = result.loc[0]
    assert confirmed["classificationMethod"] == "human"
    assert confirmed["classifiedBy"] == "Ana"
    assert confirmed["classificationProbability"] == 1
    assert confirmed["observationTags"] == "favorite"
    assert confirmed["observationComments"] == "clear call"

    machine = result.loc[2]
    assert machine["classificationMethod"] == "machine"
    assert machine["classifiedBy"] == "Birdnet 2.4"

    reassigned = result.loc[4]
    assert reassigned["scientificName"] == "Species c"
    assert reassigned["classifiedBy"] == "Beto"
    assert reassigned["observationComments"] == (
        "Model false positive (Species b), reassigned manually to Species c | not b"
    )


def test_update_observations_keep_unthresholded_and_fallback_reviewer():
    thresholds = pd.DataFrame(
        {"scientificName": ["Species a"], "appliedCutoff": [0.5]}
    )
    confusions = pd.DataFrame(columns=utils.CONFUSION_COLUMNS)
    validations = make_segment_validations().drop(columns="classifiedBy")
    parameters = {
        **PARAMETERS,
        "keep_unthresholded": True,
        "fallback_classified_by": "Fallback",
    }
    result = nodes.update_observations(
        make_observations(), thresholds, confusions, validations, parameters
    ).set_index("observationID")

    assert 5 in result.index
    assert result.loc[0, "classifiedBy"] == "Fallback"
    assert result.loc[4, "classifiedBy"] == "Fallback"


def test_update_observations_blank_reviewer_without_fallback():
    thresholds = pd.DataFrame({"scientificName": ["Species a"], "appliedCutoff": [0.5]})
    confusions = pd.DataFrame(columns=utils.CONFUSION_COLUMNS)
    validations = make_segment_validations().drop(columns="classifiedBy")
    result = nodes.update_observations(
        make_observations(), thresholds, confusions, validations, PARAMETERS
    ).set_index("observationID")
    assert pd.isna(result.loc[0, "classifiedBy"])


def test_update_observations_apply_confusions_relabels_inferred():
    thresholds = pd.DataFrame(
        {"scientificName": ["Species a"], "appliedCutoff": [0.5]}
    )
    confusions = pd.DataFrame(
        {
            "modelSpecies": ["Species d"],
            "confusedSpecies": ["Species e"],
            "appliedConfusionCutoff": [0.8],
        }
    )
    parameters = {**PARAMETERS, "apply_confusions": True}
    result = nodes.update_observations(
        make_observations(), thresholds, confusions, make_segment_validations(), parameters
    ).set_index("observationID")
    assert result.loc[5, "scientificName"] == "Species e"
    assert result.loc[5, "observationTags"] == "inferred_reassignment"
