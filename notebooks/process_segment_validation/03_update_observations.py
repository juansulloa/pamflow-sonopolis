#!/usr/bin/env python3
"""
Merge human validation labels (from segment_validation.csv, produced by
get_validation_status.py) into observations.csv and segments.csv.

Pipeline:
    1. scan directory        -> segment_validation.csv   (get_validation_status.py)
    2. this script            -> updated observations.csv + segments.csv

Join strategy
-------------
segments.csv already has a `segmentsFilePath` column built as:
    f"{classificationProbabilityRounded}_{mediaID.replace('.WAV','')}_{eventStart}_{eventEnd}.WAV"
so we join segment_validation.segmentName -> segments.segmentsFilePath directly
(case-insensitive, since directory files use .wav and the derivation uses .WAV).

observations.csv does NOT have that column, so we recompute it in-memory from
mediaID / eventStart / eventEnd / classificationProbabilityRounded using the
same formula, purely as a join key (it is never written back).

Where the label lands differs per file:
  - segments.csv: a `validationResult` column (Positive/Negative/NA), plus
    classificationMethod/classificationProbability/classifiedBy/
    classificationTimestamp columns. Any pre-existing `filePath` column is
    dropped from the output.
  - observations.csv: NO new validation column. Instead the label is folded
    into the `observationTags` column as a `validation:Positive` /
    `validation:Negative` key:value pair, alongside whatever tags are
    already there. classificationProbability is set to 1 for Positive and
    0 for Negative; classificationMethod is "human".

Only rows whose validation actually CHANGED get the classification metadata
stamped. Rows where the new value matches what's already stored are left
untouched, so a prior human/model label + its attribution isn't silently
overwritten by a re-run. NA rows (still unclassified) are skipped entirely.
"""

import argparse
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd

TAG_SEP = ","
KV_SEP = ":"


def normalize(s: str) -> str:
    """Lowercase + strip for case-insensitive filename matching."""
    return str(s).strip().lower()


def derive_segment_key(row, prob_col, media_col, start_col, end_col) -> str:
    media = str(row[media_col]).replace(".WAV", "").replace(".wav", "")
    return normalize(f"{row[prob_col]}_{media}_{row[start_col]}_{row[end_col]}.wav")


def backup(path: Path):
    bak = path.with_suffix(path.suffix + ".bak")
    shutil.copy2(path, bak)
    return bak


def normalize_timestamp(ts: str) -> str:
    """Normalize a user-supplied timestamp to YYYY-MM-DDTHH:MM:SS."""
    try:
        dt = datetime.fromisoformat(ts)
    except ValueError:
        raise SystemExit(
            f"Could not parse --classification-timestamp '{ts}'. "
            f"Please supply it as e.g. 2026-06-06T06:21:34 or 2026-06-06 06:21:34."
        )
    return dt.strftime("%Y-%m-%dT%H:%M:%S")


def parse_tags(tag_string) -> dict:
    """Parse a 'key:value,key:value' observationTags cell into a dict."""
    if pd.isna(tag_string) or str(tag_string).strip() == "":
        return {}
    tags = {}
    for part in str(tag_string).split(TAG_SEP):
        part = part.strip()
        if not part:
            continue
        if KV_SEP in part:
            k, v = part.split(KV_SEP, 1)
            tags[k.strip()] = v.strip()
        else:
            # tag with no value -- keep it as-is under its own key
            tags[part] = ""
    return tags


def format_tags(tags: dict) -> str:
    return TAG_SEP.join(f"{k}{KV_SEP}{v}" for k, v in tags.items())


def build_lookup(seg_val: pd.DataFrame) -> pd.Series:
    labeled = seg_val[seg_val["validation"] != "NA"].copy()
    labeled["_join_key"] = labeled["segmentName"].map(normalize)
    return labeled.set_index("_join_key")["validation"]


def update_segments(
    segments: pd.DataFrame,
    seg_val: pd.DataFrame,
    classified_by: str,
    timestamp: str,
    validation_col: str,
    method_col: str,
    probability_col: str,
    classified_by_col: str,
    timestamp_col: str,
) -> pd.DataFrame:
    df = segments.copy()
    df["_join_key"] = df["segmentsFilePath"].map(normalize)
    lookup = build_lookup(seg_val)

    for col in (validation_col, method_col, probability_col, classified_by_col, timestamp_col):
        if col not in df.columns:
            df[col] = pd.NA

    matched, changed, unmatched = 0, 0, []
    for key, new_val in lookup.items():
        mask = df["_join_key"] == key
        if not mask.any():
            unmatched.append(key)
            continue
        matched += mask.sum()

        current_val = df.loc[mask, validation_col]
        needs_update = current_val.isna() | (current_val != new_val)
        update_mask = mask & df.index.isin(current_val[needs_update].index)

        if update_mask.any():
            probability = 1 if new_val == "Positive" else 0 if new_val == "Negative" else pd.NA
            df.loc[update_mask, validation_col] = new_val
            df.loc[update_mask, method_col] = "human"
            df.loc[update_mask, probability_col] = probability
            df.loc[update_mask, classified_by_col] = classified_by
            df.loc[update_mask, timestamp_col] = timestamp
            changed += update_mask.sum()

    df = df.drop(columns=["_join_key"])
    if "filePath" in df.columns:
        df = df.drop(columns=["filePath"])

    print(f"  matched rows: {matched}, updated rows: {changed}, unmatched labels: {len(unmatched)}")
    if unmatched:
        print(f"  (first few unmatched keys: {unmatched[:5]})")
    return df


def update_observations(
    observations: pd.DataFrame,
    seg_val: pd.DataFrame,
    classified_by: str,
    timestamp: str,
    key_cols: dict,
    tags_col: str,
    method_col: str,
    probability_col: str,
    classified_by_col: str,
    timestamp_col: str,
) -> pd.DataFrame:
    df = observations.copy()
    df["_join_key"] = df.apply(derive_segment_key, axis=1, **key_cols)
    lookup = build_lookup(seg_val)

    if tags_col not in df.columns:
        df[tags_col] = pd.NA
    for col in (method_col, probability_col, classified_by_col, timestamp_col):
        if col not in df.columns:
            df[col] = pd.NA

    matched, changed, unmatched = 0, 0, []
    for key, new_val in lookup.items():
        mask = df["_join_key"] == key
        if not mask.any():
            unmatched.append(key)
            continue
        matched += mask.sum()

        for idx in df.index[mask]:
            current_tags = parse_tags(df.at[idx, tags_col])
            if current_tags.get("validation") == new_val:
                continue  # already up to date, don't touch attribution

            current_tags["validation"] = new_val
            df.at[idx, tags_col] = format_tags(current_tags)
            df.at[idx, method_col] = "human"
            df.at[idx, probability_col] = 1 if new_val == "Positive" else 0
            df.at[idx, classified_by_col] = classified_by
            df.at[idx, timestamp_col] = timestamp
            changed += 1

    df = df.drop(columns=["_join_key"])
    print(f"  matched rows: {matched}, updated rows: {changed}, unmatched labels: {len(unmatched)}")
    if unmatched:
        print(f"  (first few unmatched keys: {unmatched[:5]})")
    return df


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--segment-validation", required=True, help="Path to segment_validation.csv")
    p.add_argument("--observations", required=True, help="Path to observations.csv")
    p.add_argument("--segments", required=True, help="Path to segments.csv")
    p.add_argument("--classified-by", required=True, help="Name of the researcher")
    p.add_argument("--classification-timestamp", required=True,
                    help="Date/time of classification, e.g. 2026-06-06T06:21:34")

    # observations.csv key-derivation columns
    p.add_argument("--obs-media-col", default="mediaID")
    p.add_argument("--obs-start-col", default="eventStart")
    p.add_argument("--obs-end-col", default="eventEnd")
    p.add_argument("--obs-prob-col", default="classificationProbability")

    # observations.csv output columns
    p.add_argument("--obs-tags-col", default="observationTags")

    # segments.csv output columns
    p.add_argument("--segments-validation-col", default="validationResult")

    # shared output column names
    p.add_argument("--method-col", default="classificationMethod")
    p.add_argument("--probability-col", default="classificationProbability")
    p.add_argument("--classified-by-col", default="classifiedBy")
    p.add_argument("--timestamp-col", default="classificationTimestamp")

    p.add_argument("--in-place", action="store_true",
                    help="Overwrite observations.csv / segments.csv directly (a .bak backup is made first). "
                         "Without this flag, writes *_updated.csv next to the originals.")

    args = p.parse_args()
    timestamp = normalize_timestamp(args.classification_timestamp)

    # keep_default_na=False: pandas normally reads the literal string "NA" as
    # a missing value (NaN). Our "NA" is a real validation label, not a gap,
    # so we must stop pandas from silently converting it.
    seg_val = pd.read_csv(args.segment_validation, keep_default_na=False, na_values=[])
    obs_path = Path(args.observations)
    segs_path = Path(args.segments)
    observations = pd.read_csv(obs_path)
    segments = pd.read_csv(segs_path)

    print("Updating segments.csv ...")
    segments_updated = update_segments(
        segments, seg_val, args.classified_by, timestamp,
        args.segments_validation_col, args.method_col, args.probability_col,
        args.classified_by_col, args.timestamp_col,
    )

    print("Updating observations.csv ...")
    key_cols = dict(
        prob_col=args.obs_prob_col, media_col=args.obs_media_col,
        start_col=args.obs_start_col, end_col=args.obs_end_col,
    )
    observations_updated = update_observations(
        observations, seg_val, args.classified_by, timestamp, key_cols,
        args.obs_tags_col, args.method_col, args.probability_col,
        args.classified_by_col, args.timestamp_col,
    )

    if args.in_place:
        print(f"Backing up originals -> {backup(obs_path)}, {backup(segs_path)}")
        observations_updated.to_csv(obs_path, index=False)
        segments_updated.to_csv(segs_path, index=False)
        print(f"Overwrote {obs_path} and {segs_path}")
    else:
        obs_out = obs_path.with_name(obs_path.stem + "_updated.csv")
        segs_out = segs_path.with_name(segs_path.stem + "_updated.csv")
        observations_updated.to_csv(obs_out, index=False)
        segments_updated.to_csv(segs_out, index=False)
        print(f"Wrote {obs_out} and {segs_out}")


if __name__ == "__main__":
    main()