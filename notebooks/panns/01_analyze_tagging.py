""" 
Perform inferences on audio files using PANNs (Pretrained Audio Neural Networks) for Sound Event Detection (SED).

Outputs one observations.csv per deploymentID, following a Camtrap/pamDP-style
long-format observation table (one row per label per time window).

Need to run on conda environment panns
conda activate panns

--------------------------------------------------------------------------------
How detections are constructed
--------------------------------------------------------------------------------
1. Each audio file is loaded at 32 kHz mono and sliced into independent,
   non-overlapping chunks of `segment_seconds` (default 10 s), matching the
   clip length PANNs was originally trained on. The final partial chunk
   (if shorter than segment_seconds) is kept and processed as-is.

2. Each chunk is passed independently through PANNs' CNN14 clip-level
   AudioTagging model (Cnn14.pth), which outputs one confidence score per
   AudioSet class (527 total) for that chunk. Unlike framewise SED models
   (e.g. DecisionLevelMax), this matches the exact training/inference setup
   used to report PANNs' published AudioSet benchmarks, since normalization
   and prediction are computed per-clip rather than pooled from a
   longer file.

3. Within each chunk, the top_k classes by confidence are kept as detected
   labels; all other classes are dropped for that chunk.

4. Each (chunk, label) pair becomes one row in the output table — so a
   10 s chunk with 4 detected labels produces 4 rows, each carrying the
   same eventStart/eventEnd but a different label and its own
   classificationProbability (the model's raw confidence for that class
   on that chunk).

5. Rows from all files sharing the same deploymentID are combined and
   written to a single {deploymentID}_observations.csv, appending to any
   existing file for that deployment rather than overwriting it.
--------------------------------------------------------------------------------
"""

import os
from pathlib import Path
from importlib.metadata import version, PackageNotFoundError

import librosa
import numpy as np
import pandas as pd
from panns_inference import AudioTagging, labels

PANNS_CHECKPOINT = "Cnn14_mAP=0.431.pth"


def get_panns_version():
    """Return the installed panns_inference package version, if available."""
    try:
        return version("panns_inference")
    except PackageNotFoundError:
        return "panns_inference"


def get_classified_by():
    """Build a reproducible identifier for the model + package version used."""
    pkg_version = get_panns_version()
    return f"panns_inference-{pkg_version}_{PANNS_CHECKPOINT}"


def chunk_scores_to_records(audio, sr, at, deployment_id, media_id,
                             segment_seconds=10, top_k=10, batch_size=32):
    total_duration = len(audio) / sr
    chunk_len = int(segment_seconds * sr)

    # Build all chunks first, pad the last one if needed
    chunks = []
    times = []
    for i in range(0, len(audio), chunk_len):
        chunk = audio[i:i + chunk_len]
        if len(chunk) == 0:
            continue
        if len(chunk) < chunk_len:
            chunk = np.pad(chunk, (0, chunk_len - len(chunk)))
        chunks.append(chunk)
        t_start = i / sr
        t_end = min((i + chunk_len) / sr, total_duration)
        times.append((t_start, t_end))

    records = []
    # Run inference in batches instead of one chunk at a time
    for b in range(0, len(chunks), batch_size):
        batch = np.stack(chunks[b:b + batch_size])
        clipwise_output, _ = at.inference(batch)  # (batch_size, n_classes)

        for j, scores in enumerate(clipwise_output):
            t_start, t_end = times[b + j]
            top_indices = np.argsort(scores)[::-1][:top_k]
            for idx in top_indices:
                records.append({
                    "deploymentID": deployment_id,
                    "mediaID": media_id,
                    "eventStart": round(t_start, 1),
                    "eventEnd": round(t_end, 1),
                    "observationLevel": "soundscape",
                    "observationType": "interval",
                    "label": labels[idx],
                    "classificationMethod": "machine",
                    "classifiedBy": PANNS_CHECKPOINT,
                    "classificationProbability": round(float(scores[idx]), 6),
                })
    return records

def process_audio_file(audio_path, deployment_id, media_id, at, segment_seconds=10,
                        top_k=10):
    """
    Run clip-level PANNs tagging on 10 s chunks of a single audio file and
    return a list of long-format observation records.
    """
    audio_path = Path(audio_path)
    sr = 32000

    audio, _ = librosa.load(audio_path, sr=sr, mono=True)

    records = chunk_scores_to_records(
        audio=audio,
        sr=sr,
        at=at,
        deployment_id=deployment_id,
        media_id=media_id,
        segment_seconds=segment_seconds,
        top_k=top_k,
    )

    return records


def main():
    # Variables
    save_path = '../../data/output/panns/'
    media_path = "../../data/output/data_preparation/media.csv"
    segment_seconds = 10
    top_k = 10
    device = 'cpu'

    # --- Testing switch ---
    # Set to None to process all files. Set to an integer to randomly sample
    # k rows from media.csv for a quick functional test before committing to
    # the full 36k-file run.
    test_sample_k = None # e.g. 20
    random_seed = 42

    os.makedirs(save_path, exist_ok=True)

    # Load model once, reuse across files
    at = AudioTagging(checkpoint_path=None, device=device)

    # Load file list
    media = pd.read_csv(media_path)
    media = media[["filePath", "deploymentID", "mediaID"]]

    if test_sample_k is not None:
            media = (
                media.groupby("deploymentID", group_keys=False)
                .apply(lambda g: g.sample(n=min(test_sample_k, len(g)), random_state=random_seed))
            )
            print(f"TEST MODE: sampled up to {test_sample_k} rows per deployment "
                f"({len(media)} total rows across {media['deploymentID'].nunique()} deployments)")

    deployment_ids = media["deploymentID"].unique()
    print(f"Found {len(deployment_ids)} deployment(s) to process")

    for deployment_id in deployment_ids:
        output_path = os.path.join(save_path, f"{deployment_id}_observations.csv")
        deployment_media = media[media["deploymentID"] == deployment_id]

        print(f"\n--- Deployment: {deployment_id} ({len(deployment_media)} files) ---")

        deployment_records = []

        for _, row in deployment_media.iterrows():
            audio_path = row["filePath"]
            media_id = row["mediaID"]

            print(f"Processing: {audio_path}")
            records = process_audio_file(
                audio_path=audio_path,
                deployment_id=deployment_id,
                media_id=media_id,
                at=at,
                segment_seconds=segment_seconds,
                top_k=top_k,
            )
            deployment_records.extend(records)

        if not deployment_records:
            print(f"No records generated for deployment {deployment_id}.")
            continue

        new_df = pd.DataFrame(deployment_records)

        # Append to existing file for this deployment, if present
        if os.path.exists(output_path):
            existing = pd.read_csv(output_path)
            start_counter = len(existing) + 1
            combined = pd.concat([existing, new_df], ignore_index=True)
        else:
            existing = None
            start_counter = 1
            combined = new_df

        # Rebuild observationID for the newly added rows only, continuing
        # the counter from where the existing file left off
        n_new = len(new_df)
        new_ids = [
            f"{deployment_id}_{str(i).zfill(6)}"
            for i in range(start_counter, start_counter + n_new)
        ]

        if existing is not None:
            combined_ids = list(existing["observationID"]) + new_ids
        else:
            combined_ids = new_ids

        combined.insert(0, "observationID", combined_ids) if "observationID" not in combined.columns else combined.__setitem__("observationID", combined_ids)

        combined.to_csv(output_path, index=False)
        print(f"Saved: {output_path} ({len(combined)} total rows, +{n_new} new)")


if __name__ == "__main__":
    main()