""" 
Perform inferences on audio files using PANNs (Pretrained Audio Neural Networks) for Sound Event Detection (SED).

Outputs one observations.csv per deploymentID, following a Camtrap/pamDP-style
long-format observation table (one row per label per time window).

Need to run on conda environment panns
conda activate panns

--------------------------------------------------------------------------------
How detections are constructed
--------------------------------------------------------------------------------
1. Each audio file is loaded at 32 kHz mono and passed through PANNs' CNN14
   (DecisionLevelMax checkpoint), which outputs a framewise prediction matrix:
   one confidence score per AudioSet class (527 total) for every ~10 ms frame.

2. Frames are grouped into non-overlapping windows of `segment_seconds`
   (default 10 s), matching the temporal resolution of AudioSet's original
   labels and giving a manageable number of rows per file.

3. Within each window, per-class confidence is aggregated across frames
   using `score_agg` ('mean' by default; 'max' also supported). The top_k
   classes by mean confidence are kept as detected labels for that window;
   all other classes are dropped for that window.

4. Each (window, label) pair becomes one row in the output table — so a
   10 s window with 4 detected labels produces 4 rows, each carrying the
   same eventStart/eventEnd but a different label and its own
   classificationProbability.

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
from panns_inference import SoundEventDetection, labels


PANNS_CHECKPOINT = "Cnn14_DecisionLevelMax"

def get_classified_by():
    """Build a reproducible identifier for the model + package version used."""
    pkg_version = get_panns_version()
    return f"panns_inference-{pkg_version}_{PANNS_CHECKPOINT}"


def segment_scores_to_records(frames, fps, total_duration, deployment_id, media_id,
                               segment_seconds=10, top_k=10, score_agg='mean'):
    """
    Process framewise SED output into a list of long-format observation records.

    One record per (segment, label) pair — i.e. if a 10 s window has 4 labels
    in its top_k, that produces 4 rows.

    score_agg: 'mean' (average confidence across the window) or 'max' (peak
    confidence within the window) for classificationProbability.
    """
    frames_per_segment = int(fps * segment_seconds)
    total_frames = frames.shape[0]
    records = []

    for i in range(0, total_frames, frames_per_segment):
        segment = frames[i:i + frames_per_segment]
        if len(segment) == 0:
            continue

        t_start = i / fps
        t_end = min((i + frames_per_segment) / fps, total_duration)

        mean_scores = np.mean(segment, axis=0)
        max_scores = np.max(segment, axis=0)

        ranking_scores = mean_scores  # top_k always chosen by mean confidence
        top_indices = np.argsort(ranking_scores)[::-1][:top_k]

        agg_scores = mean_scores if score_agg == 'mean' else max_scores

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
                "classifiedBy": get_classified_by(),
                "classificationProbability": round(float(agg_scores[idx]), 6),
            })

    return records


def process_audio_file(audio_path, deployment_id, media_id, segment_seconds=10,
                        top_k=10, score_agg='mean', device='cpu'):
    """
    Run SED on a single audio file and return a list of long-format observation records.
    """
    audio_path = Path(audio_path)
    sr = 32000

    # Load audio
    audio, _ = librosa.load(audio_path, sr=sr, mono=True)
    total_duration = len(audio) / sr
    audio_input = audio[None, :]

    # Run inference
    sed = SoundEventDetection(checkpoint_path=None, device=device)
    framewise_output = sed.inference(audio_input)
    frames = framewise_output[0]  # (total_frames, n_classes)

    total_frames = frames.shape[0]
    fps = total_frames / total_duration

    records = segment_scores_to_records(
        frames=frames,
        fps=fps,
        total_duration=total_duration,
        deployment_id=deployment_id,
        media_id=media_id,
        segment_seconds=segment_seconds,
        top_k=top_k,
        score_agg=score_agg,
    )

    return records


def main():
    # Variables
    save_path = '../../data/output/panns/'
    media_path = "../../data/output/data_preparation/media.csv"
    segment_seconds = 10
    top_k = 10
    score_agg = 'mean'  # or 'max'
    device = 'cpu'

    os.makedirs(save_path, exist_ok=True)

    # Load file list
    media = pd.read_csv(media_path)
    media = media[["filePath", "deploymentID", "mediaID"]].iloc[0:10]  # Limit to first 10 for testing

    all_records = []

    for _, row in media.iterrows():
        audio_path = row["filePath"]
        deployment_id = row["deploymentID"]
        media_id = row["mediaID"]

        print(f"Processing: {audio_path}")
        records = process_audio_file(
            audio_path=audio_path,
            deployment_id=deployment_id,
            media_id=media_id,
            segment_seconds=segment_seconds,
            top_k=top_k,
            score_agg=score_agg,
            device=device,
        )
        all_records.extend(records)

    if not all_records:
        print("No records generated.")
        return

    obs_df = pd.DataFrame(all_records)

    # Assign a globally unique observationID per row
    obs_df.insert(0, "observationID", range(1, len(obs_df) + 1))
    obs_df["observationID"] = obs_df["deploymentID"].astype(str) + "_" + obs_df["observationID"].astype(str).str.zfill(6)

    # Save one CSV per deploymentID, appending to existing files if present
    for deployment_id, group in obs_df.groupby("deploymentID"):
        output_path = os.path.join(save_path, f"{deployment_id}_observations.csv")

        if os.path.exists(output_path):
            existing = pd.read_csv(output_path)
            group = pd.concat([existing, group], ignore_index=True)

        group.to_csv(output_path, index=False)
        print(f"Saved: {output_path} ({len(group)} rows)")


if __name__ == "__main__":
    main()