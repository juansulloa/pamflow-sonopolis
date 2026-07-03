import librosa
import numpy as np
import pandas as pd
from pathlib import Path
from panns_inference import SoundEventDetection, labels
import os

def segment_scores_to_dataframe(frames, fps, total_duration, segment_seconds=10, top_k=10):
    """
    Process framewise SED output into a DataFrame with one column per segment.
    
    Returns a DataFrame where:
    - Index: AudioSet class names (527 classes)
    - Columns: segment time ranges (e.g. '0.0-10.0s')
    - Values: mean confidence score (0 if not in top_k)
    """
    frames_per_segment = int(fps * segment_seconds)
    total_frames = frames.shape[0]
    all_segments = {}

    for i in range(0, total_frames, frames_per_segment):
        segment = frames[i:i + frames_per_segment]
        if len(segment) == 0:
            continue

        t_start = i / fps
        t_end = min((i + frames_per_segment) / fps, total_duration)
        col_name = f"{t_start:.1f}-{t_end:.1f}s"

        mean_scores = np.mean(segment, axis=0)
        top_indices = np.argsort(mean_scores)[::-1][:top_k]

        # Build full column: 0 for all, then fill top_k
        col_scores = np.zeros(len(labels))
        col_scores[top_indices] = mean_scores[top_indices]
        all_segments[col_name] = col_scores

    df = pd.DataFrame(all_segments, index=labels)
    return df


def process_audio_file(audio_path, segment_seconds=10, top_k=10, device='cpu'):
    """
    Run SED on a single audio file and save results as CSV.
    Output CSV has same name as audio file, saved in same directory.
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

    # Build DataFrame
    df = segment_scores_to_dataframe(
        frames=frames,
        fps=fps,
        total_duration=total_duration,
        segment_seconds=segment_seconds,
        top_k=top_k
    )

    return df


# --- Run ---
# Variables
save_path = '../inferencias/'
ecos_urbanos_data_path = "../ecos_urbanos_data/ecos_urbanos_merged_data.csv"
path_audios = "/Users/jsulloa/Dropbox/PostDoc/iavh/2020_Ecos Urbanos/data_upload_gbif/PSVentana"

# Load file list and keep only files from Bogotá
audio_path_list = pd.read_csv(ecos_urbanos_data_path)
audio_path_list['file_name'] = audio_path_list['file_name'].apply(lambda x: os.path.join(path_audios, x))
#audio_path_list = audio_path_list[audio_path_list['depto'] == 'BOGOTÁ, D.C.']['file_name'].tolist()[0:50]  # Keep only first 10 for testing
audio_path_list = ['/Users/jsulloa/Dropbox/PostDoc/iavh/2020_Ecos Urbanos/data_upload_gbif/PSVentana/H0035_jseb.ulloa_0404-0612.wav']

for audio_path in audio_path_list:
    df = process_audio_file(audio_path, segment_seconds=10, top_k=10, device='cpu')

    # Save CSV
    output_path = os.path.join(save_path, os.path.basename(audio_path).replace('.wav', '.csv'))
    df.to_csv(output_path)
    print(f"Saved: {output_path}")
