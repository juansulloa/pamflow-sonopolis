from maad import sound
from maad.rois import find_rois_cwt
from glob import glob
import pandas as pd

path = '/Volumes/lacie_exfat/Sonopolis - PA - Timelapse/muestras varias/*.WAV'
files = glob(path)

#%%
df_detections_all = []

for file in files:
    print(f'Processing file: {file}')
    s, fs = sound.load(file)

    detections = find_rois_cwt(s, fs, flims=(500,10000), tlen=2, th=0.01, display=False)
    detections['file'] = file
    df_detections_all.append(detections)


df_detections_all = pd.concat(df_detections_all, ignore_index=True)
df_detections_all.to_csv('detections_in_silence.csv', index=False)