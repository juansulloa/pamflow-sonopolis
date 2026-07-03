import pandas as pd

seg = pd.read_csv("../../species_detection/segments.csv")

seg.scientificName.value_counts().to_csv('segments_species_list.csv')