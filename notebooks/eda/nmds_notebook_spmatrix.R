# COMPUTE NMDS AND PLOT RESULT
library(vegan)
library(ade4)
library(RColorBrewer)
library(readxl)
library(dplyr)
library(tidyr)
library(ggplot2)
library(httpgd)
options(device = 'httpgd')  # or use: hgd()
setwd("/Users/jsulloa/Library/CloudStorage/GoogleDrive-julloa@humboldt.org.co/Shared drives/Sonópolis/02_Técnico/01_Monitoreo Acústico/Análisis Periodo Muestreo A/notebooks")

# species matrix from birdnet
obs <- read.csv("../species_detection/observations.csv", sep=",")
deployments <- read.csv("../data_preparation/deployments.csv") %>%
  mutate(location = sub("-.*", "", locationID))

# Keep only confident detections (adjust threshold as needed)
obs_filtered <- obs %>%
  filter(classificationProbability >= 0.1,
         !is.na(scientificName),
         !is.na(deploymentID))

# Join to filtered observations
obs_filtered <- obs_filtered %>%
  left_join(deployments %>% select(deploymentID, location), by = 'deploymentID')

# ── 2. Build species matrix (deployments × species) ───────────────────────────
# Using detection frequency (count of detections per species per deployment)
# Alternative: swap n() for mean(classificationProbability) as abundance proxy
sp_matrix <- obs_filtered %>%
  group_by(deploymentID, scientificName) %>%
  summarise(n_detections = n(), .groups = 'drop') %>%
  pivot_wider(names_from = scientificName,
              values_from = n_detections,
              values_fill = 0)

# Separate metadata and numeric matrix
deploy_ids <- sp_matrix$deploymentID
sp_mat_num <- as.matrix(sp_matrix[, -1])
rownames(sp_mat_num) <- deploy_ids

# ── 3. Remove empty rows/cols (deployments or species with zero detections) ───
sp_mat_num <- sp_mat_num[rowSums(sp_mat_num) > 0, colSums(sp_mat_num) > 0]

# ── 4. NMDS ───────────────────────────────────────────────────────────────────
set.seed(42)
nmds_result <- metaMDS(sp_mat_num, distance = 'bray', trymax = 500, k = 2)
cat('Stress:', nmds_result$stress, '\n')  # < 0.2 acceptable, < 0.1 good

# ── 5. Plot ───────────────────────────────────────────────────────────────────
nmds_scores <- as.data.frame(vegan::scores(nmds_result, display = 'sites'))
nmds_scores$deploymentID <- rownames(nmds_scores)

# Join location info to scores
nmds_scores <- nmds_scores %>%
  left_join(deployments %>% select(deploymentID, locationID, location), by = 'deploymentID')

ggplot(nmds_scores, aes(x = NMDS1, y = NMDS2, color = location, label = locationID)) +
  geom_point(size = 3, alpha = 0.8) +
  geom_text(vjust = -0.8, size = 2.8, show.legend = FALSE) +
  annotate('text', x = Inf, y = -Inf,
           label = paste('Stress:', round(nmds_result$stress, 3)),
           hjust = 1.1, vjust = -0.5, size = 3.5, color = 'gray40') +
  theme_bw() +
  labs(title = 'NMDS – Species composition by deployment',
       subtitle = 'Bray-Curtis dissimilarity | BirdNET detections ≥ 0.8',
       x = 'NMDS1', y = 'NMDS2',
       color = 'Location')
ggsave('nmds_plot.png', width = 8, height = 6, dpi = 150)


# ── Indicator species analysis for location EN ────────────────────────────────
library(labdsv)
library(tibble)
# Create numeric group vector (labdsv requires numeric, not character)
groups <- ifelse(nmds_scores$location == "EN", 1, 2)

# Make sure row order matches sp_mat_num
groups <- groups[match(rownames(sp_mat_num), nmds_scores$deploymentID)]

# Run indicator species analysis
indval_result <- indval(sp_mat_num, groups)

# Extract results as dataframe
indval_df <- as.data.frame(indval_result$indval) %>%
  rownames_to_column('scientificName') %>%
  rename(indval_EN = `1`, indval_other = `2`) %>%
  mutate(pvalue = indval_result$pval) %>%
  filter(pvalue <= 0.05) %>%
  arrange(desc(indval_EN))

print(indval_df)
# Save indicator species results
write.csv(indval_df, 'indicator_species_EN.csv', row.names = FALSE)