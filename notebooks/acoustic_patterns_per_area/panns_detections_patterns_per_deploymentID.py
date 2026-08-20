"""
Heatmap de patrones acusticos diarios por punto de muestreo (deploymentID),
para las etiquetas PANNs dominantes de biofonia, antropofonia y geofonia.

Para cada deployment, grafica 3 subplots apilados (uno por biofonia,
antropofonia y geofonia), cada uno un heatmap etiqueta x hora del dia (0-23)
con la intensidad de color = suma de classificationProbability de esa
etiqueta en esa hora (no un conteo de eventos discretos: cada fila de
{deploymentID}_observations.csv es una prediccion top-k de PANNs por
ventana de 10s con una probabilidad continua). Las etiquetas de cada subplot
se restringen a las top_biophony_labels / top_anthropophony_labels /
top_geophony_labels de soundscape_comparison_table.csv, ordenadas de mayor a
menor intensidad total. El titulo de cada subplot indica la categoria y su
porcentaje (pct_biophony/pct_anthropophony/pct_geophony). Al lado derecho de
cada subplot, un eje Y secundario muestra la intensidad total por etiqueta.

Input:
    ../../data/output/data_preparation/media.csv           (mediaID, timestamp)
    ../../data/output/data_preparation/deployments.csv     (deploymentID, locationID)
    ../../data/output/panns/soundscape_comparison_table.csv
        (deploymentID, pct_biophony, pct_anthropophony, pct_geophony,
         top_biophony_labels, top_anthropophony_labels, top_geophony_labels)
    ../../data/output/panns/{deploymentID}_observations.csv
        (deploymentID, mediaID, label, classificationProbability)

    La hora del dia se deriva del `timestamp` de media.csv (hora local, ISO8601
    con offset), no de `eventStart`/`eventEnd` de observations (esos son
    offsets en segundos dentro del archivo de audio, no hora absoluta).

Output:
    Un archivo acoustic_patterns_panns_<deploymentID>.png por cada deployment
    listado en soundscape_comparison_table.csv, guardado en esta misma carpeta.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

HOURS = list(range(24))
CATEGORY_ORDER = ["biophony", "anthropophony", "geophony"]
CATEGORY_LABELS = {"biophony": "Biophony", "anthropophony": "Anthropophony", "geophony": "Geophony"}


def load_data(media_path, deployments_path, comparison_table_path):
    """Carga media (hora), deployments (locationID) y la tabla comparativa de PANNs."""
    media = pd.read_csv(media_path, usecols=["mediaID", "timestamp"])
    deployments = pd.read_csv(deployments_path, usecols=["deploymentID", "locationID"])
    comparison_table = pd.read_csv(comparison_table_path)

    return media, deployments, comparison_table


def load_observations(observations_dir, deployment_id):
    """Carga las observaciones crudas (sin filtrar) de un deployment, o None si no existen."""
    observations_path = Path(observations_dir) / f"{deployment_id}_observations.csv"
    if not observations_path.exists():
        return None

    return pd.read_csv(observations_path, usecols=["mediaID", "label", "classificationProbability"])


def parse_label_list(labels_str, known_labels):
    """Separa un string de etiquetas unidas por ', ' usando el vocabulario de etiquetas conocidas.

    Algunas etiquetas de AudioSet contienen comas en su propio nombre (p.ej.
    "Bird vocalization, bird call, bird song" o "Chirp, tweet"), asi que no se
    puede separar de forma segura solo con str.split(","). En su lugar se
    hace un match greedy contra las etiquetas realmente presentes en las
    observaciones del deployment (mas largas primero).
    """
    if pd.isna(labels_str) or not str(labels_str).strip():
        return []

    remaining = str(labels_str).strip()
    known_sorted = sorted(known_labels, key=len, reverse=True)
    matched = []
    while remaining:
        for label in known_sorted:
            if remaining == label:
                matched.append(label)
                remaining = ""
                break
            if remaining.startswith(label + ", "):
                matched.append(label)
                remaining = remaining[len(label) + 2 :]
                break
        else:
            # Ninguna etiqueta conocida calza con el inicio: separa en la primera coma (best effort).
            head, _, remaining = remaining.partition(", ")
            matched.append(head)

    return matched


def get_deployment_labels(comparison_row, known_labels):
    """Extrae el mapeo etiqueta->categoria y los porcentajes de una fila de la tabla comparativa."""
    label_to_category = {}
    for category in CATEGORY_ORDER:
        labels_str = comparison_row.get(f"top_{category}_labels", "")
        for label in parse_label_list(labels_str, known_labels):
            label_to_category[label] = category

    pct = {category: comparison_row.get(f"pct_{category}", float("nan")) for category in CATEGORY_ORDER}

    return label_to_category, pct


def filter_and_localize(obs, label_to_category, media):
    """Filtra las observaciones a las etiquetas dominantes y agrega categoria + hora del dia."""
    obs = obs[obs["label"].isin(label_to_category)].copy()
    if obs.empty:
        return obs.assign(category=[], hour=[])

    obs["category"] = obs["label"].map(label_to_category)
    obs = obs.merge(media, on="mediaID", how="left")
    obs["hour"] = pd.to_datetime(obs["timestamp"]).dt.hour

    return obs


def build_pivots(df_deployment, label_to_category):
    """Construye, por categoria, la tabla etiqueta x hora (suma de classificationProbability).

    Dentro de cada categoria, las etiquetas quedan ordenadas de mayor a menor
    intensidad total. Devuelve un dict category -> (pivot, totals); las
    categorias sin datos quedan con un pivot/totals vacios.
    """
    pivots = {}
    for category in CATEGORY_ORDER:
        category_labels = [label for label, cat in label_to_category.items() if cat == category]
        df_category = df_deployment[df_deployment["category"] == category]

        if not category_labels or df_category.empty:
            pivots[category] = (pd.DataFrame(columns=HOURS), pd.Series(dtype=float))
            continue

        pivot = df_category.pivot_table(
            index="label", columns="hour", values="classificationProbability", aggfunc="sum", fill_value=0
        )
        pivot = pivot.reindex(columns=HOURS, fill_value=0)
        pivot = pivot.reindex(index=category_labels, fill_value=0)

        totals = pivot.sum(axis=1).sort_values(ascending=False)
        pivot = pivot.loc[totals.index]

        pivots[category] = (pivot, totals)

    return pivots


def plot_heatmap_grid(pivots, pct, title, output_path):
    """Grafica 3 subplots apilados (biofonia, antropofonia, geofonia), cada uno con su eje Y derecho de intensidad total."""
    row_counts = [max(len(pivots[category][0]), 1) for category in CATEGORY_ORDER]
    fig_height = sum(max(1.8, 0.45 * count) for count in row_counts) + 1.0

    fig, axes = plt.subplots(
        len(CATEGORY_ORDER), 1, figsize=(10, fig_height), height_ratios=row_counts
    )

    for ax, category in zip(axes, CATEGORY_ORDER):
        pivot, totals = pivots[category]
        subplot_title = f"{CATEGORY_LABELS[category]} — {pct[category]:.2f}%"

        if pivot.empty:
            ax.set_title(subplot_title)
            ax.set_yticks([])
            ax.set_xticks([])
            ax.text(0.5, 0.5, "Sin datos", ha="center", va="center", transform=ax.transAxes)
            continue

        sns.heatmap(
            pivot,
            ax=ax,
            cmap="viridis",
            cbar=False,
            linewidths=0.3,
            linecolor="white",
        )
        ax.set_ylabel("")
        ax.set_xlabel("")
        ax.set_title(subplot_title)

        ax_totals = ax.twinx()
        ax_totals.set_ylim(ax.get_ylim())
        ax_totals.set_yticks(ax.get_yticks())
        ax_totals.set_yticklabels([f"{v:.1f}" for v in totals.values])
        ax_totals.set_ylabel("Σ prob.", rotation=270, labelpad=12)
        sns.despine(ax=ax_totals, left=True, right=True, bottom=True, top=True)

    axes[-1].set_xlabel("Hora del día")
    fig.suptitle(title)
    fig.tight_layout()

    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)

    return output_path


if __name__ == "__main__":
    # --- Rutas ---
    MEDIA_PATH = "../../data/output/data_preparation/media.csv"
    DEPLOYMENTS_PATH = "../../data/output/data_preparation/deployments.csv"
    COMPARISON_TABLE_PATH = "../../data/output/panns/soundscape_comparison_table.csv"
    OBSERVATIONS_DIR = "../../data/output/panns/"
    OUTPUT_DIR = "../../data/output/acoustic_patterns_per_area/plots_panns_categories/"

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    media, deployments, comparison_table = load_data(MEDIA_PATH, DEPLOYMENTS_PATH, COMPARISON_TABLE_PATH)

    for _, comparison_row in comparison_table.iterrows():
        deployment_id = comparison_row["deploymentID"]

        raw_obs = load_observations(OBSERVATIONS_DIR, deployment_id)
        if raw_obs is None or raw_obs.empty:
            print(f"Omitido: {deployment_id} (sin observaciones)")
            continue

        known_labels = set(raw_obs["label"].unique())
        label_to_category, pct = get_deployment_labels(comparison_row, known_labels)
        if not label_to_category:
            print(f"Omitido: {deployment_id} (sin etiquetas dominantes)")
            continue

        df_deployment = filter_and_localize(raw_obs, label_to_category, media)
        if df_deployment.empty:
            print(f"Omitido: {deployment_id} (sin observaciones tras filtrar)")
            continue

        match = deployments.loc[deployments["deploymentID"] == deployment_id, "locationID"]
        location_id = match.iloc[0] if len(match) else None
        has_location_id = not pd.isna(location_id)
        title = f"{deployment_id}" if not has_location_id else f"{deployment_id} — {location_id}"
        file_stem = deployment_id if not has_location_id else f"{location_id}-{deployment_id}"

        pivots = build_pivots(df_deployment, label_to_category)
        output_path = Path(OUTPUT_DIR) / f"acoustic_patterns_panns_{file_stem}.png"
        plot_heatmap_grid(pivots, pct, title, output_path)

        n_labels = sum(len(pivot) for pivot, _ in pivots.values())
        total_intensity = sum(totals.sum() for _, totals in pivots.values())
        print(f"Guardado: {output_path} ({n_labels} etiquetas, intensidad total {total_intensity:.1f})")
