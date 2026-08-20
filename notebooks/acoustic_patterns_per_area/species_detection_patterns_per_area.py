"""
Heatmap de patrones acusticos diarios por punto de muestreo (deploymentID).

Para cada deployment, grafica un heatmap especie x hora del dia (0-23) con la
intensidad de color = numero de registros de deteccion (filas) de esa especie
en esa hora, ordenando las especies de mayor a menor numero total de
detecciones. Al lado derecho del heatmap, un eje Y secundario muestra el
total de registros por especie.

Input:
    ../../data/output/data_preparation/media.csv           (mediaID, timestamp)
    ../../data/output/data_preparation/deployments.csv     (deploymentID, locationName)
    ../../data/output/validation/observations_thresholded.csv
        (deploymentID, mediaID, scientificName, observationID)

    La hora del dia se deriva del `timestamp` de media.csv (hora local, ISO8601
    con offset), no de `eventStart`/`eventEnd` de observations (esos son
    offsets en segundos dentro del archivo de audio, no hora absoluta).

Output:
    Un archivo acoustic_patterns_<deploymentID>.pdf por cada deployment con
    detecciones validas, mas un acoustic_patterns_todos_los_sitios.pdf con
    todas las especies agregadas entre todos los deployments, guardados en
    esta misma carpeta.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.colors import PowerNorm

HOURS = list(range(24))
COLOR_GAMMA = 0.3  # <1 comprime los valores altos y realza los conteos bajos


def load_data(media_path, deployments_path, observations_path):
    """Carga y une observaciones + hora del dia (via media) + nombre del sitio (via deployments)."""
    obs = pd.read_csv(observations_path)
    obs = obs[obs["scientificName"].notna()].copy()

    media = pd.read_csv(media_path, usecols=["mediaID", "timestamp"])
    obs = obs.merge(media, on="mediaID", how="left")
    obs["hour"] = pd.to_datetime(obs["timestamp"]).dt.hour

    deployments = pd.read_csv(deployments_path, usecols=["deploymentID", "locationName"])

    return obs, deployments


def build_pivot(df_deployment):
    """Construye la tabla especie x hora (conteo de registros) y los totales por especie.

    Las especies quedan ordenadas de mayor a menor numero total de detecciones.
    """
    pivot = df_deployment.pivot_table(
        index="scientificName", columns="hour", values="observationID", aggfunc="count", fill_value=0
    )
    pivot = pivot.reindex(columns=HOURS, fill_value=0)

    totals = pivot.sum(axis=1).sort_values(ascending=False)
    pivot = pivot.loc[totals.index]

    return pivot, totals


def plot_heatmap(pivot, totals, title, output_path):
    """Grafica el heatmap especie x hora, con el total de registros por especie como eje Y derecho."""
    n_species = len(pivot)
    fig_height = max(4, 0.3 * n_species)

    fig, ax_heatmap = plt.subplots(figsize=(10, fig_height))

    # PowerNorm (gamma<1) comprime los conteos altos de la especie dominante y
    # realza el contraste en conteos bajos, sin excluir las celdas en 0 (a
    # diferencia de LogNorm, que no acepta ceros).
    sns.heatmap(
        pivot,
        ax=ax_heatmap,
        cmap="viridis",
        norm=PowerNorm(gamma=COLOR_GAMMA),
        cbar=False,
        linewidths=0.3,
        linecolor="white",
    )
    ax_heatmap.set_xlabel("Hora del día")
    ax_heatmap.set_ylabel("Num. detecciones")
    ax_heatmap.set_yticklabels(ax_heatmap.get_yticklabels(), fontstyle="italic")

    # Eje Y derecho con el total de registros por especie, alineado fila a fila.
    ax_totals = ax_heatmap.twinx()
    ax_totals.set_ylim(ax_heatmap.get_ylim())
    ax_totals.set_yticks(ax_heatmap.get_yticks())
    ax_totals.set_yticklabels(totals.values)
    ax_totals.set_ylabel("")
    ax_heatmap.set_title(title)
    sns.despine(ax=ax_totals, left=True, right=True, bottom=True, top=True)
    fig.tight_layout()

    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)

    return output_path


if __name__ == "__main__":
    # --- Rutas ---
    MEDIA_PATH = "../../data/output/data_preparation/media.csv"
    DEPLOYMENTS_PATH = "../../data/output/data_preparation/deployments.csv"
    OBSERVATIONS_PATH = "../../data/output/validation/observations_thresholded.csv"
    OUTPUT_DIR = "../../data/output/acoustic_patterns_per_area/plots_species/"

    obs, deployments = load_data(MEDIA_PATH, DEPLOYMENTS_PATH, OBSERVATIONS_PATH)

    for deployment_id in sorted(obs["deploymentID"].unique()):
        df_deployment = obs[obs["deploymentID"] == deployment_id]
        match = deployments.loc[deployments["deploymentID"] == deployment_id, "locationName"]
        location_name = match.iloc[0] if len(match) else None
        title = f"{deployment_id}" if pd.isna(location_name) else f"{deployment_id} — {location_name}"

        pivot, totals = build_pivot(df_deployment)
        output_path = Path(OUTPUT_DIR) / f"acoustic_patterns_{deployment_id}.png"
        plot_heatmap(pivot, totals, title, output_path)
        print(f"Guardado: {output_path} ({len(pivot)} especies, {int(totals.sum())} registros)")

    # Plot agregado: todas las especies detectadas en todos los puntos de muestreo juntos.
    pivot_all, totals_all = build_pivot(obs)
    output_path_all = Path(OUTPUT_DIR) / "acoustic_patterns_todos_los_sitios.png"
    plot_heatmap(pivot_all, totals_all, "Todos los puntos de muestreo", output_path_all)
    print(f"Guardado: {output_path_all} ({len(pivot_all)} especies, {int(totals_all.sum())} registros)")
