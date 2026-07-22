"""
Comparacion por lotes del paisaje sonoro entre todos los despliegues (deployments).

Recorre un directorio con archivos {deploymentID}_observations.csv (generados
por 01_analyze_tagging.py), calcula el indice de OCUPACION independiente por
categoria (biophony/anthropophony/geophony) para cada uno -- ver
soundscape_profile_v2.py para el detalle metodologico -- y construye una
tabla comparativa:

    deploymentID | n_windows | pct_biophony | pct_anthropophony | pct_geophony |
    top_biophony_labels | top_anthropophony_labels | top_geophony_labels

Nota: estos porcentajes son independientes entre si (no suman 100%), ya que
las categorias pueden solaparse en el tiempo (p. ej. trafico y canto de aves
simultaneos).

Ejecutar directamente (ajustar `save_path` abajo), o importar
`build_comparison_table` desde otro script.
"""

import re
from pathlib import Path

import pandas as pd
from label_category_crosswalk import LABEL_CATEGORY

MAIN_CATEGORIES = ["biophony", "anthropophony", "geophony", "silence"]


def categorize(label):
    return LABEL_CATEGORY.get(label, "unclassified")


def soundscape_profile(csv_path, top_n=5, categories=MAIN_CATEGORIES):
    """
    Indice de ocupacion independiente por categoria (no normalizado entre si).
    Ver docstring de soundscape_profile_v2.py para la logica completa.
    """
    df = pd.read_csv(csv_path)

    df["window_id"] = list(zip(df["mediaID"], df["eventStart"]))
    n_windows = df["window_id"].nunique()

    df["category"] = df["label"].apply(categorize)
    df_valid = df[df["category"].notna()].copy()

    # Maximo de confianza por (ventana, categoria) -- evita doble conteo de
    # etiquetas redundantes dentro de una misma categoria en una ventana.
    window_cat_max = (
        df_valid.groupby(["window_id", "category"])["classificationProbability"]
        .max()
        .reset_index()
    )

    occupancy = {}
    for cat in categories:
        cat_sum = window_cat_max.loc[window_cat_max["category"] == cat, "classificationProbability"].sum()
        occupancy[cat] = cat_sum / n_windows  # ventanas sin deteccion -> 0 implicito

    occupancy_series = pd.Series(occupancy).sort_values(ascending=False)

    # Etiquetas dominantes por categoria (informativo)
    label_weight = (
        df_valid.groupby(["category", "label"])["classificationProbability"]
        .sum().div(n_windows).reset_index(name="weighted_proportion")
    )
    top_labels = (
        label_weight.sort_values(["category", "weighted_proportion"], ascending=[True, False])
        .groupby("category").head(top_n)
    )

    return occupancy_series, top_labels, n_windows


def extract_deployment_id(filename, suffix="_observations.csv"):
    """Recupera el deploymentID desde un archivo '{deploymentID}_observations.csv'."""
    name = Path(filename).name
    if name.endswith(suffix):
        return name[: -len(suffix)]
    return Path(filename).stem


def build_comparison_table(save_path, top_n=3, file_pattern=r".*_observations\.csv$"):
    save_path = Path(save_path)
    files = sorted([f for f in save_path.glob("*.csv") if re.match(file_pattern, f.name)])

    if not files:
        print(f"No se encontraron CSVs de observaciones en {save_path}")
        return pd.DataFrame()

    rows = []
    for f in files:
        deployment_id = extract_deployment_id(f.name)
        try:
            occupancy, top_labels, n_windows = soundscape_profile(f, top_n=top_n)
        except Exception as e:
            print(f"  [omitido] {f.name}: {e}")
            continue

        row = {
            "deploymentID": deployment_id,
            "n_windows": n_windows,
        }
        for cat in MAIN_CATEGORIES:
            row[f"pct_{cat}"] = round(occupancy.get(cat, 0.0) * 100, 2)

        for cat in MAIN_CATEGORIES:
            subset = top_labels[top_labels["category"] == cat]
            labels_str = ", ".join(subset["label"].tolist())
            row[f"top_{cat}_labels"] = labels_str

        rows.append(row)
        print(f"  Procesado: {deployment_id} "
              f"(bio={row['pct_biophony']}%, antropo={row['pct_anthropophony']}%, "
              f"geo={row['pct_geophony']}%)")

    comparison_df = pd.DataFrame(rows)
    comparison_df = comparison_df.sort_values("deploymentID").reset_index(drop=True)
    return comparison_df


if __name__ == "__main__":
    # --- Variables ---
    save_path = "../../data/output/panns/"     # carpeta con {deploymentID}_observations.csv
    output_path = "../../data/output/panns/soundscape_comparison_table.csv"
    top_n = 3

    print(f"Escaneando: {save_path}\n")
    comparison_df = build_comparison_table(save_path, top_n=top_n)

    if comparison_df.empty:
        print("No se generaron resultados.")
    else:
        comparison_df.to_csv(output_path, index=False)
        print(f"\nGuardado: {output_path}")
        print(f"({len(comparison_df)} despliegues)\n")
        print(comparison_df[["deploymentID", "n_windows", "pct_biophony",
                              "pct_anthropophony", "pct_geophony", "pct_silence"]].to_string(index=False))