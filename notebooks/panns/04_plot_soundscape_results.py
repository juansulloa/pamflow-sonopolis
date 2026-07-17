"""
Grafica de barras apiladas horizontales: composicion del paisaje sonoro
(biofonia, antropofonia, geofonia, silencio) por sitio, agrupado por
area protegida y ordenado por %antropofonia dentro de cada area.

Input:
    soundscape_comparison_table.csv  (deploymentID, pct_biophony, pct_anthropophony,
                                       pct_geophony, pct_silence, ...)
    deployments.csv                  (deploymentID, locationName, habitat, ...)
"""

import matplotlib.pyplot as plt
import pandas as pd

# --- Cargar datos ---
comparison = pd.read_csv("../../data/output/panns/soundscape_comparison_table.csv")
deployments = pd.read_csv("../../data/output/data_preparation/deployments.csv")
output_path = "../../data/output/panns/soundscape_stacked_by_area.png"

# --- Union y extraer el nombre del area protegida ---
df = comparison.merge(
    deployments[["deploymentID", "locationName"]], on="deploymentID", how="left"
)
df["area"] = df["locationName"].str.split("-p").str[0]
df["area"] = df["area"].fillna("Sin área asignada")

# --- Orden de areas: Entrenubes primero (otro ecosistema), luego humedales
# en el orden solicitado ---
area_order = [
    "Entrenubes",
    "Córdoba",
    "JuanAmarillo",
    "Tibanica",
    "Torca-Guaymaral",
    "TinguaAzul",
    "ElTunjo",
    "SantaMariaDelLago",
    "LaVaca",
]
# Cualquier area no listada explicitamente (p. ej. "Sin área asignada" o
# SantaMariaDelLago si no se menciono) se agrega al final, en el orden en que
# aparezca en los datos.
remaining_areas = [a for a in df["area"].unique() if a not in area_order]
area_order = area_order + remaining_areas

df["area"] = pd.Categorical(df["area"], categories=area_order, ordered=True)

# --- Ordenar: por el orden de area definido arriba, y dentro de cada area
# por %antropofonia ascendente ---
df = df.sort_values(["area", "pct_anthropophony"], ascending=[True, True]).reset_index(drop=True)

# --- Categorias y colores ---
categories = ["pct_biophony", "pct_anthropophony", "pct_geophony", "pct_silence"]
labels = ["Biofonía", "Antropofonía", "Geofonía", "Silencio"]
colors = ["#4C9A5B", "#C0453D", "#4E79A7", "#D9D2C5"]

#%% --- Figura ---
# Se inserta una fila "vacia" (gap) entre cada area para separar visualmente
# los grupos, ademas de la linea divisoria existente.
GAP = 0.5  # unidades de espacio en blanco entre areas (en "filas" equivalentes)

area_list = list(dict.fromkeys(df["area"]))  # orden de aparicion, sin duplicados
y_positions = []
y_cursor = 0
area_y_ranges = {}

for area in area_list:
    n_sites = (df["area"] == area).sum()
    start = y_cursor
    for _ in range(n_sites):
        y_positions.append(y_cursor)
        y_cursor += 1
    end = y_cursor - 1
    area_y_ranges[area] = (start, end)
    y_cursor += GAP  # espacio extra antes de la siguiente area

df["y_pos"] = y_positions

fig, ax = plt.subplots(figsize=(10, max(6, y_cursor * 0.34)))

left = pd.Series([0.0] * len(df))

for cat, label, color in zip(categories, labels, colors):
    ax.barh(df["y_pos"], df[cat], left=left, color=color, label=label, height=0.7)
    left = left + df[cat]

# --- Etiquetas del eje Y: deploymentID ---
ax.set_yticks(df["y_pos"])
ax.set_yticklabels(df["deploymentID"], fontsize=8)
ax.invert_yaxis()  # primer sitio arriba
ax.set_ylim(y_cursor - GAP - 0.5, -1)  # deja espacio arriba/abajo, respeta el invert

# --- Separadores y etiquetas de area protegida ---
for area, (i_min, i_max) in area_y_ranges.items():
    # linea separadora sutil a mitad del espacio en blanco, antes del grupo
    if i_min > 0:
        sep_y = i_min - (GAP / 2) - 0.5
    # etiqueta del area, centrada verticalmente en su bloque, a la derecha
    y_center = (i_min + i_max) / 2
    ax.text(103, y_center, area, va="center", ha="left", fontsize=9, fontweight="normal")

ax.set_xlim(0, 100)
ax.set_xlabel("Composición del paisaje sonoro (%)")
ax.set_title("Composición relativa del paisaje sonoro por sitio de muestreo", fontsize=12, fontweight="bold")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.06), ncol=4, frameon=False)
ax.spines[["top", "right"]].set_visible(False)

fig.tight_layout()
fig.savefig(output_path, dpi=150, bbox_inches="tight")
print(f"Guardado: {output_path}")