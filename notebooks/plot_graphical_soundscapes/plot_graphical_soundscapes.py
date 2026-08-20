"""
Re-grafica los graphical soundscapes (graph_*.csv) de un directorio usando una
escala de color (vmin/vmax) GLOBAL y compartida entre todos los puntos de muestreo,
en vez de la normalizacion por imagen individual que usa por defecto
`maad.features.plot_graph` (imshow sin vmin/vmax fijo).

Motivacion: cuando cada imagen se normaliza con su propio maximo, un sitio con
densidad de picos muy baja (poca senal biologica, se lee sobre todo ruido del
equipo) se "estira" igual que un sitio con senal fuerte, y ambos se ven visualmente
similares (contraste completo). Fijando vmin/vmax globales, la intensidad de color
es comparable entre sitios: un sitio con senal debil se ve tenue/oscuro en relacion
a los demas, en vez de aparentar ruido.

Input:
    Directorio con archivos graph_<deploymentID>.csv, salida del pipeline
    `graphical_soundscape` de pamflow (`maad.features.graphical_soundscape`
    guardado via `pandas.CSVDataset`, que por defecto NO escribe el indice).
    Cada fila (0-23, por posicion) es una hora del dia y cada columna es un bin
    de frecuencia en Hz (incluyendo la columna "0.0", que es el bin de 0 Hz, no
    un indice de tiempo).

Output:
    Un archivo graph_<deploymentID>.png por cada CSV, guardado en el mismo
    directorio con el mismo nombre base (sobrescribe el PNG generado por el
    pipeline, que usa normalizacion por imagen).
"""

import glob
import os

import matplotlib.pyplot as plt
import pandas as pd


def _load_graphical_soundscape(csv_path):
    """Loads a graph_*.csv as saved by the pamflow graphical_soundscape pipeline.

    The pipeline saves these with `pandas.CSVDataset`, whose default save_args
    write `index=False` — so the CSV has no explicit time column. The row
    position (0-23) is the hour of day, and the header row holds the frequency
    bins in Hz (including the "0.0" Hz bin, which is real data, not an index).
    """
    graph = pd.read_csv(csv_path)
    graph.columns = graph.columns.astype(float)
    return graph


def compute_global_vmin_vmax(graphical_soundscapes_dir):
    """Calcula el vmin/vmax global a partir de todos los graphical soundscapes de un directorio.

    Parameters
    ----------
    graphical_soundscapes_dir : str
        Directorio con archivos `graph_*.csv`, uno por punto de muestreo.

    Returns
    -------
    vmin, vmax : float
        Valores minimo y maximo de densidad de picos encontrados entre todos los CSV.
    """
    csv_paths = sorted(glob.glob(os.path.join(graphical_soundscapes_dir, "graph_*.csv")))
    if not csv_paths:
        raise FileNotFoundError(f"No se encontraron archivos graph_*.csv en {graphical_soundscapes_dir}")

    global_min, global_max = float("inf"), float("-inf")
    for csv_path in csv_paths:
        values = _load_graphical_soundscape(csv_path).values
        global_min = min(global_min, values.min())
        global_max = max(global_max, values.max())

    return global_min, global_max


def plot_graph_normalized(graph, vmin, vmax, ax=None, savefig=False, fname=None):
    """Grafica un graphical soundscape usando una escala de color fija (global).

    Variante de `maad.features.plot_graph` que recibe `vmin`/`vmax` explicitos en
    lugar de dejar que `imshow` se autoescale al minimo/maximo de cada imagen.

    Parameters
    ----------
    graph : pandas.DataFrame
        Graphical soundscape como DataFrame con tiempo como indice y frecuencia
        como columnas (mismo formato que retorna `maad.features.graphical_soundscape`).
    vmin, vmax : float
        Limites de la escala de color, compartidos entre todos los graficos para
        que sean comparables. Tipicamente el minimo/maximo global entre todos los
        puntos de muestreo (ver `compute_global_vmin_vmax`).
    ax : matplotlib.axes.Axes, optional
        Ejes donde graficar. Si no se provee, se crea una figura nueva.
    savefig : bool, optional
        Si se guarda la figura en `fname`.
    fname : str, optional
        Ruta de salida, requerida si `savefig` es True.

    Returns
    -------
    ax : matplotlib.axes.Axes
    """
    if ax is None:
        fig, ax = plt.subplots()

    ax.imshow(
        graph.values.T,
        aspect="auto",
        origin="lower",
        vmin=vmin,
        vmax=vmax,
        extent=[
            int(graph.index[0]),
            int(graph.index[-1]),
            float(graph.columns[0]),
            float(graph.columns[-1]),
        ],
    )
    ax.set_xlabel("Time (h)")
    ax.set_ylabel("Frequency (Hz)")

    if savefig:
        plt.savefig(fname, bbox_inches="tight")

    return ax


def plot_graphical_soundscapes(graphical_soundscapes_dir, vmin=None, vmax=None):
    """Re-grafica todos los graphical soundscapes de un directorio con una escala de color compartida.

    Lee cada archivo `graph_*.csv` en `graphical_soundscapes_dir` y guarda un PNG
    junto a el con el mismo nombre base, usando un `vmin`/`vmax` global entre todos
    los puntos de muestreo en vez de la normalizacion por imagen individual.

    Parameters
    ----------
    graphical_soundscapes_dir : str
        Directorio con archivos `graph_*.csv` (p. ej. `data/output/graphical_soundscape`).
        Los PNG de salida se escriben en este mismo directorio.
    vmin, vmax : float, optional
        Limites globales de la escala de color. Si no se proveen, se calculan a
        partir del minimo/maximo de densidad de picos entre todos los CSV del
        directorio (ver `compute_global_vmin_vmax`).

    Returns
    -------
    vmin, vmax : float
        Limites de escala de color efectivamente usados (utiles para inspeccionar
        o reutilizar cuando se calculan automaticamente).
    """
    csv_paths = sorted(glob.glob(os.path.join(graphical_soundscapes_dir, "graph_*.csv")))
    if not csv_paths:
        raise FileNotFoundError(f"No se encontraron archivos graph_*.csv en {graphical_soundscapes_dir}")

    if vmin is None or vmax is None:
        computed_vmin, computed_vmax = compute_global_vmin_vmax(graphical_soundscapes_dir)
        vmin = computed_vmin if vmin is None else vmin
        vmax = computed_vmax if vmax is None else vmax

    for csv_path in csv_paths:
        graph = _load_graphical_soundscape(csv_path)
        fname = os.path.splitext(csv_path)[0] + ".png"

        fig, ax = plt.subplots()
        plot_graph_normalized(graph, vmin, vmax, ax=ax, savefig=True, fname=fname)
        plt.close(fig)

    print(f"{len(csv_paths)} graficos guardados en {graphical_soundscapes_dir} (vmin={vmin:.6f}, vmax={vmax:.6f})")

    return vmin, vmax


if __name__ == "__main__":
    # --- Parametros globales (editar si se quiere fijar una escala manual) ---
    GRAPHICAL_SOUNDSCAPES_DIR = "../../data/output/graphical_soundscape"
    VMIN = None  # None => calculado automaticamente como el minimo global
    VMAX = None  # None => calculado automaticamente como el maximo global

    plot_graphical_soundscapes(GRAPHICAL_SOUNDSCAPES_DIR, vmin=VMIN, vmax=VMAX)
