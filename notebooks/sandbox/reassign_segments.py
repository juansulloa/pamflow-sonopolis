#!/usr/bin/env python3
"""Copia los segmentos reasignados a una estructura por especie original / especie nueva.

Entrada:
  - CSV de ajustes (columnas: segmentName, scientificName, classificationProbability,
    validationResult, scientificNameAdjusted)
  - Carpeta de segmentos (se busca de forma recursiva)

Salida:
  segments_reasigned/<scientificName>/<scientificNameAdjusted>/<segmentName>

Solo se procesan las filas con scientificNameAdjusted diligenciado. Los archivos
se copian (nunca se mueven) y no se sobrescriben si ya existen en el destino.

Uso:
  python reassign_segments.py segment_species_adjustment.csv /ruta/a/segmentos
  python reassign_segments.py ajustes.csv /ruta/a/segmentos -o /ruta/segments_reasigned
"""

import argparse
import csv
import os
import shutil
import sys
from collections import defaultdict
from pathlib import Path


def build_index(segments_dir: Path, exclude_dir: Path) -> dict:
    """Indexa todos los archivos de la carpeta de segmentos por nombre de archivo."""
    index = defaultdict(list)
    exclude = exclude_dir.resolve()
    for root, dirs, files in os.walk(segments_dir):
        # Evita recorrer la carpeta de salida si queda dentro de la de segmentos
        dirs[:] = [d for d in dirs if (Path(root) / d).resolve() != exclude]
        for name in files:
            index[name].append(Path(root) / name)
    return index


def pick_source(candidates: list, scientific_name: str):
    """Elige el archivo fuente. Si hay varios, prefiere el que está bajo la carpeta
    de la especie original. Devuelve (ruta, advertencia)."""
    if len(candidates) == 1:
        return candidates[0], None
    in_species = [p for p in candidates if scientific_name in p.parts]
    pool = in_species or candidates
    pool = sorted(pool)
    if len(pool) == 1:
        return pool[0], None
    return pool[0], f"{len(pool)} coincidencias; se usó {pool[0]}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv_path", type=Path, help="CSV de ajustes de especie por segmento")
    parser.add_argument("segments_dir", type=Path, help="Carpeta de segmentos (búsqueda recursiva)")
    parser.add_argument("-o", "--output", type=Path, default=Path("segments_reasigned"),
                        help="Carpeta de salida (por defecto: ./segments_reasigned)")
    args = parser.parse_args()

    if not args.csv_path.is_file():
        sys.exit(f"No existe el CSV: {args.csv_path}")
    if not args.segments_dir.is_dir():
        sys.exit(f"No existe la carpeta de segmentos: {args.segments_dir}")

    with open(args.csv_path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    required = {"segmentName", "scientificName", "scientificNameAdjusted"}
    missing_cols = required - set(rows[0].keys() if rows else [])
    if missing_cols:
        sys.exit(f"Faltan columnas en el CSV: {', '.join(sorted(missing_cols))}")

    # Solo filas con especie reasignada
    to_process = [r for r in rows if (r["scientificNameAdjusted"] or "").strip()]
    print(f"Filas en el CSV: {len(rows)} | con reasignación: {len(to_process)}")

    index = build_index(args.segments_dir, args.output)
    print(f"Archivos indexados en {args.segments_dir}: {sum(len(v) for v in index.values())}")

    copied, skipped, not_found, warnings = 0, 0, [], []

    for r in to_process:
        seg = r["segmentName"].strip()
        original = r["scientificName"].strip()
        nueva = r["scientificNameAdjusted"].strip()

        candidates = index.get(seg)
        if not candidates:
            not_found.append(seg)
            continue

        src, warn = pick_source(candidates, original)
        if warn:
            warnings.append(f"{seg}: {warn}")

        dest_dir = args.output / original / nueva
        dest = dest_dir / seg
        if dest.exists():
            skipped += 1
            continue
        dest_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        copied += 1

    print(f"\nCopiados: {copied} | ya existentes (omitidos): {skipped} | no encontrados: {len(not_found)}")
    if warnings:
        print(f"\nAdvertencias ({len(warnings)}):")
        for w in warnings:
            print(f"  - {w}")
    if not_found:
        print(f"\nSegmentos no encontrados ({len(not_found)}):")
        for s in not_found:
            print(f"  - {s}")
    print(f"\nSalida: {args.output.resolve()}")
    return 1 if not_found else 0


if __name__ == "__main__":
    sys.exit(main())