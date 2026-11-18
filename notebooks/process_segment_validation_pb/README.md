# process_segment_validation_pb

Scripts para procesar las validaciones manuales de segmentos de audio y producir la tabla final de observaciones. Son una versión standalone (sin Kedro) de los mismos tres pasos que implementa el pipeline `detection_validation`, que es la forma recomendada de correr el proceso. Los scripts sirven para explorar o correr pasos sueltos.

## Con Kedro (recomendado)

Las opciones (`threshold_method`, `keep_unthresholded`, `apply_confusions`, etc.) se configuran en `detection_validation_parameters` de `conf/<entorno>/parameters.yml`. Las entradas y salidas están en `conf/base/catalog/detection_validation.yml`.

```bash
kedro run --env=sonopolis_pb --pipeline=detection_validation          # periodo B
kedro run --env=sonopolis_pb_anura --pipeline=detection_validation    # periodo B, anuros
kedro run --env=sonopolis_pa --pipeline=detection_validation          # periodo A
```

Para correr un solo paso, agregar `--node=get_validation_status_node`, `--node=find_model_threshold_node` o `--node=update_observations_node`.

## Tabla de inicio

La entrada es la tabla de validaciones exportada por el programa de revisión de segmentos **audita_gui** (una fila por segmento validado). Se guarda en `data/output_<periodo>/validation/validation_results/`, por ejemplo `validaciones_anura_pb.csv`.

Columnas esperadas: `scientificNameProposed`, `validationCategory` (`positive` | `negative` | `uncertain` | `reassigned`), `scientificNameValidated`, `fileName`, `classifiedBy`, `favorite`, `validationComment`.

## Pasos

| Script | Entrada | Salida |
|---|---|---|
| `01_get_validation_status.py` | tabla de audita_gui | `segment_validation_results.csv` |
| `02_find_model_threshold.py` | `segment_validation_results.csv` | `species_confidence_thresholds.csv`, `species_confusions.csv` |
| `03_update_observations.py` | `observations.csv` + los tres anteriores | `observations_thresholded.csv` |

Cada script documenta sus opciones en el docstring (`--help`).

## Convención de carpetas

| Periodo de muestreo | Carpeta de salida |
|---|---|
| A | `data/output_pa` |
| B | `data/output_pb` |
| B (anuros) | `data/output_pb_anura` |

Dentro de cada una: `species_detection/observations.csv` y `validation/` (donde caen los resultados).

## Comandos de los scripts (standalone)

Correr desde la raíz del repositorio. Ejemplo para el periodo **B** (`pb`); para el periodo **A** cambiar `output_pb` por `output_pa` y el nombre del CSV de validaciones.

```bash
cd notebooks/process_segment_validation_pb
V=../../data/output_pb/validation

# 1. Estado de validación por segmento
python 01_get_validation_status.py \
    $V/validation_results/validaciones_pb.csv \
    -o $V/segment_validation_results.csv

# 2. Umbrales de confianza por especie (y tabla de confusiones)
python 02_find_model_threshold.py \
    -i $V/segment_validation_results.csv \
    --output-csv $V/species_confidence_thresholds.csv \
    --output-confusions $V/species_confusions.csv \
    [--method {auto,model,sweep}] [--plot -o $V/figures]

# 3. Observaciones finales (filtradas por umbral y validadas)
python 03_update_observations.py \
    -i ../../data/output_pb/species_detection/observations.csv \
    -t $V/species_confidence_thresholds.csv \
    -v $V/segment_validation_results.csv \
    -c $V/species_confusions.csv \
    -o $V/observations_thresholded.csv \
    [--keep-unthresholded] [--apply-confusions]
```

Ejemplo para anuros del periodo B:

```bash
V=../../data/output_pb_anura/validation

python 01_get_validation_status.py \
    $V/validation_results/validaciones_anura_pb.csv \
    -o $V/segment_validation_results.csv

python 02_find_model_threshold.py \
    -i $V/segment_validation_results.csv \
    --output-csv $V/species_confidence_thresholds.csv \
    --output-confusions $V/species_confusions.csv

python 03_update_observations.py \
    -i ../../data/output_pb_anura/species_detection/observations.csv \
    -t $V/species_confidence_thresholds.csv \
    -v $V/segment_validation_results.csv \
    -c $V/species_confusions.csv \
    -o $V/observations_thresholded.csv
```
