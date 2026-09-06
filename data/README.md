# Datos para reproducir el modelamiento

El modelo final de FatigueGuard Mining se desarrolló con **UTA-RLDD Face Cropped Video**, usando
la variante `len60`. Los videos originales no se distribuyen en este repositorio debido a su
tamaño.

La inspección local identificó 60 sujetos. Se utilizaron 59 y se excluyó el sujeto 42 porque no
contenía todas las clases requeridas. Para la clasificación binaria se usa `state_code = 0` como
**Non Drowsy** y `state_code = 10` como **Drowsy**. El estado intermedio `state_code = 5` no se
utiliza en el modelo binario final.

## Descarga y ubicación esperada

La descarga debe realizarse manualmente desde la fuente original del dataset. Antes de
redistribuir cualquier archivo debe comprobarse allí su licencia o condición de uso; este
proyecto no presupone ni declara un permiso de redistribución.

Después de obtenerlo, ubique el contenido con esta estructura:

```text
data/
└── raw/
    └── uta_rldd_cropped/
        └── len60/
            ├── 01/
            │   ├── 0/
            │   ├── 5/
            │   └── 10/
            ├── 02/
            └── ...
```

Los nombres de estado presentes pueden variar según el contenido original. No renombre videos
ni carpetas.

## Reproducción

1. Descargue y ubique UTA-RLDD como se indica arriba.
2. Ejecute `notebooks/08_landmarks_feature_engineering.ipynb`. El notebook utiliza el manifiesto
   versionado `outputs/metrics/uta_rldd_subject_split.csv`, reserva 30 segundos para calibración
   personal y reconstruye `data/processed/uta_rldd_landmark_windows.csv`.
3. Ejecute `notebooks/09_ml_landmarks_xgboost.ipynb` para reproducir Random Forest vs XGBoost.

`data/raw/`, `data/processed/` y las bases SQLite se excluyen de Git. La aplicación realtime no
necesita descargar UTA-RLDD: utiliza directamente el modelo ya entrenado incluido en `models/`.
