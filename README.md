# FatigueGuard Mining

FatigueGuard Mining es un MVP académico basado en visión por computador y Machine Learning para
detectar patrones visuales asociados con posible somnolencia. Fue concebido inicialmente para
operadores de camiones mineros, pero todavía no está validado para una operación minera real.

No es una herramienta de diagnóstico médico. Tampoco controla ni detiene automáticamente el
vehículo.

## Cómo funciona

```text
Webcam
  → MediaPipe
  → calibración personal de 30 s
  → ventana temporal de 10 s
  → 24 variables
  → Random Forest
  → P(Somnolencia)
  → persistencia temporal
  → alerta
  → SQLite
  → dashboard
```

- MediaPipe procesa aproximadamente 5 FPS.
- El Random Forest actualiza su evaluación aproximadamente cada segundo.
- Cada predicción resume los últimos 10 segundos; no representa el instante exacto.
- El threshold del modelo es `0.36`.
- Una predicción positiva aislada no genera una alerta: debe mantenerse durante 8 segundos.
- Si ambos ojos permanecen claramente abiertos durante 2 segundos, se confirma recuperación y se
  reinicia la persistencia.
- La probabilidad del modelo no se modifica durante la recuperación.
- Los ojos no generan una alerta independiente. La única fuente de `ALERTA` es el Random Forest.
- El audio tiene un cooldown de 10 segundos.
- Los eventos se guardan en SQLite y pueden revisarse desde el dashboard.

El estado ocular usa la referencia personal de cada usuario. Se considera `CLOSED` cuando ambos EAR
relativos son menores que `0.75`, `OPEN` cuando ambos son mayores que `0.85` e indeterminado en la
zona intermedia. Se necesitan tres frames consecutivos para confirmar `OPEN` o `CLOSED`.

## Estados

- `CALIBRANDO`: el sistema obtiene la referencia personal del usuario.
- `NORMAL`: no se mantiene una condición positiva de somnolencia.
- `POSIBLE SOMNOLENCIA`: la probabilidad supera el threshold y se evalúa su persistencia.
- `ALERTA`: la condición positiva se mantiene durante el tiempo definido y se genera una alerta.

Durante la preparación inicial de la ventana puede mostrarse `PREPARANDO ANALISIS`; es un mensaje de
interfaz, no un estado del modelo.

## Instalación

Se recomienda Python 3.12 y una webcam accesible desde OpenCV.

```text
git clone https://github.com/luis-cortes-mier/FatigueGuard_Mining.git
cd FatigueGuard_Mining
```

Windows:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

macOS:

```bash
python3 -m venv .venv
./.venv/bin/python -m pip install --upgrade pip
./.venv/bin/python -m pip install -r requirements.txt
```

## Ejecutar el monitoreo

Windows:

```powershell
.\.venv\Scripts\python.exe -m src.realtime_fatigueguard --equipment-id CAMION_01
```

macOS:

```bash
./.venv/bin/python -m src.realtime_fatigueguard --equipment-id CAMION_01
```

Durante la calibración inicial, el usuario debe mantener una postura normal y permanecer alerta.

- `D`: mostrar u ocultar el diagnóstico.
- `Q`: salir.

Si la cámara principal no corresponde al índice 0, agregue `--camera 1` al comando.

## Ejecutar el dashboard

Windows:

```powershell
.\.venv\Scripts\python.exe -m streamlit run dashboard/app.py
```

macOS:

```bash
./.venv/bin/python -m streamlit run dashboard/app.py
```

Streamlit abrirá la interfaz en el navegador. El dashboard consulta la misma base SQLite que escribe
el monitoreo realtime.

## Notebooks

`notebooks/08_landmarks_feature_engineering.ipynb`

Procesamiento con MediaPipe, calibración personal, landmarks, construcción de ventanas y generación
de las 24 variables.

`notebooks/09_ml_landmarks_xgboost.ipynb`

Entrenamiento, comparación entre Random Forest y XGBoost, selección del modelo, definición del
threshold con Validation y evaluación final sobre Test.

Para ejecutarlos se requieren los datos indicados en `data/README.md` y las dependencias de
`requirements-notebooks.txt`.

## Estructura del proyecto

```text
FatigueGuard_Mining/
├── src/
├── dashboard/
├── models/
├── notebooks/
├── scripts/
├── data/
├── requirements.txt
├── requirements-notebooks.txt
└── README.md
```

## Limitaciones

- Es un MVP académico construido y validado con datos públicos.
- Todavía no utiliza datos de operadores mineros reales.
- La demostración actual usa una webcam convencional.
- Falta validar el sistema en condiciones reales de cabina.
- Falta evaluar baja iluminación, vibración, casco, gafas y otras oclusiones.
- Un despliegue industrial requerirá datos propios y hardware de captura adecuado.
