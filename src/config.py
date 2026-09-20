"""Configuración central y rutas portables de FatigueGuard Mining."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = PROJECT_ROOT / "models" / "fatigueguard_landmarks_ml_best.pkl"
MEDIAPIPE_MODEL_PATH = PROJECT_ROOT / "models" / "mediapipe" / "face_landmarker.task"
DATABASE_PATH = PROJECT_ROOT / "data" / "fatigueguard.db"
MPL_CONFIG_DIR = PROJECT_ROOT / ".mplconfig_local"

# Parámetros analíticos validados. Cambiarlos produciría un experimento distinto.
CALIBRATION_SECONDS = 30
PROCESSING_FPS = 5
WINDOW_SECONDS = 10
MODEL_THRESHOLD = 0.36
MODEL_ALERT_SECONDS = 8.0
EYE_RECOVERY_SECONDS = 2.0
SIGNAL_UNSTABLE_SECONDS = 2.0
ALERT_COOLDOWN_SECONDS = 10

# Se conserva por compatibilidad con el campo existente del dashboard.
HISTORY_SIZE = 5

# Controles de calidad y tiempos de la aplicación realtime.
INFERENCE_INTERVAL_SECONDS = 1.0
MIN_CALIBRATION_VALID_FRAMES = 120
MIN_BUFFER_SECONDS = WINDOW_SECONDS
MIN_BUFFER_FACE_RATE = 0.70
MAX_ABS_YAW_FOR_EYE_RULE = 35.0
REALTIME_EYE_CLOSED_THRESHOLD = 0.75
REALTIME_EYE_OPEN_THRESHOLD = 0.85
EYE_STATE_CONFIRMATION_FRAMES = 3

DEFAULT_EQUIPMENT_ID = "EQUIPO_DEMO"
