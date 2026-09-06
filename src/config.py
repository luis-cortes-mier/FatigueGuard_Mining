"""Central configuration and portable project paths for FatigueGuard Mining."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = PROJECT_ROOT / "models" / "fatigueguard_landmarks_ml_best.pkl"
MEDIAPIPE_MODEL_PATH = PROJECT_ROOT / "models" / "mediapipe" / "face_landmarker.task"
DATABASE_PATH = PROJECT_ROOT / "data" / "fatigueguard.db"
MPL_CONFIG_DIR = PROJECT_ROOT / ".mplconfig_local"

# Validated analytical configuration. Changing these values creates a different experiment.
CALIBRATION_SECONDS = 30
PROCESSING_FPS = 5
WINDOW_SECONDS = 10
MODEL_THRESHOLD = 0.36
HISTORY_SIZE = 5
MIN_POSITIVE_WINDOWS = 3
EYE_CLOSURE_ALERT_SECONDS = 4.0
ALERT_COOLDOWN_SECONDS = 10

# Operational quality controls used by the realtime application.
INFERENCE_INTERVAL_SECONDS = 1.0
MIN_CALIBRATION_VALID_FRAMES = 120
MIN_BUFFER_SECONDS = 8.0
MIN_BUFFER_FACE_RATE = 0.70
MAX_ABS_YAW_FOR_EYE_RULE = 35.0

DEFAULT_EQUIPMENT_ID = "EQUIPO_DEMO"

