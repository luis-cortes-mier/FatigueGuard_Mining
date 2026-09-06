"""First webcam MVP for FatigueGuard Mining.

For development while seated in front of a computer only. This application detects visual
patterns associated with possible drowsiness; it is not a medical diagnostic system and must
not be tested while driving or operating machinery.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import threading
import time
import tempfile
import uuid
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

try:
    from .config import (
        ALERT_COOLDOWN_SECONDS, CALIBRATION_SECONDS, DATABASE_PATH,
        DEFAULT_EQUIPMENT_ID, EYE_CLOSURE_ALERT_SECONDS, HISTORY_SIZE,
        INFERENCE_INTERVAL_SECONDS, MAX_ABS_YAW_FOR_EYE_RULE,
        MEDIAPIPE_MODEL_PATH, MIN_BUFFER_FACE_RATE, MIN_BUFFER_SECONDS,
        MIN_CALIBRATION_VALID_FRAMES, MIN_POSITIVE_WINDOWS, MODEL_PATH,
        MODEL_THRESHOLD, MPL_CONFIG_DIR, PROCESSING_FPS, WINDOW_SECONDS,
    )
except ImportError:  # Supports direct execution from src/.
    from config import (
        ALERT_COOLDOWN_SECONDS, CALIBRATION_SECONDS, DATABASE_PATH,
        DEFAULT_EQUIPMENT_ID, EYE_CLOSURE_ALERT_SECONDS, HISTORY_SIZE,
        INFERENCE_INTERVAL_SECONDS, MAX_ABS_YAW_FOR_EYE_RULE,
        MEDIAPIPE_MODEL_PATH, MIN_BUFFER_FACE_RATE, MIN_BUFFER_SECONDS,
        MIN_CALIBRATION_VALID_FRAMES, MIN_POSITIVE_WINDOWS, MODEL_PATH,
        MODEL_THRESHOLD, MPL_CONFIG_DIR, PROCESSING_FPS, WINDOW_SECONDS,
    )

# MediaPipe imports matplotlib internally. Keep its cache inside the project.
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))

import cv2
import joblib
import mediapipe as mp
import numpy as np
import pandas as pd
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

try:
    from .database import (
        get_recent_events, initialize_database, insert_event, upsert_current_status,
    )
    from .facial_features import (
        EAR_PERCLOS_THRESHOLD, FEATURE_COLUMNS, CalibrationBaseline,
        add_relative_measurements, calculate_calibration_baseline,
        calculate_window_features, invalid_measurement, measurements_from_landmarks,
    )
except ImportError:  # Supports direct execution from src/.
    from database import get_recent_events, initialize_database, insert_event, upsert_current_status
    from facial_features import (
        EAR_PERCLOS_THRESHOLD, FEATURE_COLUMNS, CalibrationBaseline,
        add_relative_measurements, calculate_calibration_baseline,
        calculate_window_features, invalid_measurement, measurements_from_landmarks,
    )


class EyeClosureTracker:
    """Track one uninterrupted, valid, frontal, bilateral eye closure."""

    def __init__(self, threshold_seconds: float = EYE_CLOSURE_ALERT_SECONDS):
        self.threshold_seconds = threshold_seconds
        self._closure_start: float | None = None
        self.duration_seconds = 0.0

    def update(self, record: Mapping, now: float) -> bool:
        valid = bool(record.get("valid_face", False))
        yaw_delta = record.get("yaw_delta", np.nan)
        ear_left = record.get("ear_left", np.nan)
        ear_right = record.get("ear_right", np.nan)
        ear_base = record.get("ear_base", np.nan)
        usable = (
            valid
            and np.isfinite(yaw_delta)
            and abs(float(yaw_delta)) <= MAX_ABS_YAW_FOR_EYE_RULE
            and np.isfinite(ear_left)
            and np.isfinite(ear_right)
            and np.isfinite(ear_base)
            and float(ear_base) > 0
        )
        both_closed = usable and (
            float(ear_left) / float(ear_base) < EAR_PERCLOS_THRESHOLD
            and float(ear_right) / float(ear_base) < EAR_PERCLOS_THRESHOLD
        )
        if both_closed:
            if self._closure_start is None:
                self._closure_start = now
            self.duration_seconds = max(0.0, now - self._closure_start)
        else:
            # Invalid/lateral/lost frames break continuity; separated blinks never accumulate.
            self._closure_start = None
            self.duration_seconds = 0.0
        return self.duration_seconds >= self.threshold_seconds


class EventLogger:
    """Aggregate possible-drowsiness states and persist one SQLite row per episode."""

    def __init__(
        self,
        session_id: str,
        equipment_id: str,
        db_path: Path = DATABASE_PATH,
    ):
        self.session_id = session_id
        self.equipment_id = equipment_id
        self.db_path = Path(db_path)
        self.active: dict | None = None
        initialize_database(self.db_path)

    def update(
        self,
        active_condition: bool,
        alert_reasons: list[str],
        probability: float | None,
        perclos: float | None,
        eye_closure_seconds: float,
        alert_triggered: bool,
        wall_time: float | None = None,
    ) -> None:
        timestamp = time.time() if wall_time is None else wall_time
        if active_condition:
            if self.active is None:
                self.active = {
                    "event_id": str(uuid.uuid4()),
                    "start_epoch": timestamp,
                    "timestamp_start": datetime.fromtimestamp(timestamp, timezone.utc).isoformat(timespec="seconds"),
                    "reasons": set(), "probabilities": [], "perclos": [],
                    "max_eye_closure_seconds": 0.0, "alert_triggered": False,
                }
            self.active["reasons"].update(alert_reasons)
            if probability is not None and np.isfinite(probability):
                self.active["probabilities"].append(float(probability))
            if perclos is not None and np.isfinite(perclos):
                self.active["perclos"].append(float(perclos))
            self.active["max_eye_closure_seconds"] = max(
                self.active["max_eye_closure_seconds"], float(eye_closure_seconds)
            )
            self.active["alert_triggered"] |= bool(alert_triggered)
        elif self.active is not None:
            self._close(timestamp)

    def close_open_event(self) -> None:
        if self.active is not None:
            self._close(time.time())

    def _close(self, end_epoch: float) -> None:
        event = self.active
        if event is None:
            return
        probabilities = event["probabilities"]
        perclos_values = event["perclos"]
        reasons = sorted(event["reasons"]) or ["POSSIBLE_DROWSINESS"]
        row = {
            "event_id": event["event_id"],
            "session_id": self.session_id,
            "equipment_id": self.equipment_id,
            "timestamp_start": event["timestamp_start"],
            "timestamp_end": datetime.fromtimestamp(end_epoch, timezone.utc).isoformat(timespec="seconds"),
            "duration_seconds": round(max(0.0, end_epoch - event["start_epoch"]), 3),
            "alert_reason": " + ".join(reasons),
            "max_probability": max(probabilities) if probabilities else None,
            "mean_probability": float(np.mean(probabilities)) if probabilities else None,
            "max_eye_closure_seconds": event["max_eye_closure_seconds"],
            "mean_perclos": float(np.mean(perclos_values)) if perclos_values else None,
            "alert_triggered": event["alert_triggered"],
        }
        insert_event(row, self.db_path)
        self.active = None


class AlertSound:
    def __init__(self, cooldown_seconds: float = ALERT_COOLDOWN_SECONDS):
        self.cooldown_seconds = cooldown_seconds
        self.last_played = -np.inf

    def play_if_allowed(self, now: float) -> None:
        if now - self.last_played < self.cooldown_seconds:
            return
        self.last_played = now
        threading.Thread(target=self._play, daemon=True).start()

    @staticmethod
    def _play() -> None:
        if platform.system() == "Windows":
            try:
                import winsound
                winsound.Beep(1_200, 700)
                winsound.Beep(1_500, 700)
                return
            except RuntimeError:
                pass
        print("\a", end="", flush=True)


def build_face_landmarker() -> vision.FaceLandmarker:
    if not MEDIAPIPE_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Falta {MEDIAPIPE_MODEL_PATH}. Ejecute primero el notebook 08 o descargue "
            "el Face Landmarker oficial de MediaPipe."
        )
    options = vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(MEDIAPIPE_MODEL_PATH)),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return vision.FaceLandmarker.create_from_options(options)


def load_model_artifact(path: Path = MODEL_PATH) -> dict:
    artifact = joblib.load(path)
    required = {"model", "model_name", "feature_columns", "threshold"}
    if not required.issubset(artifact):
        raise ValueError(f"Artefacto incompleto: faltan {required - set(artifact)}")
    if list(artifact["feature_columns"]) != FEATURE_COLUMNS:
        raise ValueError("Las 24 features del artefacto no coinciden con facial_features.py")
    if not np.isclose(float(artifact["threshold"]), MODEL_THRESHOLD):
        raise ValueError(
            f"Threshold del artefacto ({artifact['threshold']}) distinto a {MODEL_THRESHOLD}"
        )
    return artifact


def overlay_lines(frame, lines: list[tuple[str, tuple[int, int, int]]]) -> None:
    y = 32
    for text, color in lines:
        cv2.putText(frame, text, (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.68, color, 2, cv2.LINE_AA)
        y += 29


def classify_decision_history(decisions) -> tuple[str, int]:
    """Apply the validated 3-of-5 persistence rule to model decisions."""
    positives = sum(bool(value) for value in decisions)
    if positives >= MIN_POSITIVE_WINDOWS:
        return "ALERTA", positives
    if positives:
        return "POSIBLE SOMNOLENCIA", positives
    return "NORMAL", 0


def run_realtime(
    camera_index: int = 0,
    debug: bool = False,
    equipment_id: str = DEFAULT_EQUIPMENT_ID,
    session_id: str | None = None,
    db_path: Path = DATABASE_PATH,
) -> None:
    session_id = session_id or str(uuid.uuid4())
    initialize_database(db_path)
    artifact = load_model_artifact()
    model = artifact["model"]
    expected_features = list(artifact["feature_columns"])
    capture = cv2.VideoCapture(camera_index)
    if not capture.isOpened():
        raise RuntimeError(f"No se pudo abrir la webcam {camera_index}")

    detector = build_face_landmarker()
    calibration_records: list[dict] = []
    baseline: CalibrationBaseline | None = None
    buffer: deque[dict] = deque()
    decision_history: deque[bool] = deque(maxlen=HISTORY_SIZE)
    eye_tracker = EyeClosureTracker()
    sound = AlertSound()
    event_logger = EventLogger(session_id, equipment_id, db_path)

    application_start = time.perf_counter()
    last_processing_time = -np.inf
    last_inference_time = -np.inf
    last_probability: float | None = None
    last_features: dict | None = None
    current_state = "CALIBRANDO"
    positive_windows = 0
    alert_reasons: list[str] = []
    face_detected = False
    debug_landmarks = None
    calibration_message_until = 0.0

    print("FatigueGuard MVP: prueba sentado frente al computador. Presione Q para salir.")
    print(f"Sesión: {session_id} | Equipo: {equipment_id} | SQLite: {db_path}")
    upsert_current_status(session_id, equipment_id, "CALIBRANDO", None, 0, db_path=db_path)
    try:
        while True:
            success, frame = capture.read()
            if not success:
                raise RuntimeError("Se perdió la lectura de la webcam")
            now = time.perf_counter()

            if now - last_processing_time >= 1.0 / PROCESSING_FPS:
                last_processing_time = now
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                timestamp_ms = int((now - application_start) * 1000)
                result = detector.detect_for_video(mp_image, timestamp_ms)
                if result.face_landmarks:
                    debug_landmarks = result.face_landmarks[0]
                    record = measurements_from_landmarks(debug_landmarks, frame.shape[1], frame.shape[0])
                    face_detected = True
                else:
                    debug_landmarks = None
                    record = invalid_measurement()
                    face_detected = False
                record["timestamp"] = now

                if baseline is None:
                    if record["valid_face"]:
                        calibration_records.append(record)
                    elapsed = now - application_start
                    if elapsed >= CALIBRATION_SECONDS and len(calibration_records) >= MIN_CALIBRATION_VALID_FRAMES:
                        baseline = calculate_calibration_baseline(calibration_records)
                        calibration_message_until = now + 2.0
                        current_state = "NORMAL"
                        print("CALIBRACION COMPLETADA", json.dumps(baseline.to_dict(), indent=2))
                    elif elapsed >= CALIBRATION_SECONDS:
                        current_state = "CALIBRANDO - ROSTRO INSUFICIENTE, CONTINUE ALERTA"
                else:
                    record = add_relative_measurements(record, baseline)
                    record["ear_base"] = baseline.ear_base
                    buffer.append(record)
                    while buffer and now - float(buffer[0]["timestamp"]) > WINDOW_SECONDS:
                        buffer.popleft()

                    critical_active = eye_tracker.update(record, now)
                    buffer_span = now - float(buffer[0]["timestamp"]) if buffer else 0.0
                    valid_rate = np.mean([bool(item["valid_face"]) for item in buffer]) if buffer else 0.0
                    enough_buffer = (
                        buffer_span >= MIN_BUFFER_SECONDS
                        and valid_rate >= MIN_BUFFER_FACE_RATE
                    )
                    new_inference = False
                    if enough_buffer and now - last_inference_time >= INFERENCE_INTERVAL_SECONDS:
                        candidate_features = calculate_window_features(list(buffer), PROCESSING_FPS)
                        feature_frame = pd.DataFrame([candidate_features], columns=expected_features)
                        if list(feature_frame.columns) != expected_features:
                            raise RuntimeError("Orden de features incompatible con el modelo")
                        if not np.isfinite(feature_frame.to_numpy(dtype=float)).all():
                            last_probability = None
                            enough_buffer = False
                        else:
                            last_probability = float(model.predict_proba(feature_frame)[0, 1])
                            decision_history.append(last_probability >= MODEL_THRESHOLD)
                            last_features = candidate_features
                            last_inference_time = now
                            new_inference = True

                    model_state, positive_windows = classify_decision_history(decision_history)
                    model_alert = model_state == "ALERTA"
                    alert_reasons = []
                    if not record["valid_face"]:
                        # Do not preserve or accumulate a drowsiness decision without a face.
                        decision_history.clear()
                        last_probability = None
                        last_features = None
                        current_state = "ROSTRO NO DETECTADO"
                    elif critical_active:
                        if model_alert:
                            alert_reasons.append("MODEL_PERSISTENCE")
                        alert_reasons.append("PROLONGED_EYE_CLOSURE")
                        current_state = "ALERTA INMEDIATA" if critical_active else "ALERTA"
                        sound.play_if_allowed(now)
                    elif not enough_buffer:
                        decision_history.clear()
                        last_probability = None
                        last_features = None
                        current_state = "DATOS FACIALES INSUFICIENTES"
                    elif model_alert:
                        alert_reasons.append("MODEL_PERSISTENCE")
                        current_state = "ALERTA"
                        sound.play_if_allowed(now)
                    elif model_state == "POSIBLE SOMNOLENCIA":
                        current_state = model_state
                    else:
                        current_state = "NORMAL"

                    active_episode = current_state in {"POSIBLE SOMNOLENCIA", "ALERTA", "ALERTA INMEDIATA"}
                    event_logger.update(
                        active_condition=active_episode,
                        alert_reasons=alert_reasons,
                        probability=last_probability if new_inference else None,
                        perclos=(
                            last_features.get("perclos_relative")
                            if new_inference and last_features else None
                        ),
                        eye_closure_seconds=eye_tracker.duration_seconds,
                        alert_triggered=bool(alert_reasons),
                    )

                persisted_state = current_state
                if baseline is None:
                    persisted_state = "CALIBRANDO"
                elif current_state == "ALERTA INMEDIATA":
                    persisted_state = "ALERTA"
                upsert_current_status(
                    session_id=session_id,
                    equipment_id=equipment_id,
                    status=persisted_state,
                    probability=last_probability,
                    positive_windows=sum(decision_history),
                    db_path=db_path,
                )

            if baseline is None:
                elapsed = time.perf_counter() - application_start
                lines = [
                    ("CALIBRANDO - Mantenga posicion normal y permanezca alerta", (0, 220, 255)),
                    (f"Tiempo: {elapsed:.1f}/{CALIBRATION_SECONDS}s | rostros validos: {len(calibration_records)}", (255, 255, 255)),
                ]
            else:
                display_state = current_state
                if not face_detected and not alert_reasons:
                    display_state = "ROSTRO NO DETECTADO"
                color = (0, 0, 255) if alert_reasons else ((0, 165, 255) if current_state == "POSIBLE SOMNOLENCIA" else (0, 220, 0))
                lines = [(display_state, color)]
                if time.perf_counter() < calibration_message_until:
                    lines.append(("CALIBRACION COMPLETADA", (0, 220, 0)))
                positive_windows = sum(decision_history)
                ear_relative = buffer[-1].get("ear_relative", np.nan) if buffer else np.nan
                mar_relative = buffer[-1].get("mar_relative", np.nan) if buffer else np.nan
                pitch_delta = buffer[-1].get("pitch_delta", np.nan) if buffer else np.nan
                perclos = last_features.get("perclos_relative", np.nan) if last_features else np.nan
                probability_text = "--" if last_probability is None else f"{last_probability:.3f}"
                lines.extend([
                    (f"P(Drowsy): {probability_text} | positivas: {positive_windows}/{HISTORY_SIZE}", (255, 255, 255)),
                    (f"EAR rel: {ear_relative:.3f} | PERCLOS: {perclos:.3f}", (255, 255, 255)),
                    (f"MAR rel: {mar_relative:.3f} | pitch delta: {pitch_delta:.1f}", (255, 255, 255)),
                    (f"Cierre ocular actual: {eye_tracker.duration_seconds:.1f}s", (255, 255, 255)),
                ])
                if alert_reasons:
                    lines.append(("Motivo: " + " + ".join(alert_reasons), (0, 0, 255)))
            if debug and debug_landmarks:
                for landmark in debug_landmarks:
                    point = (int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0]))
                    cv2.circle(frame, point, 1, (255, 180, 0), -1)
            lines.append(("Q: salir", (200, 200, 200)))
            overlay_lines(frame, lines)
            cv2.imshow("FatigueGuard Mining - MVP", frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
                break
    finally:
        event_logger.close_open_event()
        detector.close()
        capture.release()
        cv2.destroyAllWindows()


def run_self_test() -> None:
    artifact = load_model_artifact()
    assert artifact["model_name"] == "Random Forest"
    synthetic_calibration = [
        {
            "valid_face": True, "ear_left": 0.26, "ear_right": 0.25,
            "ear_mean": 0.255 + 0.005 * np.sin(index), "mar": 0.02,
            "pitch": 170.0, "yaw": 2.0, "roll": -1.0,
        }
        for index in range(150)
    ]
    baseline = calculate_calibration_baseline(synthetic_calibration)
    normalized = []
    for index in range(50):
        record = add_relative_measurements(synthetic_calibration[index % 150], baseline)
        record["timestamp"] = index / PROCESSING_FPS
        normalized.append(record)
    features = calculate_window_features(normalized, PROCESSING_FPS)
    assert list(features) == list(artifact["feature_columns"]) == FEATURE_COLUMNS
    frame = pd.DataFrame([features], columns=FEATURE_COLUMNS)
    probability = float(artifact["model"].predict_proba(frame)[0, 1])
    assert 0.0 <= probability <= 1.0

    tracker = EyeClosureTracker(4.0)
    closed_record = {
        "valid_face": True, "yaw_delta": 0.0,
        "ear_left": baseline.ear_base * 0.5,
        "ear_right": baseline.ear_base * 0.5,
        "ear_base": baseline.ear_base,
    }
    assert not tracker.update(closed_record, 0.0)
    assert tracker.update(closed_record, 4.1)
    open_record = dict(closed_record, ear_left=baseline.ear_base, ear_right=baseline.ear_base)
    assert not tracker.update(open_record, 4.2) and tracker.duration_seconds == 0.0
    assert classify_decision_history([True, False, True, False, True]) == ("ALERTA", 3)

    with tempfile.TemporaryDirectory() as temporary_directory:
        database_path = Path(temporary_directory) / "fatigueguard.db"
        logger = EventLogger("self-test", "EQUIPO_TEST", database_path)
        logger.update(True, ["MODEL_PERSISTENCE"], probability, 0.4, 0.0, True, wall_time=100.0)
        logger.update(False, [], probability, 0.2, 0.0, False, wall_time=103.0)
        rows = get_recent_events(db_path=database_path)
        assert len(rows) == 1 and rows[0]["alert_reason"] == "MODEL_PERSISTENCE"
    landmarker = build_face_landmarker()
    landmarker.close()
    print("SELF-TEST OK")
    print(f"Modelo: {artifact['model_name']} | threshold: {artifact['threshold']}")
    print(f"Features verificadas: {len(FEATURE_COLUMNS)} | probabilidad sintética: {probability:.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FatigueGuard Mining webcam MVP")
    parser.add_argument("--camera", type=int, default=0, help="Índice OpenCV de la webcam")
    parser.add_argument("--equipment-id", default=DEFAULT_EQUIPMENT_ID, help="Identificador del equipo")
    parser.add_argument("--session-id", default=None, help="Identificador opcional de la sesión")
    parser.add_argument("--database", type=Path, default=DATABASE_PATH, help="Ruta de SQLite")
    parser.add_argument("--debug", action="store_true", help="Mostrar landmarks durante desarrollo")
    parser.add_argument("--self-test", action="store_true", help="Verificación técnica sin webcam")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.self_test:
        run_self_test()
    else:
        run_realtime(
            camera_index=arguments.camera,
            debug=arguments.debug,
            equipment_id=arguments.equipment_id,
            session_id=arguments.session_id,
            db_path=arguments.database,
        )
