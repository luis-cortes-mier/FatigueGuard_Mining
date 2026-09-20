"""Aplicación realtime de FatigueGuard Mining.

Detecta patrones visuales asociados con posible somnolencia. Es un prototipo académico para
pruebas sentado frente a un computador, no un sistema de diagnóstico médico.
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
        DEFAULT_EQUIPMENT_ID, EYE_RECOVERY_SECONDS, EYE_STATE_CONFIRMATION_FRAMES,
        INFERENCE_INTERVAL_SECONDS, MAX_ABS_YAW_FOR_EYE_RULE,
        MEDIAPIPE_MODEL_PATH, MIN_BUFFER_FACE_RATE, MIN_BUFFER_SECONDS,
        MIN_CALIBRATION_VALID_FRAMES, MODEL_ALERT_SECONDS, MODEL_PATH,
        MODEL_THRESHOLD, MPL_CONFIG_DIR, PROCESSING_FPS,
        REALTIME_EYE_CLOSED_THRESHOLD, REALTIME_EYE_OPEN_THRESHOLD,
        SIGNAL_UNSTABLE_SECONDS, WINDOW_SECONDS,
    )
except ImportError:  # Permite ejecutar el archivo directamente desde src/.
    from config import (
        ALERT_COOLDOWN_SECONDS, CALIBRATION_SECONDS, DATABASE_PATH,
        DEFAULT_EQUIPMENT_ID, EYE_RECOVERY_SECONDS, EYE_STATE_CONFIRMATION_FRAMES,
        INFERENCE_INTERVAL_SECONDS, MAX_ABS_YAW_FOR_EYE_RULE,
        MEDIAPIPE_MODEL_PATH, MIN_BUFFER_FACE_RATE, MIN_BUFFER_SECONDS,
        MIN_CALIBRATION_VALID_FRAMES, MODEL_ALERT_SECONDS, MODEL_PATH,
        MODEL_THRESHOLD, MPL_CONFIG_DIR, PROCESSING_FPS,
        REALTIME_EYE_CLOSED_THRESHOLD, REALTIME_EYE_OPEN_THRESHOLD,
        SIGNAL_UNSTABLE_SECONDS, WINDOW_SECONDS,
    )

# MediaPipe usa matplotlib internamente; su caché se guarda dentro del proyecto.
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
        FEATURE_COLUMNS, LEFT_EYE, RIGHT_EYE, CalibrationBaseline,
        add_relative_measurements, calculate_calibration_baseline,
        calculate_window_features, invalid_measurement, measurements_from_landmarks,
    )
except ImportError:  # Permite ejecutar el archivo directamente desde src/.
    from database import get_recent_events, initialize_database, insert_event, upsert_current_status
    from facial_features import (
        FEATURE_COLUMNS, LEFT_EYE, RIGHT_EYE, CalibrationBaseline,
        add_relative_measurements, calculate_calibration_baseline,
        calculate_window_features, invalid_measurement, measurements_from_landmarks,
    )


class EyeClosureTracker:
    """Confirma el estado ocular y mide el cierre con fines diagnósticos."""

    def __init__(self, confirmation_frames: int = EYE_STATE_CONFIRMATION_FRAMES):
        self.confirmation_frames = confirmation_frames
        self._closure_start: float | None = None
        self.duration_seconds = 0.0
        self.closed_frames = 0
        self.open_frames = 0
        self.confirmed_state: str | None = None

    def update(self, record: Mapping, now: float) -> str | None:
        candidate = current_eye_state(record)
        if candidate == "CLOSED":
            self.closed_frames += 1
            self.open_frames = 0
            self.confirmed_state = (
                "CLOSED" if self.closed_frames >= self.confirmation_frames else None
            )
        elif candidate == "OPEN":
            self.open_frames += 1
            self.closed_frames = 0
            self.confirmed_state = (
                "OPEN" if self.open_frames >= self.confirmation_frames else None
            )
        else:
            self.closed_frames = 0
            self.open_frames = 0
            self.confirmed_state = None

        if self.confirmed_state == "CLOSED":
            if self._closure_start is None:
                self._closure_start = now
            self.duration_seconds = max(0.0, now - self._closure_start)
        else:
            # Una muestra dudosa corta la continuidad; los parpadeos no se acumulan.
            self._closure_start = None
            self.duration_seconds = 0.0
        return self.confirmed_state


def eye_relative_values(record: Mapping) -> tuple[float | None, float | None]:
    """Devuelve los EAR relativos usados en el diagnóstico ocular."""
    valid = bool(record.get("valid_face", False))
    yaw_delta = record.get("yaw_delta", np.nan)
    ear_base = record.get("ear_base", np.nan)
    if (
        not valid
        or not np.isfinite(yaw_delta)
        or abs(float(yaw_delta)) > MAX_ABS_YAW_FOR_EYE_RULE
        or not np.isfinite(ear_base)
        or float(ear_base) <= 0
    ):
        return None, None

    def relative_value(name: str) -> float | None:
        value = record.get(name, np.nan)
        if not np.isfinite(value):
            return None
        return float(value) / float(ear_base)

    return relative_value("ear_left"), relative_value("ear_right")


def current_eye_state(record: Mapping) -> str | None:
    """Clasifica la muestra dejando una zona neutra entre ambos umbrales."""
    left_relative, right_relative = eye_relative_values(record)
    if left_relative is None or right_relative is None:
        return None
    left_closed = left_relative < REALTIME_EYE_CLOSED_THRESHOLD
    right_closed = right_relative < REALTIME_EYE_CLOSED_THRESHOLD
    left_open = left_relative > REALTIME_EYE_OPEN_THRESHOLD
    right_open = right_relative > REALTIME_EYE_OPEN_THRESHOLD
    if left_closed and right_closed:
        return "CLOSED"
    if left_open and right_open:
        return "OPEN"
    return None


def update_eye_recovery(
    eye_state: str | None,
    open_since: float | None,
    now: float,
) -> tuple[float | None, bool]:
    """Confirma la recuperación sin modificar la probabilidad del modelo."""
    if eye_state != "OPEN":
        return None, False
    if open_since is None:
        open_since = now
    return open_since, now - open_since >= EYE_RECOVERY_SECONDS


def update_facial_signal(
    signal_sufficient: bool,
    current_signal: str,
    unstable_since: float | None,
    now: float,
) -> tuple[str, float | None]:
    """Da dos segundos de tolerancia visual antes de mostrar señal inestable."""
    if signal_sufficient:
        return "BUENA", None
    if unstable_since is None:
        unstable_since = now
    if current_signal == "BUENA" and now - unstable_since < SIGNAL_UNSTABLE_SECONDS:
        return "BUENA", unstable_since
    return "INESTABLE", unstable_since


class EventLogger:
    """Agrupa estados de somnolencia posible y guarda un episodio en SQLite."""

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


def update_debug_mode(debug_enabled: bool, key: int) -> bool:
    """Activa el diagnóstico visual sin cambiar el estado analítico."""
    if key in (ord("d"), ord("D")):
        return not debug_enabled
    return debug_enabled


def _debug_number(value: float | None) -> str:
    if value is None or not np.isfinite(value):
        return "--"
    return f"{float(value):.3f}"


def _debug_probability(value: float | None) -> str:
    if value is None or not np.isfinite(value):
        return "--"
    return f"{float(value) * 100:.1f}%"


def _eye_value_state(relative: float | None) -> str:
    if relative is None:
        return "--"
    if relative < REALTIME_EYE_CLOSED_THRESHOLD:
        return "CERRADO"
    if relative > REALTIME_EYE_OPEN_THRESHOLD:
        return "ABIERTO"
    return "--"


def _draw_debug_section(frame, x: int, y: int, title: str, rows: list[str], accent) -> int:
    cv2.putText(
        frame, title, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
        0.60, accent, 2, cv2.LINE_AA,
    )
    cv2.line(frame, (x, y + 9), (frame.shape[1] - 18, y + 9), (70, 78, 88), 1)
    y += 25
    for row in rows:
        cv2.putText(
            frame, row, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
            0.48, (235, 238, 242), 1, cv2.LINE_AA,
        )
        y += 20
    return y


def draw_eye_debug(
    frame,
    landmarks,
    record: Mapping,
    closure_seconds: float,
    probability: float | None,
    persistence_seconds: float,
    recovery_active: bool,
    operational_state: str,
    alert_origin: str,
) -> np.ndarray:
    """Dibuja los datos de diagnóstico sin tapar el rostro."""
    left_relative, right_relative = eye_relative_values(record)
    eye_data = (
        ("OI", LEFT_EYE, "ear_left", left_relative, (255, 210, 0)),
        ("OD", RIGHT_EYE, "ear_right", right_relative, (255, 120, 255)),
    )
    video = frame.copy()
    height, width = video.shape[:2]
    eye_points_by_label = {}

    if landmarks is not None:
        for landmark in landmarks:
            point = (int(landmark.x * width), int(landmark.y * height))
            cv2.circle(video, point, 1, (120, 145, 165), -1)

    for label, indices, _, relative, point_color in eye_data:
        status = _eye_value_state(relative)
        status_color = {
            "ABIERTO": (0, 220, 0),
            "CERRADO": (0, 0, 255),
            "--": (160, 160, 160),
        }[status]

        if landmarks is not None and len(landmarks) > max(indices):
            points = [
                (int(landmarks[index].x * width), int(landmarks[index].y * height))
                for index in indices
            ]
            eye_points_by_label[label] = points
            cv2.polylines(video, [np.asarray(points, dtype=np.int32)], True, point_color, 1)
            for point in points:
                cv2.circle(video, point, 3, point_color, -1)
            x_values = [point[0] for point in points]
            y_values = [point[1] for point in points]
            top_left = (max(0, min(x_values) - 7), max(0, min(y_values) - 7))
            bottom_right = (min(width - 1, max(x_values) + 7), min(height - 1, max(y_values) + 7))
            cv2.rectangle(video, top_left, bottom_right, status_color, 2)
            text_x = max(6, top_left[0] - 72) if label == "OI" else min(width - 78, bottom_right[0] + 8)
            text_y = max(18, top_left[1] - 23)
            short_lines = (label, f"rel: {_debug_number(relative)}", status)
            for line in short_lines:
                cv2.putText(
                    video, line, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.44, (15, 15, 15), 3, cv2.LINE_AA,
                )
                cv2.putText(
                    video, line, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.44, status_color, 1, cv2.LINE_AA,
                )
                text_y += 16

    all_eye_points = [point for points in eye_points_by_label.values() for point in points]
    if all_eye_points:
        center_x = int(np.mean([point[0] for point in all_eye_points]))
        top_y = min(point[1] for point in all_eye_points)
        closure_text = f"CIERRE: {closure_seconds:.1f} s"
        text_size = cv2.getTextSize(closure_text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)[0]
        text_position = (max(5, center_x - text_size[0] // 2), max(24, top_y - 32))
        cv2.putText(
            video, closure_text, text_position, cv2.FONT_HERSHEY_SIMPLEX,
            0.65, (15, 15, 15), 4, cv2.LINE_AA,
        )
        cv2.putText(
            video, closure_text, text_position, cv2.FONT_HERSHEY_SIMPLEX,
            0.65, (255, 255, 255), 2, cv2.LINE_AA,
        )

    ear_base = record.get("ear_base", np.nan)
    panel_width = max(340, int(width * 0.45))
    canvas = np.zeros((height, width + panel_width, 3), dtype=video.dtype)
    canvas[:, :width] = video
    canvas[:, width:] = (22, 26, 32)
    cv2.line(canvas, (width, 0), (width, height - 1), (95, 105, 118), 2)

    panel_x = width + 22
    cv2.putText(
        canvas, "DIAGNOSTICO OCULAR", (panel_x, 32), cv2.FONT_HERSHEY_SIMPLEX,
        0.66, (255, 255, 255), 2, cv2.LINE_AA,
    )
    y = 60
    for label, _, raw_name, relative, accent in eye_data:
        y = _draw_debug_section(
            canvas, panel_x, y, label,
            [
                f"EAR raw: {_debug_number(record.get(raw_name))}",
                f"EAR rel.: {_debug_number(relative)}",
                f"Estado: {_eye_value_state(relative)}",
            ],
            accent,
        )
        y += 4

    y = _draw_debug_section(
        canvas, panel_x, y, "GENERAL",
        [
            f"EAR base: {_debug_number(ear_base)}",
            f"Cerrado: < {REALTIME_EYE_CLOSED_THRESHOLD:.2f}",
            f"Abierto: > {REALTIME_EYE_OPEN_THRESHOLD:.2f}",
            f"Cierre actual: {closure_seconds:.1f} s",
        ],
        (255, 255, 255),
    )
    y += 4
    _draw_debug_section(
        canvas, panel_x, y, "MODELO",
        [
            f"P RF ultimos 10 s: {_debug_probability(probability)}",
            f"Threshold RF: {MODEL_THRESHOLD:.2f}",
            f"Persistencia RF: {persistence_seconds:.1f} / {MODEL_ALERT_SECONDS:.1f} s",
            f"Recuperacion ocular: {'ACTIVA' if recovery_active else 'NO ACTIVA'}",
            f"Estado operativo: {operational_state}",
            f"Origen alerta: {alert_origin}",
        ],
        (0, 190, 255),
    )
    return canvas


def determine_alert_origin(model_alert: bool) -> str:
    """El Random Forest es la única fuente de alerta."""
    return "MODELO RF" if model_alert else "--"


def model_persistence_seconds(positive_since: float | None, now: float) -> float:
    if positive_since is None:
        return 0.0
    return min(MODEL_ALERT_SECONDS, max(0.0, now - positive_since))


def classify_model_probability(
    probability: float | None,
    positive_since: float | None,
    now: float,
    persistence_blocked: bool = False,
) -> tuple[str, float | None]:
    """Mide cuánto tiempo lleva positiva la salida del modelo."""
    if probability is None or probability < MODEL_THRESHOLD:
        return "NORMAL", None
    if persistence_blocked:
        return "NORMAL", None
    if positive_since is None:
        positive_since = now
    if now - positive_since >= MODEL_ALERT_SECONDS:
        return "ALERTA", positive_since
    return "POSIBLE SOMNOLENCIA", positive_since


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
    eye_tracker = EyeClosureTracker()
    sound = AlertSound()
    event_logger = EventLogger(session_id, equipment_id, db_path)

    application_start = time.perf_counter()
    last_processing_time = -np.inf
    last_inference_time = -np.inf
    last_probability: float | None = None
    last_features: dict | None = None
    model_positive_since: float | None = None
    open_eyes_since: float | None = None
    recovery_active = False
    current_state = "CALIBRANDO"
    alert_reasons: list[str] = []
    alert_origin = "--"
    facial_signal = "INESTABLE"
    signal_unstable_since: float | None = None
    eye_state: str | None = None
    has_model_inference = False
    analysis_progress = 0.0
    buffer_ready = False
    debug_enabled = bool(debug)
    debug_landmarks = None
    latest_record = invalid_measurement()

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
                else:
                    debug_landmarks = None
                    record = invalid_measurement()
                record["timestamp"] = now

                if baseline is None:
                    if record["valid_face"]:
                        calibration_records.append(record)
                    elapsed = now - application_start
                    if elapsed >= CALIBRATION_SECONDS and len(calibration_records) >= MIN_CALIBRATION_VALID_FRAMES:
                        baseline = calculate_calibration_baseline(calibration_records)
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

                    eye_state = eye_tracker.update(record, now)
                    open_eyes_since, recovery_active = update_eye_recovery(
                        eye_state, open_eyes_since, now
                    )
                    buffer_span = now - float(buffer[0]["timestamp"]) if buffer else 0.0
                    valid_rate = np.mean([bool(item["valid_face"]) for item in buffer]) if buffer else 0.0
                    minimum_buffer_span = MIN_BUFFER_SECONDS - 1.0 / PROCESSING_FPS
                    buffer_ready = buffer_span + 1e-9 >= minimum_buffer_span
                    progress_ratio = min(1.0, buffer_span / minimum_buffer_span)
                    analysis_progress = np.floor(progress_ratio * WINDOW_SECONDS * 10.0) / 10.0
                    enough_buffer = (
                        buffer_ready
                        and valid_rate >= MIN_BUFFER_FACE_RATE
                    )
                    new_inference = False
                    if (
                        record["valid_face"]
                        and enough_buffer
                        and now - last_inference_time >= INFERENCE_INTERVAL_SECONDS
                    ):
                        candidate_features = calculate_window_features(list(buffer), PROCESSING_FPS)
                        feature_frame = pd.DataFrame([candidate_features], columns=expected_features)
                        if list(feature_frame.columns) != expected_features:
                            raise RuntimeError("Orden de features incompatible con el modelo")
                        if not np.isfinite(feature_frame.to_numpy(dtype=float)).all():
                            last_probability = None
                            enough_buffer = False
                        else:
                            last_probability = float(model.predict_proba(feature_frame)[0, 1])
                            last_features = candidate_features
                            last_inference_time = now
                            new_inference = True
                            has_model_inference = True

                    facial_signal, signal_unstable_since = update_facial_signal(
                        enough_buffer and bool(record["valid_face"]),
                        facial_signal,
                        signal_unstable_since,
                        now,
                    )

                    model_state, model_positive_since = classify_model_probability(
                        last_probability, model_positive_since, now,
                        persistence_blocked=recovery_active,
                    )
                    model_alert = model_state == "ALERTA"
                    alert_reasons = []
                    alert_origin = "--"
                    if not record["valid_face"]:
                        # Sin rostro válido no se conserva ni acumula una decisión.
                        model_positive_since = None
                        last_probability = None
                        last_features = None
                        current_state = "NORMAL"
                    elif not enough_buffer:
                        model_positive_since = None
                        last_probability = None
                        last_features = None
                        current_state = "NORMAL"
                    elif model_alert:
                        alert_reasons.append("MODEL_PERSISTENCE")
                        alert_origin = determine_alert_origin(model_alert)
                        current_state = "ALERTA"
                        sound.play_if_allowed(now)
                    elif model_state == "POSIBLE SOMNOLENCIA":
                        current_state = model_state
                    else:
                        current_state = "NORMAL"

                    active_episode = current_state in {"POSIBLE SOMNOLENCIA", "ALERTA"}
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
                upsert_current_status(
                    session_id=session_id,
                    equipment_id=equipment_id,
                    status=persisted_state,
                    probability=last_probability,
                    positive_windows=0,
                    db_path=db_path,
                )
                latest_record = record

            if baseline is None:
                elapsed = time.perf_counter() - application_start
                lines = [
                    ("CALIBRANDO - Mantenga posicion normal y permanezca alerta", (0, 220, 255)),
                    (f"Tiempo: {elapsed:.1f}/{CALIBRATION_SECONDS}s | muestras faciales validas: {len(calibration_records)}", (255, 255, 255)),
                ]
            else:
                display_state = current_state
                if not has_model_inference and not buffer_ready and not alert_reasons:
                    display_state = "PREPARANDO ANALISIS"
                color = (0, 0, 255) if alert_reasons else ((0, 165, 255) if current_state == "POSIBLE SOMNOLENCIA" else (0, 220, 0))
                lines = [(display_state, color)]
                eyes_text = {"OPEN": "ABIERTOS", "CLOSED": "CERRADOS"}.get(eye_state, "--")
                lines.extend([
                    (f"Ojos: {eyes_text}", (255, 255, 255)),
                    (f"Senal facial: {facial_signal}", (255, 255, 255)),
                    (f"Cierre ocular: {eye_tracker.duration_seconds:.1f}s", (255, 255, 255)),
                    (f"Origen alerta: {alert_origin}", (255, 255, 255)),
                ])
                if display_state == "PREPARANDO ANALISIS":
                    lines.append((f"{analysis_progress:.1f} / {WINDOW_SECONDS} s", (0, 220, 255)))
            debug_hint = "D: ocultar diagnostico" if debug_enabled else "D: diagnostico"
            lines.append((f"{debug_hint} | Q: salir", (200, 200, 200)))
            overlay_lines(frame, lines)
            display_frame = frame
            if debug_enabled:
                display_frame = draw_eye_debug(
                    frame, debug_landmarks, latest_record,
                    eye_tracker.duration_seconds, last_probability,
                    model_persistence_seconds(model_positive_since, now),
                    recovery_active, current_state, alert_origin,
                )
            cv2.imshow("FatigueGuard Mining - MVP", display_frame)
            key = cv2.waitKey(1) & 0xFF
            debug_enabled = update_debug_mode(debug_enabled, key)
            if key in (ord("q"), ord("Q")):
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

    tracker = EyeClosureTracker()
    closed_record = {
        "valid_face": True, "yaw_delta": 0.0,
        "ear_left": baseline.ear_base * 0.5,
        "ear_right": baseline.ear_base * 0.5,
        "ear_base": baseline.ear_base,
    }
    assert tracker.update(closed_record, 0.0) is None
    assert tracker.update(closed_record, 0.2) is None
    assert tracker.update(closed_record, 0.4) == "CLOSED"
    assert tracker.update(closed_record, 4.4) == "CLOSED"
    assert tracker.duration_seconds >= 4.0
    open_record = dict(closed_record, ear_left=baseline.ear_base, ear_right=baseline.ear_base)
    assert tracker.update(open_record, 4.6) is None and tracker.duration_seconds == 0.0
    model_state, positive_since = classify_model_probability(MODEL_THRESHOLD, None, 0.0)
    assert model_state == "POSIBLE SOMNOLENCIA" and positive_since == 0.0
    model_state, positive_since = classify_model_probability(
        MODEL_THRESHOLD, positive_since, MODEL_ALERT_SECONDS
    )
    assert model_state == "ALERTA"
    assert classify_model_probability(0.0, positive_since, MODEL_ALERT_SECONDS + 0.1) == (
        "NORMAL", None
    )

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
    parser.add_argument(
        "--debug", action="store_true",
        help="Iniciar con el diagnostico ocular visible; D permite alternarlo",
    )
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
