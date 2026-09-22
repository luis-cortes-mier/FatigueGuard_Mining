"""Mediciones faciales y variables temporales usadas por FatigueGuard.

Las fórmulas y los nombres de columnas coinciden con el notebook 08.
Este módulo no contiene decisiones propias del modelo.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, Mapping, Sequence

import cv2
import numpy as np


TARGET_FPS = 5.0
EAR_PERCLOS_THRESHOLD = 0.80
EAR_STRONG_REDUCTION_THRESHOLD = 0.60
MAR_HIGH_THRESHOLD = 1.50
ABRUPT_PITCH_DEGREES = 10.0

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH_VERTICAL_PAIRS = [(13, 14), (82, 87), (312, 317)]
MOUTH_CORNERS = (78, 308)
POSE_INDICES = [1, 152, 33, 263, 61, 291]
MODEL_POINTS = np.array(
    [
        (0.0, 0.0, 0.0),
        (0.0, -63.6, -12.5),
        (-43.3, 32.7, -26.0),
        (43.3, 32.7, -26.0),
        (-28.9, -28.9, -24.1),
        (28.9, -28.9, -24.1),
    ],
    dtype=np.float64,
)

FEATURE_COLUMNS = [
    "n_sampled_frames",
    "n_valid_face",
    "face_detection_rate",
    "ear_valid_rate",
    "mar_valid_rate",
    "ear_relative_mean",
    "ear_relative_std",
    "ear_relative_min",
    "ear_relative_p10",
    "strong_ear_reduction_rate",
    "perclos_relative",
    "max_eye_closure_seconds",
    "eye_closure_episode_count",
    "mar_relative_mean",
    "mar_relative_std",
    "mar_relative_max",
    "high_mouth_open_rate",
    "pitch_delta_mean",
    "pitch_delta_std",
    "pitch_delta_range",
    "yaw_delta_std",
    "roll_delta_std",
    "max_head_drop_delta",
    "abrupt_pitch_movement_count",
]


@dataclass(frozen=True)
class CalibrationBaseline:
    ear_base: float
    mar_base: float
    pitch_base: float
    yaw_base: float
    roll_base: float
    n_calibration_frames: int

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def _finite(values: Iterable[float | None]) -> np.ndarray:
    array = np.asarray([np.nan if value is None else value for value in values], dtype=float)
    return array[np.isfinite(array)]


def wrap_angle_degrees(values):
    """Lleva cada ángulo al intervalo [-180, 180)."""
    return (np.asarray(values) + 180.0) % 360.0 - 180.0


def robust_circular_center(values: Iterable[float | None]) -> float:
    valid = _finite(values)
    if len(valid) == 0:
        return np.nan
    radians = np.deg2rad(valid)
    initial = np.rad2deg(np.arctan2(np.mean(np.sin(radians)), np.mean(np.cos(radians))))
    centered = wrap_angle_degrees(valid - initial)
    return float(wrap_angle_degrees(initial + np.median(centered)))


def calculate_calibration_baseline(
    records: Sequence[Mapping[str, float | bool | None]],
) -> CalibrationBaseline:
    """Calcula la referencia personal con las muestras válidas de calibración."""
    valid_records = [record for record in records if bool(record.get("valid_face", False))]
    ear = _finite(record.get("ear_mean") for record in valid_records)
    mar = _finite(record.get("mar") for record in valid_records)
    if len(ear) == 0 or len(mar) == 0:
        raise ValueError("Calibration has no valid EAR/MAR measurements")
    baseline = CalibrationBaseline(
        ear_base=float(np.quantile(ear, 0.90)),
        mar_base=float(np.median(mar)),
        pitch_base=robust_circular_center(record.get("pitch") for record in valid_records),
        yaw_base=robust_circular_center(record.get("yaw") for record in valid_records),
        roll_base=robust_circular_center(record.get("roll") for record in valid_records),
        n_calibration_frames=len(valid_records),
    )
    numeric = np.asarray(
        [baseline.ear_base, baseline.mar_base, baseline.pitch_base,
         baseline.yaw_base, baseline.roll_base],
        dtype=float,
    )
    if not np.isfinite(numeric).all() or baseline.ear_base <= 0 or baseline.mar_base <= 0:
        raise ValueError("Calibration produced an invalid baseline")
    return baseline


def _distance(points: np.ndarray, index_a: int, index_b: int) -> float:
    return float(np.linalg.norm(points[index_a] - points[index_b]))


def eye_aspect_ratio(points: np.ndarray, indices: Sequence[int]) -> float:
    p1, p2, p3, p4, p5, p6 = indices
    horizontal = _distance(points, p1, p4)
    if horizontal <= 1e-8:
        return np.nan
    return (
        _distance(points, p2, p6) + _distance(points, p3, p5)
    ) / (2.0 * horizontal)


def mouth_aspect_ratio(points: np.ndarray) -> float:
    width = _distance(points, *MOUTH_CORNERS)
    if width <= 1e-8:
        return np.nan
    vertical = np.mean([_distance(points, a, b) for a, b in MOUTH_VERTICAL_PAIRS])
    return float(vertical / width)


def estimate_head_pose(points: np.ndarray, width: int, height: int) -> tuple[float, float, float]:
    image_points = np.array(
        [(points[index, 0] * width, points[index, 1] * height) for index in POSE_INDICES],
        dtype=np.float64,
    )
    focal_length = float(width)
    camera_matrix = np.array(
        [[focal_length, 0, width / 2], [0, focal_length, height / 2], [0, 0, 1]],
        dtype=np.float64,
    )
    success, rotation_vector, _ = cv2.solvePnP(
        MODEL_POINTS,
        image_points,
        camera_matrix,
        np.zeros((4, 1)),
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not success:
        return np.nan, np.nan, np.nan
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    angles = cv2.RQDecomp3x3(rotation_matrix)[0]
    return float(angles[0]), float(angles[1]), float(angles[2])


def measurements_from_landmarks(
    landmarks: Sequence,
    frame_width: int,
    frame_height: int,
) -> dict[str, float | bool]:
    """Convierte los 478 landmarks de MediaPipe en las mediciones del entrenamiento."""
    points = np.array([(item.x, item.y, item.z) for item in landmarks], dtype=np.float64)
    ear_left = eye_aspect_ratio(points, LEFT_EYE)
    ear_right = eye_aspect_ratio(points, RIGHT_EYE)
    pitch, yaw, roll = estimate_head_pose(points, frame_width, frame_height)
    return {
        "valid_face": True,
        "ear_left": ear_left,
        "ear_right": ear_right,
        "ear_mean": float(np.nanmean([ear_left, ear_right])),
        "mar": mouth_aspect_ratio(points),
        "pitch": pitch,
        "yaw": yaw,
        "roll": roll,
    }


def invalid_measurement() -> dict[str, float | bool]:
    return {
        "valid_face": False,
        "ear_left": np.nan,
        "ear_right": np.nan,
        "ear_mean": np.nan,
        "mar": np.nan,
        "pitch": np.nan,
        "yaw": np.nan,
        "roll": np.nan,
    }


def add_relative_measurements(
    record: Mapping[str, float | bool | None],
    baseline: CalibrationBaseline,
) -> dict[str, float | bool | None]:
    def finite_value(name: str) -> bool:
        value = record.get(name)
        return value is not None and bool(np.isfinite(value))

    result = dict(record)
    result["ear_relative"] = (
        float(record["ear_mean"]) / baseline.ear_base if finite_value("ear_mean") else np.nan
    )
    result["mar_relative"] = (
        float(record["mar"]) / baseline.mar_base if finite_value("mar") else np.nan
    )
    result["pitch_delta"] = (
        float(wrap_angle_degrees(float(record["pitch"]) - baseline.pitch_base))
        if finite_value("pitch") else np.nan
    )
    result["yaw_delta"] = (
        float(wrap_angle_degrees(float(record["yaw"]) - baseline.yaw_base))
        if finite_value("yaw") else np.nan
    )
    result["roll_delta"] = (
        float(wrap_angle_degrees(float(record["roll"]) - baseline.roll_base))
        if finite_value("roll") else np.nan
    )
    return result


def _longest_true_run(values: np.ndarray) -> int:
    longest = current = 0
    for value in values:
        current = current + 1 if bool(value) else 0
        longest = max(longest, current)
    return longest


def _episode_count(values: np.ndarray) -> int:
    array = np.asarray(values, dtype=bool)
    if len(array) == 0:
        return 0
    return int(array[0]) + int(np.sum(array[1:] & ~array[:-1]))


def _stat(values: np.ndarray, function) -> float:
    valid = values[np.isfinite(values)]
    return float(function(valid)) if len(valid) else np.nan


def calculate_window_features(
    records: Sequence[Mapping[str, float | bool | None]],
    target_fps: float = TARGET_FPS,
) -> dict[str, float | int]:
    """Calcula las 24 variables usadas para entrenar el Random Forest."""
    if not records:
        raise ValueError("A window needs at least one record")

    valid_face = np.asarray([bool(record.get("valid_face", False)) for record in records])
    ear = np.asarray([record.get("ear_relative", np.nan) for record in records], dtype=float)
    mar = np.asarray([record.get("mar_relative", np.nan) for record in records], dtype=float)
    pitch = np.asarray([record.get("pitch_delta", np.nan) for record in records], dtype=float)
    yaw = np.asarray([record.get("yaw_delta", np.nan) for record in records], dtype=float)
    roll = np.asarray([record.get("roll_delta", np.nan) for record in records], dtype=float)
    ear_valid = np.isfinite(ear)
    mar_valid = np.isfinite(mar)
    closed = ear_valid & (ear < EAR_PERCLOS_THRESHOLD)
    strong_reduction = ear_valid & (ear < EAR_STRONG_REDUCTION_THRESHOLD)
    elevated_mouth = mar_valid & (mar > MAR_HIGH_THRESHOLD)
    pitch_differences = np.abs(np.diff(pitch))

    features = {
        "n_sampled_frames": len(records),
        "n_valid_face": int(valid_face.sum()),
        "face_detection_rate": float(valid_face.mean()),
        "ear_valid_rate": float(ear_valid.mean()),
        "mar_valid_rate": float(mar_valid.mean()),
        "ear_relative_mean": _stat(ear, np.mean),
        "ear_relative_std": _stat(ear, lambda x: np.std(x, ddof=0)),
        "ear_relative_min": _stat(ear, np.min),
        "ear_relative_p10": _stat(ear, lambda x: np.quantile(x, 0.10)),
        "strong_ear_reduction_rate": float(strong_reduction[ear_valid].mean()) if ear_valid.any() else np.nan,
        "perclos_relative": float(closed[ear_valid].mean()) if ear_valid.any() else np.nan,
        "max_eye_closure_seconds": _longest_true_run(closed) / target_fps,
        "eye_closure_episode_count": _episode_count(closed),
        "mar_relative_mean": _stat(mar, np.mean),
        "mar_relative_std": _stat(mar, lambda x: np.std(x, ddof=0)),
        "mar_relative_max": _stat(mar, np.max),
        "high_mouth_open_rate": float(elevated_mouth[mar_valid].mean()) if mar_valid.any() else np.nan,
        "pitch_delta_mean": _stat(pitch, np.mean),
        "pitch_delta_std": _stat(pitch, lambda x: np.std(x, ddof=0)),
        "pitch_delta_range": _stat(pitch, lambda x: np.max(x) - np.min(x)),
        "yaw_delta_std": _stat(yaw, lambda x: np.std(x, ddof=0)),
        "roll_delta_std": _stat(roll, lambda x: np.std(x, ddof=0)),
        "max_head_drop_delta": _stat(pitch, np.max),
        "abrupt_pitch_movement_count": int(np.sum(pitch_differences > ABRUPT_PITCH_DEGREES)),
    }
    if list(features) != FEATURE_COLUMNS:
        raise AssertionError("Feature order changed")
    return features
