from __future__ import annotations

import re
import hashlib
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from dashboard.app import load_dashboard_data
from src.config import (
    CALIBRATION_SECONDS,
    DATABASE_PATH,
    EYE_RECOVERY_SECONDS,
    EYE_STATE_CONFIRMATION_FRAMES,
    INFERENCE_INTERVAL_SECONDS,
    MEDIAPIPE_MODEL_PATH,
    MIN_BUFFER_SECONDS,
    MODEL_ALERT_SECONDS,
    MODEL_PATH,
    MODEL_THRESHOLD,
    PROCESSING_FPS,
    PROJECT_ROOT,
    REALTIME_EYE_CLOSED_THRESHOLD,
    REALTIME_EYE_OPEN_THRESHOLD,
    SIGNAL_UNSTABLE_SECONDS,
    WINDOW_SECONDS,
)
from src.database import (
    connect_database,
    get_current_status,
    get_recent_events,
    initialize_database,
    insert_event,
    upsert_current_status,
)
from src.facial_features import (
    EAR_PERCLOS_THRESHOLD,
    FEATURE_COLUMNS,
    CalibrationBaseline,
    add_relative_measurements,
    calculate_window_features,
)
from src.realtime_fatigueguard import (
    EyeClosureTracker,
    build_face_landmarker,
    classify_model_probability,
    current_eye_state,
    determine_alert_origin,
    draw_eye_debug,
    eye_relative_values,
    load_model_artifact,
    model_persistence_seconds,
    run_realtime,
    update_debug_mode,
    update_eye_recovery,
    update_facial_signal,
)


class TestValidatedConfiguration(unittest.TestCase):
    def test_validated_parameters_are_unchanged(self):
        self.assertEqual(CALIBRATION_SECONDS, 30)
        self.assertEqual(PROCESSING_FPS, 5)
        self.assertEqual(WINDOW_SECONDS, 10)
        self.assertEqual(MODEL_THRESHOLD, 0.36)
        self.assertEqual(MODEL_ALERT_SECONDS, 8.0)
        self.assertEqual(EYE_RECOVERY_SECONDS, 2.0)
        self.assertEqual(SIGNAL_UNSTABLE_SECONDS, 2.0)
        self.assertEqual(REALTIME_EYE_CLOSED_THRESHOLD, 0.75)
        self.assertEqual(REALTIME_EYE_OPEN_THRESHOLD, 0.85)
        self.assertEqual(EYE_STATE_CONFIRMATION_FRAMES, 3)
        self.assertEqual(INFERENCE_INTERVAL_SECONDS, 1.0)
        self.assertEqual(MIN_BUFFER_SECONDS, WINDOW_SECONDS)
        self.assertEqual(EAR_PERCLOS_THRESHOLD, 0.80)

    def test_paths_are_portable_and_inside_project(self):
        for path in (MODEL_PATH, MEDIAPIPE_MODEL_PATH, DATABASE_PATH):
            self.assertTrue(path.is_relative_to(PROJECT_ROOT))
        personal_path = re.compile(r"[A-Za-z]:[\\/]Users[\\/]")
        files = list((PROJECT_ROOT / "src").glob("*.py")) + [PROJECT_ROOT / "dashboard" / "app.py"]
        for path in files:
            self.assertIsNone(personal_path.search(path.read_text(encoding="utf-8")), path)


class TestModelArtifact(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.artifact = load_model_artifact()
        cls.model = cls.artifact["model"]

    def test_model_is_expected_random_forest(self):
        self.assertIsInstance(self.model, RandomForestClassifier)
        self.assertEqual(self.artifact["model_name"], "Random Forest")
        self.assertEqual(self.model.n_estimators, 300)

    def test_feature_schema_and_threshold(self):
        self.assertEqual(len(FEATURE_COLUMNS), 24)
        self.assertEqual(list(self.artifact["feature_columns"]), FEATURE_COLUMNS)
        self.assertEqual(list(self.model.feature_names_in_), FEATURE_COLUMNS)
        self.assertEqual(self.model.n_features_in_, 24)
        self.assertAlmostEqual(float(self.artifact["threshold"]), 0.36)

    def test_artifact_is_unchanged_by_loading(self):
        direct = joblib.load(MODEL_PATH)
        self.assertEqual(direct["model"].get_params(), self.model.get_params())
        digest = hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest().upper()
        self.assertEqual(
            digest,
            "438D31F5E630D90639DEF50BBCD3A7764170556977836CDDCCBBE610A97AAAEE",
        )


class TestRealtimeRulesAndFeatures(unittest.TestCase):
    def test_model_probability_persistence(self):
        state, started = classify_model_probability(MODEL_THRESHOLD - 0.01, 1.0, 3.0)
        self.assertEqual((state, started), ("NORMAL", None))

        state, started = classify_model_probability(MODEL_THRESHOLD, None, 10.0)
        self.assertEqual((state, started), ("POSIBLE SOMNOLENCIA", 10.0))
        state, started = classify_model_probability(MODEL_THRESHOLD, started, 17.99)
        self.assertEqual((state, started), ("POSIBLE SOMNOLENCIA", 10.0))
        state, started = classify_model_probability(MODEL_THRESHOLD, started, 18.0)
        self.assertEqual((state, started), ("ALERTA", 10.0))
        self.assertEqual(determine_alert_origin(state == "ALERTA"), "MODELO RF")

    def test_negative_probability_restarts_persistence(self):
        state, started = classify_model_probability(0.8, None, 0.0)
        state, started = classify_model_probability(0.8, started, 7.99)
        self.assertEqual(state, "POSIBLE SOMNOLENCIA")
        state, started = classify_model_probability(0.2, started, 8.0)
        self.assertEqual((state, started), ("NORMAL", None))
        state, started = classify_model_probability(0.8, started, 9.0)
        self.assertEqual((state, started), ("POSIBLE SOMNOLENCIA", 9.0))
        state, started = classify_model_probability(0.8, started, 16.99)
        self.assertEqual(state, "POSIBLE SOMNOLENCIA")
        state, started = classify_model_probability(0.8, started, 17.0)
        self.assertEqual((state, started), ("ALERTA", 9.0))

    def test_brief_confirmed_open_does_not_reset_model_persistence(self):
        probability = 0.80
        state, model_started = classify_model_probability(probability, None, 0.0)
        self.assertEqual((state, model_started), ("POSIBLE SOMNOLENCIA", 0.0))

        opened_at, recovery_active = update_eye_recovery("OPEN", None, 1.0)
        self.assertFalse(recovery_active)
        opened_at, recovery_active = update_eye_recovery("OPEN", opened_at, 2.5)
        self.assertFalse(recovery_active)
        state, model_started = classify_model_probability(
            probability, model_started, 2.5, persistence_blocked=recovery_active,
        )
        self.assertEqual((state, model_started), ("POSIBLE SOMNOLENCIA", 0.0))

        opened_at, recovery_active = update_eye_recovery("CLOSED", opened_at, 2.6)
        self.assertEqual((opened_at, recovery_active), (None, False))
        state, model_started = classify_model_probability(
            probability, model_started, 2.6, persistence_blocked=recovery_active,
        )
        self.assertEqual((state, model_started), ("POSIBLE SOMNOLENCIA", 0.0))
        self.assertEqual(probability, 0.80)

    def test_confirmed_recovery_blocks_until_open_ends(self):
        probability = 0.80
        state, model_started = classify_model_probability(probability, None, 0.0)
        self.assertEqual((state, model_started), ("POSIBLE SOMNOLENCIA", 0.0))

        opened_at, recovery_active = update_eye_recovery("OPEN", None, 1.0)
        self.assertFalse(recovery_active)
        opened_at, recovery_active = update_eye_recovery("OPEN", opened_at, 3.0)
        self.assertTrue(recovery_active)
        state, model_started = classify_model_probability(
            probability, model_started, 3.0, persistence_blocked=recovery_active,
        )
        self.assertEqual((state, model_started), ("NORMAL", None))

        opened_at, recovery_active = update_eye_recovery("OPEN", opened_at, 7.0)
        self.assertTrue(recovery_active)
        state, model_started = classify_model_probability(
            probability, model_started, 7.0, persistence_blocked=recovery_active,
        )
        self.assertEqual((state, model_started), ("NORMAL", None))
        self.assertEqual(probability, 0.80)

        opened_at, recovery_active = update_eye_recovery("CLOSED", opened_at, 7.1)
        self.assertEqual((opened_at, recovery_active), (None, False))
        state, model_started = classify_model_probability(probability, model_started, 7.1)
        self.assertEqual((state, model_started), ("POSIBLE SOMNOLENCIA", 7.1))
        state, model_started = classify_model_probability(probability, model_started, 15.1)
        self.assertEqual((state, model_started), ("ALERTA", 7.1))

    def test_facial_signal_uses_two_second_visual_tolerance(self):
        signal, unstable_since = update_facial_signal(False, "BUENA", None, 10.0)
        self.assertEqual((signal, unstable_since), ("BUENA", 10.0))
        signal, unstable_since = update_facial_signal(False, signal, unstable_since, 11.99)
        self.assertEqual(signal, "BUENA")
        signal, unstable_since = update_facial_signal(False, signal, unstable_since, 12.0)
        self.assertEqual(signal, "INESTABLE")
        signal, unstable_since = update_facial_signal(True, signal, unstable_since, 12.1)
        self.assertEqual((signal, unstable_since), ("BUENA", None))

    def test_invalid_eye_data_never_confirms_recovery(self):
        invalid_records = [
            {"valid_face": False},
            {"valid_face": True, "yaw_delta": 40.0, "ear_left": 0.25,
             "ear_right": 0.25, "ear_base": 0.25},
            {"valid_face": True, "yaw_delta": 0.0, "ear_left": np.nan,
             "ear_right": 0.25, "ear_base": 0.25},
            {"valid_face": True, "yaw_delta": 0.0, "ear_left": 0.10,
             "ear_right": 0.25, "ear_base": 0.25},
        ]
        for record in invalid_records:
            with self.subTest(record=record):
                self.assertIsNone(current_eye_state(record))
                self.assertEqual(
                    update_eye_recovery(current_eye_state(record), 0.0, 3.0),
                    (None, False),
                )

    def test_eye_state_uses_two_thresholds_and_neutral_band(self):
        def record(left_relative, right_relative):
            return {
                "valid_face": True,
                "yaw_delta": 0.0,
                "ear_left": left_relative,
                "ear_right": right_relative,
                "ear_base": 1.0,
            }

        self.assertEqual(current_eye_state(record(0.74, 0.70)), "CLOSED")
        self.assertEqual(current_eye_state(record(0.86, 0.90)), "OPEN")
        self.assertIsNone(current_eye_state(record(0.75, 0.74)))
        self.assertIsNone(current_eye_state(record(0.85, 0.90)))
        self.assertIsNone(current_eye_state(record(0.70, 0.90)))

    def test_alert_origin_only_comes_from_model(self):
        self.assertEqual(determine_alert_origin(False), "--")
        self.assertEqual(determine_alert_origin(True), "MODELO RF")

    def test_eye_debug_is_visual_only(self):
        record = {
            "valid_face": True,
            "yaw_delta": 0.0,
            "ear_left": 0.25,
            "ear_right": np.nan,
            "ear_base": 0.25,
        }
        original_record = dict(record)
        probability = 0.73
        landmarks = [SimpleNamespace(x=0.5, y=0.5, z=0.0) for _ in range(478)]
        frame = np.zeros((240, 320, 3), dtype=np.uint8)

        self.assertTrue(update_debug_mode(False, ord("d")))
        self.assertFalse(update_debug_mode(True, ord("D")))
        self.assertTrue(update_debug_mode(True, ord("x")))
        self.assertEqual(eye_relative_values(record), (1.0, None))

        module = "src.realtime_fatigueguard"
        with (
            patch(f"{module}.cv2.circle") as circle,
            patch(f"{module}.cv2.rectangle") as rectangle,
            patch(f"{module}.cv2.polylines") as polylines,
            patch(f"{module}.cv2.putText") as put_text,
        ):
            canvas = draw_eye_debug(
                frame, landmarks, record, closure_seconds=1.5,
                probability=probability, persistence_seconds=2.5,
                recovery_active=True, operational_state="NORMAL",
                alert_origin="--",
            )

        self.assertEqual(circle.call_count, 490)
        self.assertEqual(rectangle.call_count, 2)
        self.assertEqual(polylines.call_count, 2)
        self.assertGreater(canvas.shape[1], frame.shape[1])
        rendered_text = " ".join(call.args[1] for call in put_text.call_args_list)
        self.assertIn("OI", rendered_text)
        self.assertIn("OD", rendered_text)
        self.assertIn("EAR raw: 0.250", rendered_text)
        self.assertIn("EAR raw: --", rendered_text)
        self.assertIn("EAR rel.: --", rendered_text)
        self.assertIn("P RF ultimos 10 s: 73.0%", rendered_text)
        self.assertIn("Persistencia RF: 2.5 / 8.0 s", rendered_text)
        self.assertIn("Recuperacion ocular: ACTIVA", rendered_text)
        self.assertIn("Estado operativo: NORMAL", rendered_text)
        self.assertIn("Origen alerta: --", rendered_text)
        self.assertNotIn("nan", rendered_text.lower())
        self.assertEqual(record, original_record)
        self.assertEqual(probability, 0.73)
        self.assertEqual(model_persistence_seconds(None, 10.0), 0.0)
        self.assertEqual(model_persistence_seconds(8.5, 10.0), 1.5)

    def test_realtime_full_window_cadence_and_eye_closure_is_diagnostic(self):
        clock = [0.0]
        inference_times = []
        statuses = []
        debug_keys = {"enabled": False, "disabled": False}
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        capture = Mock()
        capture.isOpened.return_value = True

        def read_frame():
            clock[0] += 0.205
            return True, frame

        def measurements(*args):
            ear = 0.25 if clock[0] <= CALIBRATION_SECONDS else 0.10
            return {
                "valid_face": True, "ear_left": ear, "ear_right": ear,
                "ear_mean": ear, "mar": 0.02,
                "pitch": 0.0, "yaw": 0.0, "roll": 0.0,
            }

        def predict(features):
            inference_times.append(clock[0])
            return np.array([[0.9, 0.1]])

        def save_status(*args, **kwargs):
            if "status" in kwargs:
                statuses.append((clock[0], kwargs["status"], kwargs["positive_windows"]))

        def wait_key(_):
            if clock[0] >= 55:
                return ord("q")
            if clock[0] >= 35 and not debug_keys["disabled"]:
                debug_keys["disabled"] = True
                return ord("d")
            if clock[0] >= 32 and not debug_keys["enabled"]:
                debug_keys["enabled"] = True
                return ord("d")
            return -1

        capture.read.side_effect = read_frame
        detector = Mock()
        landmarks = [SimpleNamespace(x=0.5, y=0.5, z=0.0) for _ in range(478)]
        detector.detect_for_video.return_value = SimpleNamespace(face_landmarks=[landmarks])
        model = Mock()
        model.predict_proba.side_effect = predict
        artifact = {"model": model, "feature_columns": FEATURE_COLUMNS}
        module = "src.realtime_fatigueguard"
        with (
            patch(f"{module}.cv2.VideoCapture", return_value=capture),
            patch(f"{module}.build_face_landmarker", return_value=detector),
            patch(f"{module}.load_model_artifact", return_value=artifact),
            patch(f"{module}.measurements_from_landmarks", side_effect=measurements),
            patch(f"{module}.time.perf_counter", side_effect=lambda: clock[0]),
            patch(f"{module}.initialize_database"),
            patch(f"{module}.upsert_current_status", side_effect=save_status),
            patch(f"{module}.EventLogger"),
            patch(f"{module}.AlertSound"),
            patch(f"{module}.cv2.imshow"),
            patch(f"{module}.cv2.waitKey", side_effect=wait_key),
            patch(f"{module}.cv2.destroyAllWindows"),
            patch("builtins.print"),
        ):
            run_realtime()

        calibration_end = next(t for t, state, _ in statuses if state == "NORMAL")
        buffer_start = calibration_end + 0.205
        self.assertGreaterEqual(len(inference_times), 3)
        self.assertGreaterEqual(inference_times[0] - buffer_start, WINDOW_SECONDS - 1.0 / PROCESSING_FPS)
        self.assertLess(inference_times[0] - buffer_start, WINDOW_SECONDS)
        for previous, current in zip(inference_times, inference_times[1:]):
            self.assertGreaterEqual(current - previous, INFERENCE_INTERVAL_SECONDS)
            self.assertLess(current - previous, INFERENCE_INTERVAL_SECONDS + 0.205)
        self.assertNotIn("ALERTA", [state for _, state, _ in statuses])
        self.assertTrue(all(positives == 0 for _, _, positives in statuses))
        self.assertEqual(debug_keys, {"enabled": True, "disabled": True})

    def test_eye_closure_duration_is_diagnostic_only(self):
        tracker = EyeClosureTracker()
        closed = {
            "valid_face": True,
            "yaw_delta": 0.0,
            "ear_left": 0.10,
            "ear_right": 0.10,
            "ear_base": 0.25,
        }
        self.assertEqual(current_eye_state(closed), "CLOSED")
        self.assertIsNone(tracker.update(closed, 0.0))
        self.assertIsNone(tracker.confirmed_state)
        self.assertIsNone(tracker.update(closed, 0.2))
        self.assertIsNone(tracker.confirmed_state)
        self.assertEqual(tracker.update(closed, 0.4), "CLOSED")
        self.assertEqual(tracker.confirmed_state, "CLOSED")
        self.assertEqual(tracker.update(closed, 4.4), "CLOSED")
        self.assertAlmostEqual(tracker.duration_seconds, 4.0)
        self.assertEqual(determine_alert_origin(False), "--")
        self.assertEqual(classify_model_probability(0.10, None, 4.4), ("NORMAL", None))

        neutral = dict(closed, ear_left=0.20, ear_right=0.20)
        self.assertIsNone(current_eye_state(neutral))
        self.assertIsNone(tracker.update(neutral, 4.6))
        self.assertIsNone(tracker.confirmed_state)
        self.assertEqual(tracker.duration_seconds, 0.0)

        self.assertIsNone(tracker.update(closed, 5.0))
        self.assertIsNone(tracker.update(closed, 5.2))
        self.assertEqual(tracker.update(closed, 5.4), "CLOSED")
        self.assertEqual(tracker.update(closed, 9.4), "CLOSED")
        self.assertAlmostEqual(tracker.duration_seconds, 4.0)

        opened = dict(closed, ear_left=0.25, ear_right=0.25)
        self.assertEqual(current_eye_state(opened), "OPEN")
        self.assertIsNone(tracker.update(opened, 9.6))
        self.assertIsNone(tracker.confirmed_state)
        self.assertEqual(tracker.duration_seconds, 0.0)
        self.assertIsNone(tracker.update(opened, 9.8))
        self.assertEqual(tracker.update(opened, 10.0), "OPEN")
        self.assertEqual(tracker.confirmed_state, "OPEN")

    def test_feature_order_and_known_window_values(self):
        records = []
        for index in range(50):
            records.append(
                {
                    "valid_face": True,
                    "ear_relative": 0.75 if index < 20 else 1.0,
                    "mar_relative": 1.6 if index < 5 else 1.0,
                    "pitch_delta": float(index % 4),
                    "yaw_delta": 0.0,
                    "roll_delta": 0.0,
                }
            )
        features = calculate_window_features(records, PROCESSING_FPS)
        self.assertEqual(list(features), FEATURE_COLUMNS)
        self.assertEqual(len(features), 24)
        self.assertAlmostEqual(features["perclos_relative"], 0.4)
        self.assertAlmostEqual(features["max_eye_closure_seconds"], 4.0)
        self.assertEqual(features["eye_closure_episode_count"], 1)
        self.assertAlmostEqual(features["high_mouth_open_rate"], 0.1)

    def test_real_training_window_equivalence_when_local_data_exists(self):
        windows_path = PROJECT_ROOT / "data" / "processed" / "uta_rldd_landmark_windows.csv"
        frames_path = PROJECT_ROOT / "local_archive" / "outputs" / "metrics" / "uta_landmarks_per_frame.csv"
        baseline_path = PROJECT_ROOT / "outputs" / "metrics" / "uta_subject_calibration_baselines.csv"
        if not all(path.exists() for path in (windows_path, frames_path, baseline_path)):
            self.skipTest("Datos locales de equivalencia no incluidos en el clon")

        expected_row = pd.read_csv(windows_path, dtype={"subject_id": str}).iloc[0]
        frames = pd.read_csv(frames_path, dtype={"subject_id": str})
        frames["subject_id"] = frames.subject_id.str.zfill(2)
        subject_id = str(expected_row.subject_id).zfill(2)
        group = frames[
            (frames.video_id == expected_row.video_id)
            & (frames.timestamp >= expected_row.window_start)
            & (frames.timestamp < expected_row.window_end)
        ].sort_values("timestamp")
        baselines = pd.read_csv(baseline_path, dtype={"subject_id": str})
        baselines["subject_id"] = baselines.subject_id.str.zfill(2)
        baseline_row = baselines[baselines.subject_id == subject_id].iloc[0]
        baseline = CalibrationBaseline(
            ear_base=baseline_row.ear_base,
            mar_base=baseline_row.mar_base,
            pitch_base=baseline_row.pitch_base,
            yaw_base=baseline_row.yaw_base,
            roll_base=baseline_row.roll_base,
            n_calibration_frames=int(baseline_row.n_valid_face),
        )
        records = [add_relative_measurements(row, baseline) for row in group.to_dict("records")]
        observed = calculate_window_features(records, PROCESSING_FPS)
        expected = expected_row[FEATURE_COLUMNS].to_numpy(dtype=float)
        actual = np.asarray([observed[column] for column in FEATURE_COLUMNS], dtype=float)
        self.assertTrue(np.allclose(actual, expected, equal_nan=True, atol=1e-12))

    def test_mediapipe_landmarker_initializes_without_webcam(self):
        landmarker = build_face_landmarker()
        landmarker.close()


class TestSQLiteAndDashboard(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temporary_directory.name) / "fatigueguard.db"

    def tearDown(self):
        self.temporary_directory.cleanup()

    def test_database_initializes_required_tables(self):
        initialize_database(self.db_path)
        with closing(connect_database(self.db_path)) as connection:
            tables = {
                row[0]
                for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
            }
        self.assertTrue({"current_status", "events"}.issubset(tables))

    def test_status_event_and_dashboard_roundtrip(self):
        upsert_current_status(
            "session-test", "CAMION_TEST", "ALERTA", 0.82, 3,
            "2026-09-04T22:00:00+00:00", self.db_path,
        )
        event = {
            "event_id": "event-test",
            "session_id": "session-test",
            "equipment_id": "CAMION_TEST",
            "timestamp_start": "2026-09-04T21:59:50+00:00",
            "timestamp_end": "2026-09-04T22:00:00+00:00",
            "duration_seconds": 10.0,
            "alert_reason": "MODEL_PERSISTENCE",
            "max_probability": 0.82,
            "mean_probability": 0.70,
            "mean_perclos": 0.45,
            "max_eye_closure_seconds": 2.0,
            "alert_triggered": True,
        }
        insert_event(event, self.db_path)

        status = get_current_status(self.db_path)
        events = get_recent_events(db_path=self.db_path)
        dashboard_status, dashboard_events, total = load_dashboard_data(self.db_path)
        self.assertEqual(status["positive_windows"], 3)
        self.assertEqual(events[0]["event_id"], "event-test")
        self.assertEqual(dashboard_status, status)
        self.assertEqual(dashboard_events, events)
        self.assertEqual(total, 1)


if __name__ == "__main__":
    unittest.main()
