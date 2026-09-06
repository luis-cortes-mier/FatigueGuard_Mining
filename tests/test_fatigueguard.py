from __future__ import annotations

import re
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from dashboard.app import load_dashboard_data
from src.config import (
    CALIBRATION_SECONDS,
    DATABASE_PATH,
    EYE_CLOSURE_ALERT_SECONDS,
    HISTORY_SIZE,
    MEDIAPIPE_MODEL_PATH,
    MIN_POSITIVE_WINDOWS,
    MODEL_PATH,
    MODEL_THRESHOLD,
    PROCESSING_FPS,
    PROJECT_ROOT,
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
    FEATURE_COLUMNS,
    CalibrationBaseline,
    add_relative_measurements,
    calculate_window_features,
)
from src.realtime_fatigueguard import (
    EyeClosureTracker,
    build_face_landmarker,
    classify_decision_history,
    load_model_artifact,
)


class TestValidatedConfiguration(unittest.TestCase):
    def test_validated_parameters_are_unchanged(self):
        self.assertEqual(CALIBRATION_SECONDS, 30)
        self.assertEqual(PROCESSING_FPS, 5)
        self.assertEqual(WINDOW_SECONDS, 10)
        self.assertEqual(MODEL_THRESHOLD, 0.36)
        self.assertEqual(HISTORY_SIZE, 5)
        self.assertEqual(MIN_POSITIVE_WINDOWS, 3)
        self.assertEqual(EYE_CLOSURE_ALERT_SECONDS, 4.0)

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


class TestRealtimeRulesAndFeatures(unittest.TestCase):
    def test_three_of_five_rule(self):
        self.assertEqual(classify_decision_history([]), ("NORMAL", 0))
        self.assertEqual(
            classify_decision_history([True, False, False, False, False]),
            ("POSIBLE SOMNOLENCIA", 1),
        )
        self.assertEqual(
            classify_decision_history([True, False, True, False, True]),
            ("ALERTA", 3),
        )

    def test_continuous_eye_closure_rule(self):
        tracker = EyeClosureTracker(EYE_CLOSURE_ALERT_SECONDS)
        closed = {
            "valid_face": True,
            "yaw_delta": 0.0,
            "ear_left": 0.10,
            "ear_right": 0.10,
            "ear_base": 0.25,
        }
        self.assertFalse(tracker.update(closed, 0.0))
        self.assertFalse(tracker.update(closed, 3.99))
        self.assertTrue(tracker.update(closed, 4.0))
        opened = dict(closed, ear_left=0.25, ear_right=0.25)
        self.assertFalse(tracker.update(opened, 4.1))
        self.assertEqual(tracker.duration_seconds, 0.0)

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
