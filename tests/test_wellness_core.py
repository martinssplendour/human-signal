import unittest
from unittest.mock import patch

import numpy as np

from configs.settings import AppConfig, load_config
from core.quality import estimate_quality
from core.session_utils import calibration_gate_status, compact_summary
from modules import attention, fatigue
from modules.care_events import CareMonitor
from modules.driver_events import DriverMonitor
from modules.healthcare_events import HealthcareMonitor


def synthetic_face(ear_open=True):
    face = np.zeros((478, 3), dtype=np.float32)
    face[:, 0] = 100
    face[:, 1] = 100
    face[33, :2] = [80, 100]
    face[133, :2] = [120, 100]
    face[160, :2] = [88, 90 if ear_open else 99]
    face[158, :2] = [112, 90 if ear_open else 99]
    face[153, :2] = [112, 110 if ear_open else 101]
    face[144, :2] = [88, 110 if ear_open else 101]
    face[362, :2] = [180, 100]
    face[263, :2] = [220, 100]
    face[385, :2] = [188, 90 if ear_open else 99]
    face[387, :2] = [212, 90 if ear_open else 99]
    face[373, :2] = [212, 110 if ear_open else 101]
    face[380, :2] = [188, 110 if ear_open else 101]
    face[61, :2] = [130, 150]
    face[291, :2] = [190, 150]
    face[13, :2] = [160, 145]
    face[14, :2] = [160, 155]
    face[1, :2] = [150, 125]
    face[10, :2] = [150, 60]
    face[152, :2] = [150, 230]
    face[70, :2] = [100, 75]
    face[63, :2] = [115, 73]
    face[300, :2] = [200, 75]
    face[293, :2] = [185, 73]
    face[468:477, :2] = [100, 100]
    return face


class ConfigTests(unittest.TestCase):
    def test_default_config_loads(self):
        cfg = load_config().as_legacy_dict()
        self.assertEqual(cfg["video"]["width"], 1280)
        self.assertIn("recording", cfg)
        self.assertIn("min_sharpness", cfg["quality"])

    def test_invalid_distance_thresholds_raise(self):
        with self.assertRaises(Exception):
            AppConfig(thresholds={"distance_face_ratio_far": 0.5, "distance_face_ratio_close": 0.2})


class QualityTests(unittest.TestCase):
    def test_quality_reports_missing_face(self):
        cfg = load_config().as_legacy_dict()
        frame = np.zeros((120, 160, 3), dtype=np.uint8)
        quality = estimate_quality(frame, {"face": None, "pose": None, "size": (160, 120)}, {}, cfg)
        self.assertFalse(quality["signal_ok"])
        self.assertIn("face not detected", quality["reasons"])

    def test_calibration_gate_requires_quality_checks(self):
        cfg = load_config().as_legacy_dict()
        quality = {
            "face_present": True,
            "brightness": cfg["quality"]["min_brightness"],
            "sharpness": cfg["quality"]["min_sharpness"],
            "face_ratio": 0.02,
        }
        gate = calibration_gate_status(quality, cfg)
        self.assertTrue(gate["ok"])
        quality["face_present"] = False
        self.assertFalse(calibration_gate_status(quality, cfg)["ok"])


class FatigueTests(unittest.TestCase):
    def setUp(self):
        fatigue.reset()
        self.cfg = load_config().as_legacy_dict()
        self.quality = {"conf_base": 1.0, "face_present": True, "signal_ok": True}

    def test_perclos_uses_closed_eye_time_window(self):
        feats = {"ear_left": 0.1, "ear_right": 0.1, "mar": 0.1, "neck_angle": None}
        with patch("modules.fatigue.time.time", side_effect=[0.0, 1.0, 2.0, 3.0]):
            fatigue.fatigue_score(feats, self.quality, self.cfg)
            fatigue.fatigue_score(feats, self.quality, self.cfg)
            fatigue.fatigue_score(feats, self.quality, self.cfg)
            result = fatigue.fatigue_score(feats, self.quality, self.cfg)
        self.assertGreaterEqual(result["perclos"], 0.99)
        self.assertEqual(result["state"], "microsleep")


class AttentionTests(unittest.TestCase):
    def setUp(self):
        attention.reset()
        self.cfg = load_config().as_legacy_dict()
        self.quality = {"conf_base": 1.0, "face_present": True, "signal_ok": True}

    def test_face_absent_state_after_dwell(self):
        feats = {"face": None, "ear_left": None, "ear_right": None, "gaze": 0.0, "head_yaw": None}
        with patch("modules.attention.time.time", side_effect=[0.0, 0.3]):
            attention.attention_score(feats, self.quality, self.cfg)
            result = attention.attention_score(feats, self.quality, self.cfg)
        self.assertEqual(result["state"], "face_absent")
        self.assertEqual(result["score"], 0.0)


class DriverMonitorTests(unittest.TestCase):
    def base_result(self):
        return {
            "fatigue": {
                "state": "alert",
                "perclos": 0.0,
                "microsleep": False,
                "blink_duration": 0.0,
                "yawns": 0,
                "head_nods": 0,
            },
            "attention": {"state": "looking_forward", "offscreen_duration": 0.0},
            "quality": {"signal_ok": True},
            "smoothed": {"fatigue": 10.0, "attention": 95.0, "tension": 0.0},
        }

    def test_microsleep_is_critical_driver_event(self):
        monitor = DriverMonitor()
        result = self.base_result()
        result["fatigue"]["state"] = "microsleep"
        result["fatigue"]["microsleep"] = True
        result["smoothed"]["fatigue"] = 95.0
        state = monitor.update(result, now=10.0)
        self.assertEqual(state["risk"], "critical")
        self.assertEqual(state["events"][0]["event"], "microsleep_detected")

    def test_poor_signal_is_insufficient_when_no_other_risk(self):
        monitor = DriverMonitor()
        result = self.base_result()
        result["quality"]["signal_ok"] = False
        state = monitor.update(result, now=10.0)
        self.assertEqual(state["risk"], "insufficient_signal")
        self.assertEqual(state["events"][0]["event"], "poor_signal")

    def test_single_yawn_does_not_create_break_alert(self):
        monitor = DriverMonitor()
        result = self.base_result()
        result["fatigue"]["state"] = "fatigue_signs"
        result["fatigue"]["yawns"] = 1
        state = monitor.update(result, now=10.0)
        self.assertEqual(state["events"], [])

    def test_mid_fatigue_score_alone_stays_normal(self):
        monitor = DriverMonitor()
        result = self.base_result()
        result["fatigue"]["state"] = "fatigue_signs"
        result["raw_scores"] = {"fatigue": 55.0, "attention": 95.0, "tension": 0.0}
        state = monitor.update(result, now=10.0)
        self.assertEqual(state["events"], [])
        self.assertEqual(state["risk"], "normal")

    def test_slow_blink_without_other_fatigue_is_watch_only(self):
        monitor = DriverMonitor()
        result = self.base_result()
        result["fatigue"]["blink_duration"] = 0.35
        result["raw_scores"] = {"fatigue": 55.0, "attention": 95.0, "tension": 0.0}
        state = monitor.update(result, now=10.0)
        self.assertEqual(state["risk"], "watch")
        self.assertEqual(state["events"][0]["event"], "slow_blink_watch")

    def test_five_yawns_in_five_minutes_recommends_rest(self):
        monitor = DriverMonitor()
        result = self.base_result()
        for i in range(5):
            result["fatigue"]["yawns"] = i + 1
            state = monitor.update(result, now=10.0 + i * 50.0)
        self.assertEqual(state["risk"], "elevated")
        self.assertTrue(any(e["event"] == "repeated_yawning_5m" for e in state["events"]))

    def test_two_second_eye_closure_is_critical(self):
        monitor = DriverMonitor()
        result = self.base_result()
        result["fatigue"]["state"] = "drowsy"
        result["fatigue"]["closed_duration"] = 2.1
        state = monitor.update(result, now=10.0)
        self.assertEqual(state["risk"], "critical")
        self.assertEqual(state["events"][0]["event"], "sustained_eye_closure")

    def test_looking_away_does_not_trigger_driver_event(self):
        monitor = DriverMonitor()
        result = self.base_result()
        result["attention"] = {"state": "looking_away", "offscreen_duration": 30.0}
        state = monitor.update(result, now=10.0)
        self.assertEqual(state["events"], [])
        self.assertEqual(state["risk"], "normal")

    def test_eye_critical_suppresses_lower_eye_alerts(self):
        monitor = DriverMonitor()
        result = self.base_result()
        result["fatigue"]["closed_duration"] = 2.1
        result["fatigue"]["perclos"] = 0.3
        result["fatigue"]["blink_duration"] = 0.4
        state = monitor.update(result, now=10.0)
        eye_events = [e for e in state["events"] if e["event"] in {"sustained_eye_closure", "perclos_critical", "slow_blink_elevated"}]
        self.assertEqual(len(eye_events), 1)
        self.assertEqual(eye_events[0]["event"], "sustained_eye_closure")

    def test_two_elevated_events_escalate_to_compound_critical(self):
        monitor = DriverMonitor()
        result = self.base_result()
        result["fatigue"]["yawns"] = 5
        first = monitor.update(result, now=10.0)
        self.assertTrue(any(e["severity"] == "elevated" for e in first["events"]))
        result = self.base_result()
        result["fatigue"]["perclos"] = 0.16
        second = monitor.update(result, now=60.0)
        self.assertEqual(second["risk"], "critical")
        self.assertTrue(any(e["event"] == "compound_fatigue_escalation" for e in second["events"]))

    def test_hydration_reminder_is_watch(self):
        monitor = DriverMonitor()
        monitor.reset(0.0)
        state = monitor.update(self.base_result(), now=5401.0)
        self.assertEqual(state["risk"], "watch")
        self.assertTrue(any(e["event"] == "hydration_reminder" for e in state["events"]))


class CareMonitorTests(unittest.TestCase):
    def base_result(self):
        return {
            "fatigue": {"state": "alert"},
            "attention": {"state": "looking_forward"},
            "quality": {"signal_ok": True, "face_present": True},
            "posture": {"state": "upright"},
            "smoothed": {"fatigue": 10.0, "attention": 95.0, "tension": 0.0},
        }

    def test_observed_state_when_visible_and_signal_ok(self):
        monitor = CareMonitor()
        state = monitor.update(self.base_result(), now=10.0, context="Chair", motion_energy=0.02)
        self.assertEqual(state["state"], "observed")
        self.assertEqual(state["events"], [])

    def test_extended_absence_becomes_urgent_review(self):
        monitor = CareMonitor()
        monitor.reset(0.0)
        result = self.base_result()
        result["quality"]["face_present"] = False
        result["quality"]["signal_ok"] = False
        result["attention"]["state"] = "face_absent"
        state = monitor.update(result, now=200.0, context="Chair", motion_energy=0.0)
        self.assertEqual(state["state"], "urgent_review")
        self.assertTrue(any(e["event"] == "resident_absent_extended" for e in state["events"]))

    def test_possible_slump_needs_review(self):
        monitor = CareMonitor()
        result = self.base_result()
        result["posture"]["state"] = "slouching"
        result["attention"]["state"] = "eyes_closed"
        result["fatigue"]["state"] = "drowsy"
        state = monitor.update(result, now=10.0, context="Chair", motion_energy=0.0)
        self.assertEqual(state["state"], "needs_review")
        self.assertTrue(any(e["event"] == "possible_slump" for e in state["events"]))


class HealthcareMonitorTests(unittest.TestCase):
    def base_result(self):
        return {
            "fatigue": {
                "state": "alert",
                "microsleep": False,
                "blink_duration": 0.0,
            },
            "attention": {"state": "looking_forward"},
            "quality": {"signal_ok": True, "face_present": True},
            "posture": {"state": "upright"},
            "motion_energy": 0.0,
            "smoothed": {"fatigue": 10.0, "attention": 95.0, "tension": 0.0},
        }

    def test_stable_healthcare_observation(self):
        monitor = HealthcareMonitor()
        state = monitor.update(self.base_result(), now=10.0, patient_session_id="S1")
        self.assertEqual(state["state"], "stable_observation")
        self.assertEqual(state["events"], [])

    def test_calibration_failure_marks_insufficient_signal(self):
        monitor = HealthcareMonitor()
        state = monitor.update(self.base_result(), now=10.0, patient_session_id="S1", calibration_ok=False)
        self.assertEqual(state["state"], "insufficient_signal")
        self.assertEqual(state["events"][0]["event"], "calibration_failed")

    def test_observation_interrupted_is_urgent(self):
        monitor = HealthcareMonitor()
        monitor.reset(0.0)
        result = self.base_result()
        result["quality"]["face_present"] = False
        result["quality"]["signal_ok"] = False
        result["attention"]["state"] = "face_absent"
        state = monitor.update(result, now=300.0, patient_session_id="S1")
        self.assertEqual(state["state"], "urgent_review")
        self.assertTrue(any(e["event"] == "observation_interrupted" for e in state["events"]))

    def test_reduced_responsiveness_proxy_needs_review(self):
        monitor = HealthcareMonitor()
        result = self.base_result()
        result["fatigue"]["state"] = "drowsy"
        result["attention"]["state"] = "eyes_closed"
        state = monitor.update(result, now=10.0, patient_session_id="S1")
        self.assertEqual(state["state"], "needs_review")
        self.assertTrue(any(e["event"] == "reduced_responsiveness_proxy" for e in state["events"]))


class SessionSummaryTests(unittest.TestCase):
    def test_compact_summary_uses_driver_state(self):
        summary = compact_summary(
            "Driver",
            {"risk": "critical", "summary": {"trip_secs": 120.0, "usable_signal_pct": 90.0, "critical_alerts": 2}},
            {},
            {},
            {},
        )
        self.assertEqual(summary["state"], "critical")
        self.assertEqual(summary["review_events"], 2)

    def test_compact_summary_uses_healthcare_state(self):
        summary = compact_summary(
            "Healthcare observation",
            {},
            {},
            {"state": "needs_review", "summary": {"session_secs": 60.0, "usable_signal_pct": 80.0, "visible_pct": 70.0, "needs_review_events": 3, "urgent_review_events": 1}},
            {},
        )
        self.assertEqual(summary["state"], "needs_review")
        self.assertEqual(summary["review_events"], 4)


if __name__ == "__main__":
    unittest.main()
