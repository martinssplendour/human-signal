import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

from core.features import compute_features
from core.filters import ema
from core.quality import estimate_quality
from modules.attention import attention_score
from modules.ergonomics_distance import distance_status
from modules.fatigue import fatigue_score
from modules.fuse import fuse_scores
from modules.posture import posture_status
from modules.stress import stress_score
from ml import SignalClassifierBundle


@dataclass
class PipelineState:
    fatigue_sm: float = 0.0
    attention_sm: float = 0.0
    tension_sm: float = 0.0
    prev_time: Optional[float] = None
    frame_drops: int = 0


class WellnessPipeline:
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.state = PipelineState()
        self.ml = SignalClassifierBundle.from_config(cfg)

    def process(self, frame_rgb, det: Dict[str, Any], calibrating: bool = False, motion_energy: float = 0.0):
        now = time.time()
        frame_gap = 0.0
        if self.state.prev_time is not None:
            frame_gap = now - self.state.prev_time
            if frame_gap > self.cfg["quality"].get("max_frame_gap_secs", 0.5):
                self.state.frame_drops += 1
        self.state.prev_time = now

        feats = compute_features(det)
        quality = estimate_quality(frame_rgb, det, feats, self.cfg)
        quality["frame_gap_secs"] = frame_gap
        quality["frame_drops"] = self.state.frame_drops

        post = posture_status(feats, quality, self.cfg, calibrating=calibrating)
        fat = fatigue_score(
            feats,
            quality,
            self.cfg,
            calibrating=calibrating,
            posture_state=post["state"],
            motion_energy=motion_energy,
        )
        att = attention_score(feats, quality, self.cfg, calibrating=calibrating)
        tension = stress_score(feats, quality, self.cfg, calibrating=calibrating)
        if not calibrating:
            applied = self.ml.apply(feats, quality, {"fatigue": fat, "attention": att, "tension": tension})
            fat = applied["fatigue"]
            att = applied["attention"]
            tension = applied["tension"]
        dist = distance_status(det, self.cfg)

        alpha = self.cfg["smoothing"]["ema_alpha"]
        if quality["signal_ok"]:
            self.state.fatigue_sm = ema(fat["score"], self.state.fatigue_sm, alpha)
            self.state.attention_sm = ema(att["score"], self.state.attention_sm, alpha)
            self.state.tension_sm = ema(tension["score"], self.state.tension_sm, alpha)

        display_fat = {**fat, "score": self.state.fatigue_sm}
        display_att = {**att, "score": self.state.attention_sm}
        display_tension = {**tension, "score": self.state.tension_sm}
        fused = fuse_scores(
            fatigue=display_fat,
            attention=display_att,
            stress=display_tension,
            weights=self.cfg["fusion"]["weights"],
            min_conf=self.cfg["quality"]["min_confidence"],
        )
        raw_fused = fuse_scores(
            fatigue=fat,
            attention=att,
            stress=tension,
            weights=self.cfg["fusion"]["weights"],
            min_conf=self.cfg["quality"]["min_confidence"],
        )

        return {
            "features": feats,
            "quality": quality,
            "fatigue": fat,
            "attention": att,
            "tension": tension,
            "posture": post,
            "distance": dist,
            "fused": fused,
            "raw_fused": raw_fused,
            "raw_scores": {
                "fatigue": float(fat["score"]),
                "attention": float(att["score"]),
                "tension": float(tension["score"]),
            },
            "smoothed": {
                "fatigue": self.state.fatigue_sm,
                "attention": self.state.attention_sm,
                "tension": self.state.tension_sm,
            },
        }
