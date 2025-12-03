from typing import Dict, Any
from .base import PredictionInput, PredictionOutput
import yaml


class FallbackHeuristics:
    def __init__(self, track_config_path: str = "config/tracks.yaml"):
        try:
            with open(track_config_path, "r", encoding="utf-8") as fh:
                self.tracks = yaml.safe_load(fh)
        except Exception:
            self.tracks = {}

    def predict(self, inp: PredictionInput) -> PredictionOutput:
        tcfg = self.tracks.get(inp.track_name, {})
        base = tcfg.get("base_pit_loss", 20.0)
        congestion_penalty = inp.cars_in_pit_window * 2.0
        compound_penalty = 0.5 if inp.tire_compound_change else 0.0
        traffic_penalty = 1.0 if (inp.gap_to_ahead is not None and inp.gap_to_ahead < 3.0) else 0.0
        total = base + congestion_penalty + compound_penalty + traffic_penalty
        pit_delta = total - base
        return PredictionOutput(total_pit_loss=float(total), pit_delta=float(pit_delta), window_sensitivity=0.5, congestion_penalty=float(congestion_penalty), base_pit_loss=float(base), confidence=0.65, metadata={})
