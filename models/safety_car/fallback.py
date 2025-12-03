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
        base_rate = 0.1
        tcfg = self.tracks.get(inp.track_name, {})
        base_rate = tcfg.get("base_sc_prob", base_rate)
        incident_bonus = 0.0
        for inc in inp.incident_logs:
            if inc.severity == "minor":
                incident_bonus += 0.1
            elif inc.severity == "moderate":
                incident_bonus += 0.2
            else:
                incident_bonus += 0.4
        lap_factor = 1.0
        if inp.current_lap <= 5:
            lap_factor = 1.2
        if inp.current_lap >= inp.total_laps - 10:
            lap_factor = 1.3
        prob = min(1.0, base_rate * lap_factor + incident_bonus)
        window = (inp.current_lap + 2, min(inp.total_laps, inp.current_lap + 20))
        risk = {"base_rate": base_rate, "incident_bonus": incident_bonus, "lap_factor": lap_factor}
        return PredictionOutput(sc_probability=prob, deployment_window=window, confidence=0.6, risk_factors=risk, metadata={})
