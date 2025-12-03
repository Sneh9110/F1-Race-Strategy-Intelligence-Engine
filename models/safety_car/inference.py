from typing import Dict, Any, List
from .base import PredictionInput, PredictionOutput
from .registry import ModelRegistry
from .fallback import FallbackHeuristics
import hashlib
import json
import time


class SafetyCarPredictor:
    def __init__(self, model_version: str = "latest", registry: ModelRegistry = None, cache=None):
        self.model_version = model_version
        self.registry = registry or ModelRegistry()
        self.model = self.registry.load_model(model_version)
        self.cache = cache
        self.fallback = FallbackHeuristics()
        self.stats = {"total": 0, "cache_hits": 0, "fallbacks": 0, "latencies": []}

    def _cache_key(self, inp: PredictionInput) -> str:
        s = json.dumps(inp.dict(), sort_keys=True, default=str)
        return hashlib.md5(s.encode()).hexdigest()

    def predict(self, inp: PredictionInput) -> PredictionOutput:
        self.stats["total"] += 1
        key = self._cache_key(inp)
        if self.cache:
            res = self.cache.get(key)
            if res:
                self.stats["cache_hits"] += 1
                return res
        start = time.time()
        try:
            out = self.model.predict(inp)
            latency = (time.time() - start) * 1000.0
            self.stats["latencies"].append(latency)
            if self.cache:
                self.cache.set(key, out)
            return out
        except Exception:
            self.stats["fallbacks"] += 1
            return self.fallback.predict(inp)

    def predict_batch(self, inputs: List[PredictionInput]):
        return [self.predict(i) for i in inputs]

    def get_stats(self) -> Dict[str, Any]:
        lat = self.stats["latencies"]
        return {
            "total": self.stats["total"],
            "cache_hits": self.stats["cache_hits"],
            "fallbacks": self.stats["fallbacks"],
            "avg_latency_ms": (sum(lat) / len(lat)) if lat else None,
        }
