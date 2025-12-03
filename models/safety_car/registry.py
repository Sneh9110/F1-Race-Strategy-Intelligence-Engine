from typing import Optional, Dict, Any
import json
import os


class ModelRegistry:
    def __init__(self, base_dir: str = "models/saved/safety_car"):
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)
        self.registry_file = os.path.join(self.base_dir, "registry.json")
        if not os.path.exists(self.registry_file):
            with open(self.registry_file, "w", encoding="utf-8") as fh:
                json.dump({"models": {}, "aliases": {}}, fh)

    def _load(self):
        with open(self.registry_file, "r", encoding="utf-8") as fh:
            return json.load(fh)

    def _save(self, data):
        with open(self.registry_file, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)

    def register_model(self, version: str, metadata: Dict[str, Any]):
        data = self._load()
        data["models"][version] = metadata
        self._save(data)

    def load_model(self, version: str = "latest"):
        data = self._load()
        if version == "latest":
            versions = list(data.get("models", {}).keys())
            if not versions:
                raise FileNotFoundError("No models registered")
            version = sorted(versions)[-1]
        md = data.get("models", {}).get(version)
        if not md:
            raise FileNotFoundError(f"Model {version} not found")
        # in a real impl we'd load model binary; here we return a stub that raises
        model_path = md.get("model_path")
        return md.get("model_object") or None

    def list_models(self):
        data = self._load()
        return data.get("models", {})

    def promote_model(self, version: str, alias: str = "production"):
        data = self._load()
        data.setdefault("aliases", {})[alias] = version
        self._save(data)

    def delete_model(self, version: str):
        data = self._load()
        if version in data.get("models", {}):
            del data["models"][version]
            self._save(data)
