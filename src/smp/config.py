"""Project configuration with repository-local defaults and optional overrides."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_FILE = REPOSITORY_ROOT / "config" / "default.json"
LOCAL_CONFIG_FILE = REPOSITORY_ROOT / "config" / "local.json"


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = value
    return merged


@dataclass(frozen=True)
class ProjectConfig:
    """Resolved project settings."""

    repository_root: Path
    data_root: Path
    output_root: Path
    random_seed: int
    values: dict[str, Any]

    def path(self, name: str) -> Path:
        """Return a named path, resolved against the repository root."""
        raw_path = Path(self.values["paths"][name]).expanduser()
        if raw_path.is_absolute():
            return raw_path
        parts = raw_path.parts
        if parts and parts[0] == "data":
            return self.data_root.joinpath(*parts[1:])
        if parts and parts[0] == "outputs":
            return self.output_root.joinpath(*parts[1:])
        return self.repository_root / raw_path

    def ensure_output(self, name: str) -> Path:
        """Create and return a named output directory."""
        output_path = self.path(name)
        output_path.mkdir(parents=True, exist_ok=True)
        return output_path


def load_config(config_file: str | Path | None = None) -> ProjectConfig:
    """Load defaults, local overrides, environment variables, and an explicit file."""
    values = _read_json(DEFAULT_CONFIG_FILE)
    values = _merge(values, _read_json(LOCAL_CONFIG_FILE))
    if config_file is not None:
        values = _merge(values, _read_json(Path(config_file)))

    data_root = Path(os.getenv("SMP_DATA_ROOT", values["data_root"])).expanduser()
    output_root = Path(os.getenv("SMP_OUTPUT_ROOT", values["output_root"])).expanduser()
    if not data_root.is_absolute():
        data_root = REPOSITORY_ROOT / data_root
    if not output_root.is_absolute():
        output_root = REPOSITORY_ROOT / output_root

    return ProjectConfig(
        repository_root=REPOSITORY_ROOT,
        data_root=data_root.resolve(),
        output_root=output_root.resolve(),
        random_seed=int(os.getenv("SMP_RANDOM_SEED", values["random_seed"])),
        values=values,
    )
