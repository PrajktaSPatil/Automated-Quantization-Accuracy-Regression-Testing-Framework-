"""Load YAML, validate it, and hash the canonical form."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml
from pydantic import ValidationError

from quantguard.config.schema import QuantGuardConfig
from quantguard.exceptions import ConfigurationError


def load_config(path: Path) -> QuantGuardConfig:
    if not path.is_file():
        raise ConfigurationError(f"Configuration file not found: {path}")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigurationError(f"Invalid YAML in {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigurationError(f"Configuration root must be a mapping: {path}")
    try:
        return QuantGuardConfig.model_validate(raw)
    except ValidationError as exc:
        raise ConfigurationError(f"Invalid configuration in {path}: {exc}") from exc


def canonical_json(config: QuantGuardConfig) -> str:
    payload = config.model_dump(mode="json")
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def config_hash(config: QuantGuardConfig) -> str:
    return hashlib.sha256(canonical_json(config).encode("utf-8")).hexdigest()
