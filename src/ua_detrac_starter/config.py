"""Configuration and path helpers for repository-level scripts."""

from __future__ import annotations

from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "competition.yaml"


def resolve_repo_path(path: str | Path, *, base: Path = REPO_ROOT) -> Path:
    """Resolve a path relative to the repository root unless it is absolute."""

    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    return (base / candidate).resolve()


def resolve_config_path(path: str | Path | None = None) -> Path:
    """Resolve config paths from either CWD or the repository root."""

    if path is None:
        return DEFAULT_CONFIG_PATH

    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate

    cwd_candidate = (Path.cwd() / candidate).resolve()
    if cwd_candidate.is_file():
        return cwd_candidate

    return resolve_repo_path(candidate)


def load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_config(path: str | Path | None = None) -> tuple[dict[str, Any], Path]:
    config_path = resolve_config_path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    return load_yaml(config_path), config_path


def config_path(config: dict[str, Any], key: str, default: str) -> Path:
    paths = config.get("paths", {})
    return resolve_repo_path(str(paths.get(key, default)))


def dataset_root(dataset_yaml: Path, dataset_config: dict[str, Any]) -> Path:
    raw_root = Path(str(dataset_config.get("path", "."))).expanduser()
    if raw_root.is_absolute():
        return raw_root
    return (dataset_yaml.parent / raw_root).resolve()


def dataset_split_path(
    dataset_yaml: Path,
    dataset_config: dict[str, Any],
    split: str,
) -> Path:
    raw_split = dataset_config.get(split)
    if raw_split is None:
        raise KeyError(f"Missing `{split}` in {dataset_yaml}")

    split_path = Path(str(raw_split)).expanduser()
    if split_path.is_absolute():
        return split_path
    return (dataset_root(dataset_yaml, dataset_config) / split_path).resolve()

