"""Runtime facts recorded with every experiment."""

from __future__ import annotations

import platform
import subprocess
import sys
from importlib import metadata
from pathlib import Path

from pydantic import BaseModel, Field


class EnvironmentInfo(BaseModel):
    python_version: str
    platform: str
    git_sha: str | None
    packages: dict[str, str] = Field(default_factory=dict)
    cuda_version: str | None = None
    torch_version: str | None = None
    gpu_name: str | None = None
    gpu_total_mib: int | None = None
    compute_capability: str | None = None


_TRACKED_PACKAGES = (
    "torch",
    "transformers",
    "accelerate",
    "lm-eval",
    "torchao",
    "numpy",
    "pydantic",
    "typer",
)


def _git_sha(repo: Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    sha = completed.stdout.strip()
    return sha or None


def _package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def collect_environment(repo: Path) -> EnvironmentInfo:
    packages = {
        name: version
        for name in _TRACKED_PACKAGES
        if (version := _package_version(name)) is not None
    }
    info = EnvironmentInfo(
        python_version=sys.version.split()[0],
        platform=platform.platform(),
        git_sha=_git_sha(repo),
        packages=packages,
    )
    try:
        import torch
    except (ImportError, OSError):  # missing or broken torch install
        return info

    info.torch_version = torch.__version__
    info.cuda_version = torch.version.cuda
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info.gpu_name = props.name
        info.gpu_total_mib = props.total_memory // (1024 * 1024)
        info.compute_capability = f"{props.major}.{props.minor}"
    return info
