"""Resolve standard paths for dictionaries, user data, logs."""
from __future__ import annotations

from pathlib import Path

from platformdirs import user_data_dir, user_log_dir


APP_NAME = "sokuhen-llm"
APP_AUTHOR = "sokuhen-llm"


def project_root() -> Path:
    """Repository root (two levels up from this file)."""
    return Path(__file__).resolve().parents[2]


def data_dir() -> Path:
    """Bundled / downloaded dictionary storage (inside repo for dev)."""
    p = project_root() / "data"
    p.mkdir(parents=True, exist_ok=True)
    return p


def models_dir() -> Path:
    """Storage for downloaded local LLM weights. Lives next to ``data/``
    so both are accessible from the repo root. Size can be hundreds of
    MB for even the smallest usable Japanese LM; we don't commit these
    (see .gitignore). HuggingFace's ``snapshot_download`` writes here."""
    p = project_root() / "models"
    p.mkdir(parents=True, exist_ok=True)
    return p


def user_dir() -> Path:
    """Per-user writable directory for learning data and config."""
    p = Path(user_data_dir(APP_NAME, APP_AUTHOR))
    p.mkdir(parents=True, exist_ok=True)
    return p


def log_dir() -> Path:
    p = Path(user_log_dir(APP_NAME, APP_AUTHOR))
    p.mkdir(parents=True, exist_ok=True)
    return p


def learning_file() -> Path:
    return user_dir() / "learning.json"


def config_file() -> Path:
    return user_dir() / "config.json"


def pid_file() -> Path:
    """Single-instance marker. The app writes its PID on startup and deletes
    it on clean shutdown. The stop launcher reads it to find the running
    process."""
    return user_dir() / "sokuhen-llm.pid"


def status_file() -> Path:
    """Startup-progress marker read by the BAT launcher. A single line
    of text -- ``loading`` / ``ready`` / ``failed:<msg>`` -- kept in
    the user data dir so launchers can polling-wait for LLM readiness
    before surrendering the console window."""
    return user_dir() / "sokuhen-llm.status"
