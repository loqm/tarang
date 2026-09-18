"""Initialize library logging without configuring the host application.

Exposes get_logger(), configure_logging(), set_tidy3d_output(), and tidy3d_verbose().
"""

import logging
from pathlib import Path
from threading import RLock
from typing import TextIO

_logger = logging.getLogger("tarang")
_lock = RLock()
if not any(
    isinstance(
        handler,
        logging.NullHandler,
    )
    for handler in _logger.handlers
):
    _logger.addHandler(logging.NullHandler())
_owned_handlers: list[logging.Handler] = []


def get_logger(name: str) -> logging.Logger:
    """Return a module logger beneath the shared tarang namespace.
    Records propagate to application handlers until explicit configuration.
    """
    if name == "tarang" or name.startswith("tarang."):
        return logging.getLogger(name)
    return _logger.getChild(name)


def configure_logging(
    level: int | str = logging.INFO,
    *,
    stream: TextIO | None = None,
    console: bool = True,
    log_file: str | Path | None = None,
) -> None:
    """Configure TARANG console output and an optional UTF-8 log file.
    Repeated calls replace only owned handlers and never change root logging.
    """
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    # Validate and open new destinations before disturbing existing handlers.
    validated_level = (
        logging.getLevelName(level.upper())
        if isinstance(
            level,
            str,
        )
        else level
    )
    if not isinstance(
        validated_level,
        int,
    ):
        raise ValueError(f"Unknown logging level: {level!r}")
    handlers = []
    if log_file is not None:
        handlers.append(
            logging.FileHandler(
                log_file,
                encoding="utf-8",
            )
        )
    if console:
        handlers.append(logging.StreamHandler(stream))
    with _lock:
        for handler in _owned_handlers:
            _logger.removeHandler(handler)
            handler.close()
        _owned_handlers.clear()
        for handler in handlers:
            handler.setFormatter(formatter)
            _logger.addHandler(handler)
        _owned_handlers.extend(handlers)
        _logger.setLevel(validated_level)
        _logger.propagate = False


_tidy3d_state = None


def set_tidy3d_output(enabled: bool = True) -> None:
    """Toggle Tidy3D chatter and TARANG-managed web progress for this process.
    Disabling saves existing settings; enabling restores them, including Rich warnings.
    """
    import warnings
    import tidy3d as td

    global _tidy3d_state
    if not isinstance(
        enabled,
        bool,
    ):
        raise TypeError("enabled must be a boolean")
    with _lock:
        if not enabled and _tidy3d_state is None:
            loggers = [logging.getLogger(name) for name in ("tidy3d", "tidy3d.web")]
            state = {
                "level": td.config.logging.level,
                "loggers": [(logger, logger.level) for logger in loggers],
                "progress": getattr(
                    td.config,
                    "show_progress_bars",
                    None,
                ),
                "has_progress": hasattr(
                    td.config,
                    "show_progress_bars",
                ),
            }
            td.config.logging.level = "ERROR"
            if state["has_progress"]:
                td.config.show_progress_bars = False
            for logger in loggers:
                logger.setLevel(logging.ERROR)
            warnings.filterwarnings(
                "ignore",
                category=UserWarning,
                module=r"^rich(?:\.|$)(?#tarang-quiet)",
            )
            state["warning_filter"] = warnings.filters[0]
            _tidy3d_state = state
        elif enabled and _tidy3d_state is not None:
            state = _tidy3d_state
            td.config.logging.level = state["level"]
            if state["has_progress"]:
                td.config.show_progress_bars = state["progress"]
            for logger, level in state["loggers"]:
                logger.setLevel(level)
            warnings.filters[:] = [
                entry
                for entry in warnings.filters
                if entry is not state["warning_filter"]
            ]
            # Notify Python that warning registries must re-evaluate their filters.
            warnings._filters_mutated()
            _tidy3d_state = None


def tidy3d_verbose(requested: bool = True) -> bool:
    """Resolve verbosity for a TARANG-managed Tidy3D operation.
    Quiet mode overrides per-call progress requests until output is restored.
    """
    return bool(requested) and _tidy3d_state is None
