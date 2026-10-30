"""Shared TARANG infrastructure.

Exposes configure_logging(), get_logger(), set_tidy3d_output() and
set_tidy3d_log_level().
Device imports retain their existing package names; __version__ identifies releases.
"""

from ._logging import (
    configure_logging,
    get_logger,
    set_tidy3d_log_level,
    set_tidy3d_output,
)

__all__ = [
    "configure_logging",
    "get_logger",
    "set_tidy3d_log_level",
    "set_tidy3d_output",
]

__version__ = "0.1.2"
