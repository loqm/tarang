"""Console-safe output for the command-line tools.

On Windows, output redirected to a file or pipe uses the legacy code page
(cp1252), which cannot encode characters such as λ, α or μ. Printing them would
raise UnicodeEncodeError and abort the tool. ``safe_console()`` keeps the
encoding but replaces unencodable characters with '?'. Files the tools write are
always UTF-8 and unaffected.
"""
import sys


def safe_console() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):       # not a reconfigurable text stream
            pass
