"""python -m tarang.gui: open the TARANG design GUI in the browser.

    python -m tarang.gui                       # designs under ./design_runs
    python -m tarang.gui --root D:\\my\\designs # another folder
    python -m tarang.gui --port 8502           # another port (default 8501)

Stop it with Exit TARANG (bottom of the left panel) or Ctrl+C in this window. Runs started from the GUI keep going on their own;
their output is in <design>/gui_log.txt.
"""
from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m tarang.gui", description="TARANG design GUI.",
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--root", default="design_runs", help="folder holding your design folders")
    ap.add_argument("--port", type=int, default=8501, help="local port (default 8501)")
    ap.add_argument("--no-browser", action="store_true", help="do not open a browser tab")
    a = ap.parse_args(argv)
    if importlib.util.find_spec("streamlit") is None:
        print("The GUI needs Streamlit. Install it with:\n    pip install -e .[gui]\n"
              "(in the TARANG folder), or: pip install \"streamlit>=1.50\"")
        return 2
    app = Path(__file__).with_name("app.py")
    url = f"http://localhost:{a.port}"
    # headless: Streamlit would otherwise ask for an e-mail address on its first start;
    # the browser tab is opened here instead
    cmd = [sys.executable, "-m", "streamlit", "run", str(app), "--server.port", str(a.port),
           "--server.address", "localhost", "--server.headless", "true",
           "--browser.gatherUsageStats", "false", "--client.toolbarMode", "minimal",
           "--theme.primaryColor", "#2563eb",
           "--", "--root", str(Path(a.root).resolve())]
    print(f"TARANG GUI on {url}  (designs under {Path(a.root).resolve()}); Exit TARANG in its left panel, or Ctrl+C here, stops it.")
    proc = subprocess.Popen(cmd)
    if not a.no_browser:
        import time
        import urllib.request
        import webbrowser
        for _ in range(60):                       # wait until the server answers (<= 30 s)
            try:
                urllib.request.urlopen(url + "/_stcore/health", timeout=1)
                break
            except OSError:
                if proc.poll() is not None:
                    break
                time.sleep(0.5)
        if proc.poll() is None:
            webbrowser.open(url)
    try:
        proc.wait()
    except KeyboardInterrupt:
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.terminate()
    print("TARANG GUI stopped.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
