"""Start the MachineCare AI dashboard:   python run_app.py   (then open http://127.0.0.1:8050)"""

import argparse

from app.server import serve

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="MachineCare AI dashboard")
    ap.add_argument("--port", type=int, default=8050)
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args()
    serve(port=a.port, open_browser=not a.no_browser)
