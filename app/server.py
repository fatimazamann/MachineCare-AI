"""MachineCare AI dashboard - a small web server (Python standard library + the trained model)."""

from __future__ import annotations

import json
import mimetypes
import os
import sys
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ml.kbs_rules import explain  # noqa: E402
from ml.portable import PortableLogit  # noqa: E402
from ml.preprocess import FAILURES  # noqa: E402

STATIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
FIG = os.path.join(ROOT, "figures")
MODELS = os.path.join(ROOT, "models")
RESULTS = os.path.join(ROOT, "results", "results.json")
REPLAY = os.path.join(ROOT, "results", "replay.csv.gz")

# The model is loaded from plain JSON (works with any scikit-learn/pandas version); the joblib file is a fallback.
try:
    if os.path.exists(os.path.join(MODELS, "leak_model.json")):
        _m = PortableLogit(os.path.join(MODELS, "leak_model.json"))
        MODEL = {"model": _m, "features": _m.features, "name": _m.name}
    else:
        MODEL = joblib.load(os.path.join(MODELS, "leak_model.joblib"))
    LOOKUP = pd.read_csv(os.path.join(MODELS, "lookup.csv.gz"), index_col=0, parse_dates=True)
except FileNotFoundError:
    sys.exit("No trained model found. Run:  python train.py   first.")
except Exception as exc:
    sys.exit(f"Could not load the saved model ({exc}). Run:  python train.py   to retrain it with your versions.")
REPLAY_DF = pd.read_csv(REPLAY, index_col=0, parse_dates=True)

# the five summary values a technician can read off the train's monitoring screen
FIELDS = ["duty_pct_30", "current_mean_a_30", "oil_temp_c_max_30", "panel_pressure_min_bar_30", "dryer_dv_pressure_bar_30"]
LOOKUP_COLS = ["loaded_frac_30", "current_mean_30", "oil_temp_max_30", "tp3_min_30", "dv_pressure_mean_30"]
SCALE = np.array([0.10, 0.5, 2.0, 0.15, 0.3])     # what counts as a "big" difference for each value

EPISODES = {f"Failure {i} ({pd.Timestamp(s):%d %b})": (pd.Timestamp(s) - pd.Timedelta(hours=12), pd.Timestamp(e) + pd.Timedelta(hours=12))
            for i, (s, e, _, _) in enumerate(FAILURES, 1)}
EPISODES["Unreported episode (29 Mar)"] = (pd.Timestamp("2020-03-28 12:00"), pd.Timestamp("2020-03-30 06:00"))
EPISODES["Normal day (10 Jun)"] = (pd.Timestamp("2020-06-10 00:00"), pd.Timestamp("2020-06-11 00:00"))


def nearest_minute(vals: np.ndarray):
    """Most similar real minute in the data, so the ML model sees a complete, realistic feature row."""
    target = vals.copy()
    target[0] = target[0] / 100                                  # duty % -> fraction
    d = ((LOOKUP[LOOKUP_COLS].to_numpy() - target) / SCALE) ** 2
    i = int(np.argmin(d.sum(axis=1)))
    return LOOKUP.iloc[[i]], float(np.sqrt(d[i].sum()))


def predict(inp: dict) -> dict:
    meas = {f: float(inp[f]) for f in FIELDS}
    meas["lps_active"] = "yes" if str(inp.get("lps_active", "no")).lower() in ("yes", "true", "1") else "no"
    row, dist = nearest_minute(np.array([meas[f] for f in FIELDS]))
    p_ml = float(MODEL["model"].predict_proba(row[MODEL["features"]])[0, 1])
    kbs = explain(meas, policy=inp.get("policy", "balanced"))
    faults = {f["id"]: f["cf"] for f in kbs["faults"]}
    cf = max(faults.get("air_leak", 0.0), 0.0)
    hybrid = (p_ml + cf) / 2
    top = [f for f in kbs["faults"] if f["cf"] >= 0.2]
    return {
        "input": meas,
        "ml": {"model": MODEL["name"], "leak_probability": round(p_ml, 4), "prediction": "Air leak" if p_ml >= 0.5 else "Normal",
               "matched_minute": str(row.index[0]), "match_distance": round(dist, 2)},
        "rules": {"leak_cf": round(cf, 3), "faults": [{"id": f["id"], "name": f["name"], "cf": round(f["cf"], 3), "status": f["status"]} for f in kbs["faults"]],
                  "causes": kbs["causes"], "safety": kbs["safety"], "prediction": "Air leak" if cf >= 0.5 else "Normal",
                  "summary": kbs["summary"], "how": top[0]["how_text"] if top else [],
                  "plan": [{"step": s["step"], "name": s["name"], "when": s["when"], "stage": s["stage"]} for s in kbs["plan"]["steps"]],
                  "urgency": kbs["plan"]["overall_urgency"], "urgency_label": kbs["plan"]["overall_label"]},
        "hybrid": {"score": round(hybrid, 4), "prediction": "Air leak" if hybrid >= 0.5 else "Normal"},
        "agreement": (p_ml >= 0.5) == (cf >= 0.5),
    }


def replay(name: str) -> dict:
    s, e = EPISODES[name]
    d = REPLAY_DF.loc[s:e]
    return {"name": name, "t": [t.strftime("%d %b %H:%M") for t in d.index],
            **{c: [None if pd.isna(v) else round(float(v), 3) for v in d[c]] for c in d.columns}}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, body: bytes, ctype: str, status=200):
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, status=200):
        self._send(json.dumps(obj).encode(), "application/json; charset=utf-8", status)

    def _file(self, base, rel):
        full = os.path.normpath(os.path.join(base, rel))
        if not full.startswith(base) or not os.path.isfile(full):
            return self._send(b"not found", "text/plain", 404)
        ctype = mimetypes.guess_type(full)[0] or "application/octet-stream"
        with open(full, "rb") as fh:
            self._send(fh.read(), ctype + ("; charset=utf-8" if ctype.startswith("text/") else ""))

    def do_GET(self):
        url = urlparse(self.path)
        if url.path == "/api/results":
            with open(RESULTS, encoding="utf-8") as fh:
                return self._send(fh.read().encode(), "application/json; charset=utf-8")
        if url.path == "/api/episodes":
            return self._json(list(EPISODES))
        if url.path == "/api/replay":
            name = parse_qs(url.query).get("name", [""])[0]
            if name not in EPISODES:
                return self._json({"error": "unknown episode"}, 404)
            return self._json(replay(name))
        if url.path.startswith("/figures/"):
            return self._file(FIG, url.path[len("/figures/"):])
        return self._file(STATIC, "index.html" if url.path in ("", "/") else url.path.lstrip("/"))

    def do_POST(self):
        try:
            n = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(n) or b"{}")
            if urlparse(self.path).path == "/api/predict":
                return self._json(predict(body))
            return self._json({"error": "unknown endpoint"}, 404)
        except (KeyError, ValueError) as exc:
            return self._json({"error": f"bad input: {exc}"}, 400)
        except Exception as exc:  # pragma: no cover
            traceback.print_exc()
            return self._json({"error": str(exc)}, 500)


def serve(host="127.0.0.1", port=8050, open_browser=True):
    httpd = ThreadingHTTPServer((host, port), Handler)
    url = f"http://{host}:{port}/"
    print(f"MachineCare AI dashboard at {url}  (Ctrl+C to stop)")
    if open_browser:
        try:
            import webbrowser
            webbrowser.open(url)
        except Exception:
            pass
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
