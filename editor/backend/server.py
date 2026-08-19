from __future__ import annotations

import json
import mimetypes
import re
import sys
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .core import Action, Snapshot, add_action, apply_repair, create_session, diagnose, image_data_url, image_from_data_url, replay, serialize_session
from .dataset import TraceCatalog


ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web"
SESSIONS = {}
CATALOG = TraceCatalog.discover(ROOT)


class ApiError(Exception):
    def __init__(self, status: int, message: str):
        self.status = status
        self.message = message


class Handler(BaseHTTPRequestHandler):
    server_version = "EditTrace/0.1"

    def log_message(self, fmt: str, *args) -> None:
        sys.stderr.write("[%s] %s\n" % (self.log_date_time_string(), fmt % args))

    def _json(self, payload, status=200):
        data = json.dumps(payload, separators=(",", ":")).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _body(self):
        length = int(self.headers.get("Content-Length", "0"))
        if length > 20_000_000:
            raise ApiError(413, "Request exceeds the 20 MB limit.")
        try:
            return json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError as exc:
            raise ApiError(400, "Malformed JSON body.") from exc

    def _session(self, session_id):
        if session_id not in SESSIONS:
            raise ApiError(404, "Session not found.")
        return SESSIONS[session_id]

    def do_GET(self):
        try:
            path = urlparse(self.path).path
            if path == "/api/health":
                from .core import MODEL
                return self._json({"status": "ok", "model": getattr(MODEL, "weights_id", "surrogate-visual-v1"), "mode": "EXPERIMENTAL" if getattr(MODEL, "experimental", False) else "DEMO"})
            if path == "/api/dataset":
                return self._json(CATALOG.status() if CATALOG else {"configured": False, "expected_data_root": str(ROOT / "data")})
            if path == "/api/traces":
                if not CATALOG:
                    raise ApiError(404, "No trace catalog is configured.")
                query = parse_qs(urlparse(self.path).query)
                return self._json(CATALOG.listing(query.get("q", [""])[0], int(query.get("offset", [0])[0]), min(500, int(query.get("limit", [100])[0]))))
            match = re.fullmatch(r"/api/sessions/([^/]+)", path)
            if match:
                return self._json(serialize_session(self._session(match.group(1))))
            self._static(path)
        except ApiError as exc:
            self._json({"error": exc.message}, exc.status)
        except Exception as exc:
            self._json({"error": str(exc)}, 500)

    def do_POST(self):
        try:
            path = urlparse(self.path).path
            body = self._body()
            if path == "/api/dataset/refresh":
                global CATALOG
                CATALOG = TraceCatalog.discover(ROOT)
                return self._json(CATALOG.status() if CATALOG else {"configured": False, "expected_data_root": str(ROOT / "data")})
            if path == "/api/traces/open":
                if not CATALOG:
                    raise ApiError(404, "No trace catalog is configured.")
                try:
                    session = CATALOG.open_session(body.get("trace_id", ""))
                except KeyError as exc:
                    raise ApiError(404, str(exc)) from exc
                except FileNotFoundError as exc:
                    raise ApiError(404, str(exc)) from exc
                SESSIONS[session.id] = session
                return self._json(serialize_session(session), HTTPStatus.CREATED)
            if path == "/api/sessions":
                image = image_from_data_url(body["image"]) if body.get("image") else None
                session = create_session(image, body.get("filename", "edittrace-starter.png"))
                SESSIONS[session.id] = session
                return self._json(serialize_session(session), HTTPStatus.CREATED)
            match = re.fullmatch(r"/api/sessions/([^/]+)/(actions|diagnose|preview|apply-repair|undo|reset)", path)
            if not match:
                raise ApiError(404, "Endpoint not found.")
            session = self._session(match.group(1))
            command = match.group(2)
            if command == "actions":
                add_action(session, body.get("type", ""), body.get("params", {}))
                return self._json(serialize_session(session), HTTPStatus.CREATED)
            if command == "diagnose":
                exploratory = bool(body.get("allow_exploratory", False))
                if session.dataset_mode and session.eligible is not True and not exploratory:
                    raise ApiError(409, f"Frozen-protocol diagnosis blocked: {session.eligibility_reason}.")
                if session.dataset_mode and session.replay_valid is not True and not exploratory:
                    raise ApiError(409, f"Frozen-protocol diagnosis blocked: {session.replay_reason}.")
                report = diagnose(session)
                report["analysis_mode"] = "exploratory" if session.dataset_mode and (session.eligible is not True or session.replay_valid is not True) else "confirmatory"
                report["protocol_gates"] = {
                    "eligible": session.eligible,
                    "eligibility_reason": session.eligibility_reason,
                    "replay_valid": session.replay_valid,
                    "replay_reason": session.replay_reason,
                }
                return self._json(report)
            if command == "preview":
                image, _ = replay(session, set(body.get("remove", [])), body.get("overrides", {}))
                from .core import MODEL
                return self._json({"image": image_data_url(image), "prediction": MODEL.prediction(image, session.target_class)})
            if command == "apply-repair":
                report = diagnose(session)
                candidate = next((item for item in report["repairs"] if item["id"] == body.get("candidate_id")), None)
                if not candidate:
                    raise ApiError(409, "Repair candidate is stale or unavailable.")
                apply_repair(session, candidate)
                return self._json(serialize_session(session))
            if command == "undo":
                if not session.undo:
                    raise ApiError(409, "Nothing to undo.")
                session.actions = session.undo.pop().actions
                return self._json(serialize_session(session))
            if command == "reset":
                session.undo.append(Snapshot(actions=[Action(**vars(item)) for item in session.actions]))
                session.actions = []
                return self._json(serialize_session(session))
        except ApiError as exc:
            self._json({"error": exc.message}, exc.status)
        except (ValueError, KeyError) as exc:
            self._json({"error": str(exc)}, 400)
        except Exception as exc:
            self._json({"error": str(exc)}, 500)

    def _static(self, path):
        relative = "index.html" if path in {"", "/"} else path.lstrip("/")
        target = (WEB / relative).resolve()
        if WEB not in target.parents or not target.is_file():
            target = WEB / "index.html"
        data = target.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", mimetypes.guess_type(target.name)[0] or "application/octet-stream")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def main():
    host, port = "127.0.0.1", 8017
    if "--host" in sys.argv:
        host = sys.argv[sys.argv.index("--host") + 1]
    if "--port" in sys.argv:
        port = int(sys.argv[sys.argv.index("--port") + 1])
    print(f"EditTrace-XAI running at http://{host}:{port}")
    ThreadingHTTPServer((host, port), Handler).serve_forever()


if __name__ == "__main__":
    main()
