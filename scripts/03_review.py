"""Review the candidate questions in a browser.

    python scripts/03_review.py              # opens http://127.0.0.1:8765

Shows each candidate next to the section it came from, with its evidence
highlighted. Keep it (editing the question, answer or evidence as needed) or
drop it with a reason; write the questions the docs cannot answer yourself.

Every change is written to eval/review_state.json the moment it is made, so the
review can be stopped and picked up again at any point. Nothing leaves this
machine: the server listens on 127.0.0.1 only.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import corpus  # noqa: E402
from src import review as rv  # noqa: E402
from src.evalset import sections  # noqa: E402

CANDIDATES = ROOT / "eval" / "candidates.jsonl"
STATE = ROOT / "eval" / "review_state.json"
PAGE = Path(__file__).with_name("review.html")
MAX_BODY = 1_000_000


class Review:
    """Everything the page needs, and the only code that changes the state."""

    def __init__(self, state_path: Path = STATE) -> None:
        self.state_path = state_path
        self.docs = {d.doc_id: d for d in corpus.load_corpus()}
        self.sections = {s.key: s for d in self.docs.values() for s in sections(d)}
        rows = [json.loads(line) for line in CANDIDATES.open(encoding="utf-8") if line.strip()]
        self.candidates = [r for r in rows if r["status"] == "ok"]
        self.by_id = {c["candidate_id"]: c for c in self.candidates}
        self.state = rv.load_state(state_path)
        digest = rv.candidates_digest(CANDIDATES)
        if self.state["candidates_digest"] not in (None, digest) and self.state["decisions"]:
            print("WARNING: eval/candidates.jsonl changed since this review started; "
                  "decisions are matched by candidate id and may no longer fit.")
        self.state["candidates_digest"] = digest
        self.search_index = rv.SectionSearch(list(self.docs.values()))
        self.lock = threading.Lock()

    # ---------------------------------------------------------------- reads

    def section(self, key: str) -> dict:
        s = self.sections[key]
        d = self.docs[s.doc_id]
        return {"key": key, "doc_id": s.doc_id, "title": s.title, "page": d.title,
                "url": d.cite(s.start), "start": s.start, "end": s.end, "text": s.text(d)}

    def payload(self) -> dict:
        fields = ("candidate_id", "kind", "question", "answer", "evidence", "overlap")
        return {
            "candidates": [{**{f: c[f] for f in fields},
                            "sections": [self.section(k) for k in c["sections"]]}
                           for c in self.candidates],
            "drop_reasons": rv.DROP_REASONS,
            **self.progress(),
        }

    def progress(self) -> dict:
        return {"decisions": self.state["decisions"], "handwritten": self.state["handwritten"]}

    def search(self, query: str) -> list[dict]:
        return [{**self.section(s.key), "score": round(score, 2)}
                for score, s in self.search_index.search(query)]

    # --------------------------------------------------------------- writes

    def _evidence(self, items: list[dict], allowed: set[str] | None) -> list:
        out = []
        for item in items:
            key = item["section_key"]
            if key not in self.sections or (allowed is not None and key not in allowed):
                raise ValueError("evidence must come from the section(s) shown")
            s = self.sections[key]
            out.append(rv.evidence_from_selection(
                self.docs[s.doc_id], int(item["start"]), int(item["end"]),
                str(item["text"]), bounds=(s.start, s.end)))
        return rv.merge_evidence(out, self.docs)

    def apply(self, action: str, body: dict) -> dict:
        with self.lock:
            if action == "decide":
                cand = self.by_id[body["candidate_id"]]
                evidence = self._evidence(body.get("evidence", []), set(cand["sections"])) \
                    if body["decision"] == "keep" else None
                rv.decide(self.state, cand, body["decision"],
                          question=body.get("question", ""), answer=body.get("answer", ""),
                          evidence=evidence, reason=body.get("reason"))
            elif action == "undo":
                rv.undo(self.state, body["candidate_id"])
            elif action == "add":
                rv.add_handwritten(self.state, body.get("question", ""), body.get("answer"),
                                   self._evidence(body.get("evidence", []), None),
                                   body.get("note", ""))
            elif action == "delete":
                rv.delete_handwritten(self.state, body["hid"])
            else:
                raise ValueError(f"unknown action {action!r}")
            rv.save_state(self.state, self.state_path)
            return self.progress()


def make_handler(app: Review):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, status: int, body: bytes, kind: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, status: int, obj) -> None:
            self._send(status, json.dumps(obj, ensure_ascii=False).encode("utf-8"),
                       "application/json; charset=utf-8")

        def do_GET(self) -> None:  # noqa: N802
            url = urlparse(self.path)
            if url.path == "/":
                self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
            elif url.path == "/api/state":
                self._json(200, app.payload())
            elif url.path == "/api/search":
                self._json(200, app.search(parse_qs(url.query).get("q", [""])[0]))
            else:
                self._json(404, {"error": "not found"})

        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers.get("Content-Length", "0"))
            if length > MAX_BODY:
                return self._json(413, {"error": "request too large"})
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
                self._json(200, app.apply(urlparse(self.path).path.removeprefix("/api/"), body))
            except (ValueError, KeyError) as exc:
                self._json(400, {"error": str(exc)})

        def log_message(self, *args) -> None:  # keep the terminal quiet
            pass

    return Handler


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--state", type=Path, default=STATE,
                        help="where decisions are saved (default: eval/review_state.json)")
    args = parser.parse_args()

    app = Review(args.state)
    decided = len(app.state["decisions"])
    url = f"http://127.0.0.1:{args.port}/"
    print(f"{len(app.candidates)} candidates, {decided} decided so far, "
          f"{len(app.state['handwritten'])} written by hand")
    print(f"review at {url}  (Ctrl+C to stop; progress is saved as you go)")
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(app))
    if not args.no_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print(f"\nstopped; progress is in {args.state}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
