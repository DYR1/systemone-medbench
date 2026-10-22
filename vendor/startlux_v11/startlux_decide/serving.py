"""What a public server needs besides the decision path: API keys, files that are re-read when they change, the
OpenRouter models document, an access log without content, the opt-in request log with content, and TLS.

No key is ever written to a log or an error message: logs name a key by its key id (key_id). States, questions and
answers go only to the request log, which an operator turns on (RequestLog).
"""
from __future__ import annotations

import datetime
import hashlib
import hmac
import json
import logging
import os
import ssl
from dataclasses import dataclass
from typing import Any, Callable, Optional

log = logging.getLogger("startlux_decide.serving")
MIN_KEY_LENGTH = 16
DECISION_PATHS = ("/v1/systemone", "/v1/decisions")       # answered by the model; subject to the in-flight limit
OPEN_PATHS = ("/health", "/v1/health")                     # never need a key


class WatchedFile:
    """A file parsed at start and again whenever it changes (modification time, size or inode). The first read must
    succeed; when a later read fails, the last good value stays in use and a warning names the file and the problem
    (parsers must not put file content into their error messages)."""

    def __init__(self, path: str, parse: Callable[[bytes], Any], what: str):
        self.path, self.parse, self.what = path, parse, what
        self._signature = self._stat()
        with open(path, "rb") as f:
            self.value = parse(f.read())

    def _stat(self) -> tuple:
        st = os.stat(self.path)
        return st.st_mtime_ns, st.st_size, st.st_ino

    def get(self) -> Any:
        """The current value, re-read first when the file changed."""
        try:
            signature = self._stat()
        except OSError as e:
            log.warning("%s %s: cannot stat (%s); keeping the last version", self.what, self.path, type(e).__name__)
            return self.value
        if signature != self._signature:
            self._signature = signature                 # a bad version is not read again until it changes
            try:
                with open(self.path, "rb") as f:
                    self.value = self.parse(f.read())
                log.info("%s %s re-read", self.what, self.path)
            except (OSError, ValueError) as e:
                log.error("%s %s not applied, keeping the last version: %s", self.what, self.path, e)
        return self.value


def parse_keys(data: bytes) -> frozenset:
    """SHA-256 digests of the keys of a key file: one key per line, blank lines and lines starting with # ignored.
    A key is one word of printable ASCII, at least MIN_KEY_LENGTH characters. Errors name the line, never the key."""
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        raise ValueError("the file is not UTF-8 text") from None
    digests = set()
    for n, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if not (line.isascii() and line.isprintable()) or any(c.isspace() for c in line):
            raise ValueError(f"line {n}: a key must be one word of printable ASCII characters")
        if len(line) < MIN_KEY_LENGTH:
            raise ValueError(f"line {n}: a key must have at least {MIN_KEY_LENGTH} characters")
        digests.add(hashlib.sha256(line.encode()).digest())
    return frozenset(digests)


class ApiKeys:
    """Bearer-token check against a key file that is re-read when it changes. Keys are kept as SHA-256 digests and
    compared in constant time with every key, so neither the key that matched nor the key lengths show in timing."""

    def __init__(self, path: str):
        self.file = WatchedFile(path, parse_keys, "API key file")
        if not self.file.value:
            raise ValueError(f"API key file {path}: no keys")

    def check(self, authorization: Optional[str]) -> bool:
        """True when the Authorization header is "Bearer <key>" with a key of the file."""
        scheme, _, token = (authorization or "").partition(" ")
        token = token.strip()
        if scheme.lower() != "bearer" or not token:
            return False
        digest = hashlib.sha256(token.encode("utf-8", "surrogateescape")).digest()
        ok = False
        for key in self.file.get():
            ok |= hmac.compare_digest(digest, key)
        return ok


def key_id(authorization: Optional[str]) -> Optional[str]:
    """The key of an "Authorization: Bearer <key>" header named by the first 16 hex digits of its SHA-256, so that logs
    tell keys apart without holding them. The id of a key file line is sha256(line).hexdigest()[:16]."""
    scheme, _, token = (authorization or "").partition(" ")
    token = token.strip()
    if scheme.lower() != "bearer" or not token:
        return None
    return hashlib.sha256(token.encode("utf-8", "surrogateescape")).hexdigest()[:16]


def client_address(peer: Optional[str], headers: Any) -> dict:
    """Where a request came from: the TCP peer and the proxy headers as received. "ip" is the first X-Forwarded-For
    entry, else X-Real-IP, else the peer; a client can write its own X-Forwarded-For, so the whole chain is kept."""
    xff = headers.get("X-Forwarded-For")
    first = xff.split(",")[0].strip() if xff else None
    return {"ip": first or headers.get("X-Real-IP") or peer, "peer": peer, "x_forwarded_for": xff,
            "x_real_ip": headers.get("X-Real-IP"), "user_agent": headers.get("User-Agent")}


def parse_models_document(data: bytes) -> bytes:
    """The OpenRouter provider models document, unchanged, after checking that it is JSON with a non-empty "data"
    array of objects with a string "id"."""
    try:
        doc = json.loads(data)
    except ValueError as e:
        raise ValueError(f"not valid JSON ({e})") from None
    if not isinstance(doc, dict) or not isinstance(doc.get("data"), list) or not doc["data"]:
        raise ValueError('expected an object with a non-empty "data" array')
    for i, model in enumerate(doc["data"]):
        if not isinstance(model, dict) or not isinstance(model.get("id"), str) or not model["id"]:
            raise ValueError(f'data[{i}] needs a string "id"')
    return data


class AccessLog:
    """One JSON line per request, appended to a file: time, request id, method, route, status, total time, client IP,
    key id, and for decision requests the number of questions, the token counts and the phase timings. Never any
    content."""

    def __init__(self, path: str):
        self.path = path
        self._file = open(path, "a", buffering=1, encoding="utf-8")
        self._failed = False

    def write(self, record: dict) -> None:
        try:
            self._file.write(json.dumps(record, separators=(",", ":")) + "\n")
            self._failed = False
        except OSError as e:
            if not self._failed:
                log.error("access log %s: write failed (%s)", self.path, type(e).__name__)
            self._failed = True

    def close(self) -> None:
        self._file.close()


_IMAGE_EXT = {"image/png": ".png", "image/jpeg": ".jpg", "image/jpg": ".jpg", "image/webp": ".webp",
              "image/gif": ".gif", "image/bmp": ".bmp", "image/tiff": ".tif"}


class RequestLog:
    """The request log, for operators who keep what clients sent and what they got back (off unless configured).
    One JSON line per decision request in DIR/requests-YYYY-MM-DD.jsonl (UTC days): time, request id, route, status,
    the client address (client_address), the key id, the request body and the answers with their details and
    thoughts. Images are written once each to DIR/images/<2 hex digits>/<sha256><ext> and the line holds their
    references. The directory and its files are readable by the server's user only. A failed write never fails a
    request: it is logged once."""

    def __init__(self, directory: str):
        self.dir = directory
        os.makedirs(os.path.join(directory, "images"), mode=0o700, exist_ok=True)
        os.chmod(directory, 0o700)
        self._day: Optional[str] = None
        self._file = None
        self._failed = False

    def image(self, value: Any) -> dict:
        """One image of a request (base64 or data URI) -> its reference; the bytes are written when they are new."""
        from .images import decode

        try:
            raw, mime = decode(value)
        except ValueError:
            return {"unreadable": True, "chars": len(value) if isinstance(value, str) else None}
        sha = hashlib.sha256(raw).hexdigest()
        rel = os.path.join("images", sha[:2], sha + _IMAGE_EXT.get(mime, ".bin"))
        path = os.path.join(self.dir, rel)
        if not os.path.exists(path):
            os.makedirs(os.path.dirname(path), mode=0o700, exist_ok=True)
            tmp = f"{path}.{os.getpid()}.tmp"
            with os.fdopen(os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), "wb") as f:
                f.write(raw)
            os.replace(tmp, path)
        return {"sha256": sha, "mime": mime or None, "bytes": len(raw), "file": rel}

    def write(self, record: dict) -> None:
        try:
            day = record["time"][:10]
            if day != self._day:
                self._open(day)
            self._file.write(json.dumps(record, ensure_ascii=False, separators=(",", ":"), default=str) + "\n")
            self._failed = False
        except (OSError, TypeError, ValueError) as e:
            if not self._failed:
                log.error("request log %s: write failed (%s)", self.dir, type(e).__name__)
            self._failed = True

    def _open(self, day: str) -> None:
        self.close()
        fd = os.open(os.path.join(self.dir, f"requests-{day}.jsonl"), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        self._file, self._day = os.fdopen(fd, "a", buffering=1, encoding="utf-8"), day

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None


def utc_now() -> str:
    """ISO 8601 time in UTC with milliseconds."""
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def tls_context(certfile: str, keyfile: str) -> ssl.SSLContext:
    """Server TLS context: the certificate chain and its key, TLS 1.2 or newer."""
    ctx = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
    ctx.minimum_version = ssl.TLSVersion.TLSv1_2
    ctx.load_cert_chain(certfile, keyfile)
    return ctx


@dataclass
class ServeOptions:
    """Options for serving publicly; all off by default."""
    api_keys: Optional[ApiKeys] = None
    max_inflight: Optional[int] = None
    models_document: Optional[WatchedFile] = None
    models_path: str = "/openrouter/models"
    access_log: Optional[AccessLog] = None
    request_log: Optional[RequestLog] = None
