#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""route-split 自验收：进程内起两个 mock 后端 + router，逐项断言。

运行：python selftest.py
全部 PASS 时打印 SELFTEST_ALL_PASS，否则打印失败项并以退出码 1 结束。
"""
import http.client
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import router as router_mod  # noqa: E402

STREAM_GAP = 0.25
RECORDS = {}
REC_LOCK = threading.Lock()
FAILURES = []


def check(name, ok, detail=""):
    print("%s - %s%s" % ("PASS" if ok else "FAIL", name, (" | " + detail) if detail else ""))
    sys.stdout.flush()
    if not ok:
        FAILURES.append(name)


def record(tag, path, body, headers):
    try:
        payload = json.loads(body.decode("utf-8")) if body else None
    except Exception:
        payload = None
    with REC_LOCK:
        RECORDS.setdefault(tag, []).append(
            {"path": path, "payload": payload, "authorization": headers.get("Authorization")}
        )


def last_record(tag):
    with REC_LOCK:
        items = RECORDS.get(tag) or []
    return items[-1] if items else None


def make_mock_handler(tag):
    class MockHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):
            return

        def _json(self, status, payload):
            body = json.dumps(payload).encode("utf-8")
            self.send_response_only(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            self.wfile.flush()

        def do_GET(self):
            path = self.path.rstrip("/")
            if path == "/health":
                self._json(200, {"status": "ok", "mock": tag})
            elif path == "/v1/models":
                self._json(200, {"object": "list", "data": [{"id": "mock-model", "mock": tag}]})
            else:
                self._json(404, {"error": {"message": "not found"}})

        def do_POST(self):
            length = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(length) if length else b""
            record(tag, self.path, body, self.headers)
            try:
                payload = json.loads(body.decode("utf-8")) or {}
            except Exception:
                payload = {}
            if payload.get("stream"):
                self._sse()
            else:
                self._json(
                    200,
                    {"mock": tag, "choices": [{"message": {"content": "MARKER-%s-NONSTREAM" % tag}}]},
                )

        def _sse(self):
            self.send_response_only(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            events = [
                'data: {"choices":[{"delta":{"content":"%s-A"}}]}\n\n' % tag,
                'data: {"choices":[{"delta":{"content":"%s-B"}}]}\n\n' % tag,
                "data: [DONE]\n\n",
            ]
            for index, event in enumerate(events):
                if index:
                    time.sleep(STREAM_GAP)
                data = event.encode("utf-8")
                self.wfile.write(b"%x\r\n" % len(data) + data + b"\r\n")
                self.wfile.flush()
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()

    return MockHandler


class MockBackend(object):
    def __init__(self, tag):
        self.tag = tag
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), make_mock_handler(tag))
        self.server.daemon_threads = True
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, args=(0.05,))
        self.thread.daemon = True

    @property
    def url(self):
        return "http://127.0.0.1:%d" % self.port

    def start(self):
        self.thread.start()
        return self

    def stop(self):
        try:
            self.server.shutdown()
        except Exception:
            pass
        try:
            self.server.server_close()
        except Exception:
            pass


class RouterProcess(object):
    """进程内起 router（等价于 python router.py --...）。"""

    def __init__(self, prefill_url, decode_url, threshold=4096):
        args = router_mod.parse_args(
            [
                "--host", "127.0.0.1",
                "--port", "0",
                "--prefill-backend", prefill_url,
                "--decode-backend", decode_url,
                "--char-threshold", str(threshold),
                "--timeout", "30",
            ]
        )
        router_mod.CONFIG = router_mod.build_config(args)
        self.server = router_mod.RouterServer(("127.0.0.1", 0), router_mod.RouterHandler)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, args=(0.05,))
        self.thread.daemon = True

    def start(self):
        self.thread.start()
        return self

    def stop(self):
        try:
            self.server.shutdown()
        except Exception:
            pass
        try:
            self.server.server_close()
        except Exception:
            pass


def request(method, path, payload=None, auth=True):
    conn = http.client.HTTPConnection("127.0.0.1", ROUTER.port, timeout=30)
    headers = {"Content-Type": "application/json"}
    if auth:
        headers["Authorization"] = "Bearer test-key"
    body = json.dumps(payload).encode("utf-8") if payload is not None else None
    conn.request(method, path, body=body, headers=headers)
    resp = conn.getresponse()
    data = resp.read()
    status = resp.status
    conn.close()
    return status, data


def post_json(path, payload):
    status, data = request("POST", path, payload)
    try:
        parsed = json.loads(data.decode("utf-8"))
    except Exception:
        parsed = None
    return status, parsed, data


def post_stream(path, payload):
    """逐 chunk 读流，返回 (status, 文本, 事件到达时间列表)。"""
    conn = http.client.HTTPConnection("127.0.0.1", ROUTER.port, timeout=30)
    headers = {"Content-Type": "application/json", "Authorization": "Bearer test-key"}
    conn.request("POST", path, body=json.dumps(payload).encode("utf-8"), headers=headers)
    resp = conn.getresponse()
    status = resp.status
    buf = b""
    events = []
    stamps = []
    while True:
        piece = resp.read(1)
        if not piece:
            break
        buf += piece
        if buf.endswith(b"\n\n"):
            events.append(buf.decode("utf-8", "replace"))
            stamps.append(time.time())
            buf = b""
    conn.close()
    return status, "".join(events), stamps


PREFILL = MockBackend("PREFILL-MOCK").start()
DECODE = MockBackend("DECODE-MOCK").start()
ROUTER = RouterProcess(PREFILL.url, DECODE.url).start()
time.sleep(0.2)

print("mock prefill=%s decode=%s router=%d" % (PREFILL.url, DECODE.url, ROUTER.port))

try:
    # 1. 短 prompt → decode
    status, parsed, raw = post_json(
        "/v1/chat/completions",
        {"model": "mock-model", "messages": [{"role": "user", "content": "hi"}], "stream": False},
    )
    rec = last_record("DECODE-MOCK")
    check(
        "short prompt routed to decode backend",
        status == 200 and parsed and parsed.get("mock") == "DECODE-MOCK"
        and rec and rec["path"] == "/v1/chat/completions",
        "status=%s mock=%s" % (status, parsed and parsed.get("mock")),
    )

    # 2. 5000 字符长 prompt → prefill
    long_text = "x" * 5000
    status, parsed, raw = post_json(
        "/v1/chat/completions",
        {"model": "mock-model", "messages": [{"role": "user", "content": long_text}], "stream": False},
    )
    rec = last_record("PREFILL-MOCK")
    check(
        "5000-char prompt routed to prefill backend",
        status == 200 and parsed and parsed.get("mock") == "PREFILL-MOCK"
        and rec and rec["payload"].get("model") == "mock-model",
        "status=%s mock=%s" % (status, parsed and parsed.get("mock")),
    )

    # 3. -prefill 后缀强制 prefill（短 prompt 也不走 decode）
    status, parsed, raw = post_json(
        "/v1/chat/completions",
        {"model": "mock-model-prefill", "messages": [{"role": "user", "content": "hi"}], "stream": False},
    )
    rec = last_record("PREFILL-MOCK")
    check(
        "-prefill suffix forces prefill backend",
        status == 200 and parsed and parsed.get("mock") == "PREFILL-MOCK"
        and rec and rec["payload"].get("model") == "mock-model",
        "status=%s model_forwarded=%s" % (status, rec and rec["payload"].get("model")),
    )

    # 4. -decode 后缀强制 decode，且转发体 model 已去后缀
    status, parsed, raw = post_json(
        "/v1/chat/completions",
        {"model": "mock-model-decode", "messages": [{"role": "user", "content": long_text}], "stream": False},
    )
    rec = last_record("DECODE-MOCK")
    check(
        "-decode suffix forces decode backend and strips suffix",
        status == 200 and parsed and parsed.get("mock") == "DECODE-MOCK"
        and rec and rec["payload"].get("model") == "mock-model"
        and rec["authorization"] == "Bearer test-key",
        "status=%s model_forwarded=%s auth=%s"
        % (status, rec and rec["payload"].get("model"), rec and rec["authorization"]),
    )

    # 5. stream 逐 chunk 到达 + 标记正确
    status, text, stamps = post_stream(
        "/v1/chat/completions",
        {"model": "mock-model", "messages": [{"role": "user", "content": "hi"}], "stream": True},
    )
    spread = (stamps[-1] - stamps[0]) if len(stamps) >= 2 else 0.0
    check(
        "stream chunks arrive incrementally with decode marker",
        status == 200 and len(stamps) == 3 and "DECODE-MOCK-A" in text and "DECODE-MOCK-B" in text
        and "[DONE]" in text and spread >= STREAM_GAP * 0.6,
        "status=%s events=%d spread=%.3fs" % (status, len(stamps), spread),
    )

    status, text, stamps = post_stream(
        "/v1/completions",
        {"model": "mock-model", "prompt": long_text, "stream": True},
    )
    spread = (stamps[-1] - stamps[0]) if len(stamps) >= 2 else 0.0
    check(
        "stream on /v1/completions routed to prefill with marker",
        status == 200 and "PREFILL-MOCK-A" in text and "[DONE]" in text and spread >= STREAM_GAP * 0.6,
        "status=%s events=%d spread=%.3fs" % (status, len(stamps), spread),
    )

    # 6. 非 stream /v1/completions 正常
    status, parsed, raw = post_json(
        "/v1/completions", {"model": "mock-model", "prompt": "hello", "stream": False}
    )
    check(
        "non-stream /v1/completions OK via decode",
        status == 200 and parsed and "MARKER-DECODE-MOCK-NONSTREAM" in json.dumps(parsed),
        "status=%s" % status,
    )

    # 7. /v1/models 转发
    status, data = request("GET", "/v1/models")
    check("GET /v1/models proxied", status == 200 and b"mock-model" in data, "status=%s" % status)

    # 8. /route-stats 计数
    status, data = request("GET", "/route-stats")
    stats = json.loads(data.decode("utf-8"))
    check(
        "GET /route-stats counts both legs",
        status == 200 and stats["prefill"]["requests"] >= 3 and stats["decode"]["requests"] >= 3
        and stats["prefill"]["errors"] == 0 and stats["decode"]["errors"] == 0,
        "prefill=%s decode=%s" % (stats["prefill"], stats["decode"]),
    )

    # 9. 两个 mock 都在 → /health 200
    status, data = request("GET", "/health")
    check("GET /health 200 while both backends up", status == 200, "status=%s" % status)

    # 10. 停掉 prefill mock → /health 503 且写明哪个后端挂了
    PREFILL.stop()
    time.sleep(0.3)
    status, data = request("GET", "/health")
    body = json.loads(data.decode("utf-8"))
    check(
        "GET /health 503 and names the dead backend",
        status == 503 and body.get("down") == ["prefill"] and "prefill" in body.get("message", ""),
        "status=%s down=%s" % (status, body.get("down")),
    )

    # 11. 后端不可达时转发返回 502 JSON
    status, parsed, raw = post_json(
        "/v1/chat/completions",
        {"model": "mock-model-prefill", "messages": [{"role": "user", "content": "hi"}], "stream": False},
    )
    check(
        "502 JSON error body when backend unreachable",
        status == 502 and parsed and "error" in parsed,
        "status=%s body=%s" % (status, raw[:120]),
    )
finally:
    PREFILL.stop()
    DECODE.stop()
    ROUTER.stop()

if FAILURES:
    print("SELFTEST_FAILED: %d item(s) -> %s" % (len(FAILURES), "; ".join(FAILURES)))
    sys.exit(1)

print("SELFTEST_ALL_PASS")
sys.exit(0)
