#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""OpenAI 兼容分流代理（零依赖：仅标准库，Python 3.8+）。

路线一：
  - prefill 重业务（长 prompt / 显式 -prefill 后缀）→ 单机 SGLang 后端
  - decode  重业务（短 prompt / 显式 -decode 后缀）→ 双机 Strata 后端

对外接口：
  POST /v1/chat/completions   流式与非流式
  POST /v1/completions
  GET  /v1/models             转发到 decode 后端
  GET  /health                两个后端都可达返回 200，否则 503
  GET  /route-stats           两个 leg 的请求数 / 错误数
"""
import argparse
import json
import sys
import threading
import time
from datetime import datetime
from http.client import HTTPConnection, HTTPSConnection
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

LEG_PREFILL = "prefill"
LEG_DECODE = "decode"
LEGS = (LEG_PREFILL, LEG_DECODE)

CHAT_PATH = "/v1/chat/completions"
COMPLETIONS_PATH = "/v1/completions"
MODELS_PATH = "/v1/models"
HEALTH_PATH = "/health"
STATS_PATH = "/route-stats"

PREFILL_SUFFIX = "-prefill"
DECODE_SUFFIX = "-decode"

HEALTH_TIMEOUT = 3.0
CHUNK_SIZE = 65536
CRLF = b"\r\n"

# 逐跳头 / 由本层自己管理的头：不原样透传
SKIP_REQUEST_HEADERS = {
    "host", "content-length", "connection", "keep-alive", "proxy-connection",
    "transfer-encoding", "te", "trailer", "upgrade", "accept-encoding",
}
SKIP_RESPONSE_HEADERS = {
    "content-length", "transfer-encoding", "connection", "keep-alive",
    "trailer", "upgrade", "server", "date",
}

CONFIG = {}


# ---------------------------------------------------------------------------
# 统计
# ---------------------------------------------------------------------------
class Stats(object):
    """两个 leg 各自的请求数与错误数。"""

    def __init__(self):
        self._lock = threading.Lock()
        self._data = dict((leg, {"requests": 0, "errors": 0}) for leg in LEGS)

    def record(self, leg):
        with self._lock:
            self._data[leg]["requests"] += 1

    def record_error(self, leg):
        with self._lock:
            self._data[leg]["errors"] += 1

    def snapshot(self):
        with self._lock:
            return dict((leg, dict(self._data[leg])) for leg in LEGS)


STATS = Stats()


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------
def parse_backend(url):
    """把后端 URL 拆成 scheme / host / port / base path。"""
    parts = urlsplit(url)
    scheme = (parts.scheme or "http").lower()
    host = parts.hostname or "127.0.0.1"
    port = parts.port or (443 if scheme == "https" else 80)
    base = (parts.path or "").rstrip("/")
    return {"url": url, "scheme": scheme, "host": host, "port": port, "base": base}


def _connect(backend, timeout):
    cls = HTTPSConnection if backend["scheme"] == "https" else HTTPConnection
    return cls(backend["host"], backend["port"], timeout=timeout)


def _text_len(value):
    """content / prompt 字段的字符数（字符串或多模态数组都尽量算全）。"""
    if isinstance(value, str):
        return len(value)
    if isinstance(value, (list, tuple)):
        total = 0
        for item in value:
            if isinstance(item, str):
                total += len(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    total += len(text)
                else:
                    total += _text_len(item.get("content"))
        return total
    if isinstance(value, dict):
        return _text_len(value.get("content"))
    return 0


def count_prompt_chars(path, payload):
    """chat 累加所有 message 的 content 长度；completions 取 prompt 字段。"""
    if not isinstance(payload, dict):
        return 0
    if path == COMPLETIONS_PATH:
        return _text_len(payload.get("prompt"))
    total = 0
    messages = payload.get("messages")
    if isinstance(messages, list):
        for message in messages:
            if isinstance(message, dict):
                total += _text_len(message.get("content"))
    return total


def decide_leg(model, payload, threshold, path=None):
    """返回 (leg, 转发用 model)：后缀强制路由优先，其次按 prompt 字符数。"""
    if isinstance(model, str):
        if model.endswith(PREFILL_SUFFIX):
            return LEG_PREFILL, model[: -len(PREFILL_SUFFIX)]
        if model.endswith(DECODE_SUFFIX):
            return LEG_DECODE, model[: -len(DECODE_SUFFIX)]
    chars = count_prompt_chars(path, payload)
    return (LEG_PREFILL if chars >= threshold else LEG_DECODE), model


def probe_backend(backend, timeout=HEALTH_TIMEOUT):
    """GET /health；无该路径或失败时退回 GET /。返回 (ok, detail)。"""
    detail = "unreachable"
    for path in (HEALTH_PATH, "/"):
        conn = None
        try:
            conn = _connect(backend, timeout)
            conn.request("GET", backend["base"] + path)
            resp = conn.getresponse()
            resp.read()
            status = resp.status
        except Exception as exc:
            detail = "%s -> %s" % (path, exc)
            continue
        finally:
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass
        detail = "%s -> %d" % (path, status)
        if 200 <= status < 500:
            return True, detail
    return False, detail


def log_route(model, chars, leg, backend, status, first_chunk, total):
    """每个请求落一行路由日志到 stderr。"""
    first = "-" if first_chunk is None else "%.3f" % first_chunk
    sys.stderr.write(
        "%s model=%s chars=%d leg=%s backend=%s status=%s first_chunk=%s total=%.3fs\n"
        % (
            datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
            model if isinstance(model, str) else "-",
            chars,
            leg,
            backend["url"],
            status,
            first,
            total,
        )
    )
    sys.stderr.flush()


def route_of(payload, threshold, path=None):
    """按请求体算出 (leg, 转发用 model, 字符数)。"""
    if not isinstance(payload, dict):
        payload = {}
    model = payload.get("model")
    chars = count_prompt_chars(path, payload)
    leg, out_model = decide_leg(model, payload, threshold, path)
    return leg, out_model, chars


# ---------------------------------------------------------------------------
# HTTP 处理
# ---------------------------------------------------------------------------
class RouterHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "route-split"
    sys_version = ""

    def log_message(self, fmt, *args):  # 屏蔽默认 access log，路由日志另行输出
        return

    # ---------------- 入口 ----------------
    def do_GET(self):
        path = self._norm_path()
        if path == HEALTH_PATH:
            self._handle_health()
        elif path == STATS_PATH:
            self._handle_stats()
        elif path == MODELS_PATH:
            self._handle_models()
        else:
            self._send_json(404, self._error_body("unknown path: %s" % path, "not_found"))

    def do_POST(self):
        path = self._norm_path()
        if path in (CHAT_PATH, COMPLETIONS_PATH):
            self._handle_inference(path)
        else:
            self._read_body()
            self._send_json(404, self._error_body("unknown path: %s" % path, "not_found"))

    # ---------------- 基础工具 ----------------
    def _norm_path(self):
        path = urlsplit(self.path).path.rstrip("/")
        return path or "/"

    @staticmethod
    def _error_body(message, code):
        return {"error": {"message": message, "type": "route_split_error", "code": code}}

    def _read_body(self):
        encoding = (self.headers.get("Transfer-Encoding") or "").lower()
        if "chunked" in encoding:
            chunks = []
            while True:
                line = self.rfile.readline().strip()
                if not line:
                    break
                if b";" in line:
                    line = line.split(b";")[0]
                try:
                    size = int(line, 16)
                except ValueError:
                    break
                if size == 0:
                    while True:
                        trailer = self.rfile.readline()
                        if trailer in (b"\r\n", b"\n", b""):
                            break
                    break
                chunks.append(self.rfile.read(size))
                self.rfile.read(2)
            return b"".join(chunks)
        length = self.headers.get("Content-Length")
        if not length:
            return b""
        try:
            return self.rfile.read(int(length))
        except (ValueError, TypeError):
            return b""

    def _send_json(self, status, payload):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response_only(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        try:
            self.wfile.write(body)
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _forward_headers(self):
        return dict(
            (k, v) for k, v in self.headers.items() if k.lower() not in SKIP_REQUEST_HEADERS
        )

    # ---------------- /health ----------------
    def _handle_health(self):
        results = {}
        lock = threading.Lock()

        def probe(leg, backend):
            ok, detail = probe_backend(backend, HEALTH_TIMEOUT)
            with lock:
                results[leg] = (ok, detail)

        threads = []
        for leg in LEGS:
            thread = threading.Thread(target=probe, args=(leg, CONFIG[leg]))
            thread.daemon = True
            thread.start()
            threads.append(thread)
        for thread in threads:
            thread.join(HEALTH_TIMEOUT + 2.0)

        down = [leg for leg in LEGS if not results.get(leg, (False, ""))[0]]
        payload = {
            "status": "ok" if not down else "degraded",
            "down": down,
            "backends": dict(
                (
                    leg,
                    {
                        "ok": results.get(leg, (False, ""))[0],
                        "detail": results.get(leg, (False, ""))[1],
                        "url": CONFIG[leg]["url"],
                    },
                )
                for leg in LEGS
            ),
        }
        if down:
            payload["message"] = "backend down: %s" % ", ".join(down)
        self._send_json(503 if down else 200, payload)

    # ---------------- /route-stats ----------------
    def _handle_stats(self):
        snapshot = STATS.snapshot()
        payload = {
            LEG_PREFILL: snapshot[LEG_PREFILL],
            LEG_DECODE: snapshot[LEG_DECODE],
            "char_threshold": CONFIG["char_threshold"],
            "prefill_backend": CONFIG[LEG_PREFILL]["url"],
            "decode_backend": CONFIG[LEG_DECODE]["url"],
        }
        self._send_json(200, payload)

    # ---------------- /v1/models ----------------
    def _handle_models(self):
        backend = CONFIG[LEG_DECODE]
        conn = None
        try:
            conn = _connect(backend, CONFIG["timeout"])
            conn.request("GET", backend["base"] + MODELS_PATH, headers={"Accept": "application/json"})
            resp = conn.getresponse()
            data = resp.read()
            status = resp.status
            headers = [kv for kv in resp.getheaders() if kv[0].lower() not in SKIP_RESPONSE_HEADERS]
        except Exception as exc:
            self._send_json(
                502,
                self._error_body(
                    "upstream %s backend unavailable: %s" % (LEG_DECODE, exc),
                    "upstream_unreachable",
                ),
            )
            return
        finally:
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass

        self.send_response_only(status)
        for key, value in headers:
            self.send_header(key, value)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        try:
            self.wfile.write(data)
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass

    # ---------------- 推理请求 ----------------
    def _handle_inference(self, path):
        start = time.time()
        raw_body = self._read_body()
        forward_headers = self._forward_headers()

        payload = None
        if raw_body:
            try:
                parsed = json.loads(raw_body.decode("utf-8"))
            except Exception:
                parsed = None
            if isinstance(parsed, dict):
                payload = parsed

        model = payload.get("model") if payload else None
        leg, out_model, chars = route_of(payload, CONFIG["char_threshold"], path)
        backend = CONFIG[leg]
        is_stream = bool(payload and payload.get("stream"))

        # 除 model 去后缀外，请求体字段原样透传
        out_body = raw_body
        if payload is not None and out_model != model:
            new_payload = dict(payload)
            new_payload["model"] = out_model
            out_body = json.dumps(new_payload, ensure_ascii=False).encode("utf-8")

        STATS.record(leg)

        conn = None
        first_chunk = None
        status = None
        try:
            conn = _connect(backend, CONFIG["timeout"])
            conn.request("POST", backend["base"] + path, body=out_body, headers=forward_headers)
            resp = conn.getresponse()
            status = resp.status
            if is_stream:
                first_chunk = self._relay_stream(resp, start)
            else:
                first_chunk = self._relay_buffered(resp, start)
        except (BrokenPipeError, ConnectionResetError):
            STATS.record_error(leg)
        except Exception as exc:
            STATS.record_error(leg)
            if status is None:
                log_route(model, chars, leg, backend, 502, None, time.time() - start)
                self._send_json(
                    502,
                    self._error_body(
                        "upstream %s backend unavailable: %s" % (leg, exc),
                        "upstream_unreachable",
                    ),
                )
                return
        finally:
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass

        total = time.time() - start
        log_route(model, chars, leg, backend, status, first_chunk, total)

    def _relay_buffered(self, resp, start):
        data = resp.read()
        first_chunk = time.time() - start
        self.send_response_only(resp.status, resp.reason)
        for key, value in resp.getheaders():
            if key.lower() not in SKIP_RESPONSE_HEADERS:
                self.send_header(key, value)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self._write(data)
        return first_chunk

    def _relay_stream(self, resp, start):
        """逐 chunk 原样转发并立即 flush，不缓冲整段。"""
        self.send_response_only(resp.status, resp.reason)
        content_length = None
        for key, value in resp.getheaders():
            lowered = key.lower()
            if lowered == "content-length":
                content_length = value
            elif lowered not in SKIP_RESPONSE_HEADERS:
                self.send_header(key, value)
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Accel-Buffering", "no")
        if content_length is not None:
            # 上游自带长度，按原长度直转
            self.send_header("Content-Length", content_length)
            chunked_out = False
        else:
            # SSE 常态：上游按连接关闭定界，本层改以 chunked 封装，边界即时可见
            self.send_header("Transfer-Encoding", "chunked")
            chunked_out = True
        self.end_headers()
        first_chunk = None
        while True:
            chunk = resp.read1(CHUNK_SIZE)
            if not chunk:
                break
            if first_chunk is None:
                first_chunk = time.time() - start
            if chunked_out:
                self._write(b"%x" % len(chunk) + CRLF + chunk + CRLF)
            else:
                self._write(chunk)
        if chunked_out:
            self._write(b"0" + CRLF + CRLF)
        if first_chunk is None:
            first_chunk = time.time() - start
        return first_chunk

    def _write(self, data):
        self.wfile.write(data)
        try:
            self.wfile.flush()
        except AttributeError:
            pass


# ---------------------------------------------------------------------------
# 启动
# ---------------------------------------------------------------------------
def build_config(args):
    return {
        LEG_PREFILL: parse_backend(args.prefill_backend),
        LEG_DECODE: parse_backend(args.decode_backend),
        "char_threshold": args.char_threshold,
        "timeout": args.timeout,
        "port": args.port,
        "host": args.host,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="OpenAI 兼容分流代理：prefill 重业务走 SGLang 单机，decode 重业务走 Strata 双机"
    )
    parser.add_argument("--host", default="0.0.0.0", help="监听地址（默认 0.0.0.0）")
    parser.add_argument("--port", type=int, default=8400, help="监听端口（默认 8400）")
    parser.add_argument("--prefill-backend", required=True, help="prefill 后端 base URL，如 http://127.0.0.1:30000")
    parser.add_argument("--decode-backend", required=True, help="decode 后端 base URL，如 http://127.0.0.1:30001")
    parser.add_argument(
        "--char-threshold",
        type=int,
        default=4096,
        help="prompt 字符数阈值，>= 阈值走 prefill（默认 4096）",
    )
    parser.add_argument("--timeout", type=float, default=600, help="后端请求超时秒数（默认 600）")
    return parser.parse_args(argv)


class RouterServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def main(argv=None):
    global CONFIG
    args = parse_args(argv)
    CONFIG = build_config(args)
    server = RouterServer((args.host, args.port), RouterHandler)
    sys.stderr.write(
        "route-split listening on %s:%d prefill=%s decode=%s threshold=%d timeout=%ss\n"
        % (
            args.host,
            args.port,
            CONFIG[LEG_PREFILL]["url"],
            CONFIG[LEG_DECODE]["url"],
            args.char_threshold,
            args.timeout,
        )
    )
    sys.stderr.flush()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
