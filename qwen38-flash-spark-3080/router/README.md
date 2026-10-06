# route-split · OpenAI 兼容分流代理

零依赖（仅 Python 标准库，3.8+）的 OpenAI 兼容反向代理，按 prompt 体量把请求分流到两条 leg：

| leg | 承载业务 | 典型后端 |
| --- | --- | --- |
| `prefill` | prefill 重（长 prompt、长上下文、批量预填） | SGLang 单机 |
| `decode` | decode 重（短 prompt、低延迟交互） | Strata 双机 |

文件：

- `router.py` —— 代理本体（单文件，`ThreadingHTTPServer` 并发）
- `selftest.py` —— 自验收脚本（进程内起两个 mock 后端 + router，逐项断言）
- `README.md` —— 本文件

## 启动

```bash
# 最简：默认 0.0.0.0:8400，阈值 4096 字符，超时 600s
python router.py --prefill-backend http://127.0.0.1:30000 \
                 --decode-backend  http://127.0.0.1:30001

# 完整参数
python router.py \
  --host 0.0.0.0 \
  --port 8400 \
  --prefill-backend http://127.0.0.1:30000 \
  --decode-backend  http://127.0.0.1:30001 \
  --char-threshold 4096 \
  --timeout 600
```

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--host` | `0.0.0.0` | 监听地址 |
| `--port` | `8400` | 监听端口 |
| `--prefill-backend` | 必填 | prefill 后端 base URL，可带路径前缀（如 `http://10.0.0.1:30000/openai`） |
| `--decode-backend` | 必填 | decode 后端 base URL，同上 |
| `--char-threshold` | `4096` | prompt 字符数阈值，`>=` 走 prefill，`<` 走 decode |
| `--timeout` | `600` | 转发到后端的读超时（秒） |

客户端把 `base_url` 指到本代理即可，其余不变：

```bash
curl http://127.0.0.1:8400/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer <你的后端 key>' \
  -d '{"model":"Qwen3.8-27B","messages":[{"role":"user","content":"你好"}],"stream":true}'
```

后端 `--timeout` 建议按最慢的后端设；流式请求本层不做整段缓冲，超时只作用于连接建立与后端首字节之外的读空闲。

## 路由规则

1. `model` 以 `-prefill` 结尾 → 强制走 prefill 后端；以 `-decode` 结尾 → 强制走 decode 后端。
   转发时后缀会被剥掉（`qwen3-27b-prefill` → `qwen3-27b`），其余字段一字不改。
2. 否则统计 prompt 字符数：
   - `/v1/chat/completions`：所有 `message.content` 字符串长度之和（多模态数组按其中 `text` 累加）。
   - `/v1/completions`：`prompt` 字段长度（字符串按字符数，数组按元素累加）。
   - `>= --char-threshold` 走 prefill，否则走 decode。

## 接口

| 方法 | 路径 | 行为 |
| --- | --- | --- |
| POST | `/v1/chat/completions` | 按规则分流，stream / 非 stream 均支持 |
| POST | `/v1/completions` | 同上 |
| GET | `/v1/models` | 转发到 decode 后端 |
| GET | `/health` | 两个后端都可达 → 200；任一不可达 → 503，响应体 `down` 写明是哪条 leg |
| GET | `/route-stats` | 两条 leg 各自的 `requests` / `errors`，附阈值与后端地址 |

其他路径返回 404 JSON。

转发行为：

- 请求体原样透传（除 `model` 去后缀这一处改写）。
- `Authorization` 等客户端头逐字透传（`Host`、`Content-Length`、`Connection` 等逐跳头由本层重建）。
- 流式响应逐 chunk 转发并立即 flush，不缓冲整段（上游未给 `Content-Length` 时本层改用 chunked 封装，保证下游边界即时可见）。
- 后端连接失败 / 不可达 → HTTP 502，JSON 错误体：
  `{"error": {"message": "...", "type": "route_split_error", "code": "upstream_unreachable"}}`
- 每个请求向 stderr 落一行路由日志：
  `时间 model=... chars=... leg=... backend=... status=... first_chunk=... total=...s`

`/health` 的后端探活：先 `GET <backend>/health`（3 秒超时），路径不存在时退回 `GET <backend>/`；返回 2xx~4xx 视为该后端存活。

## 怎么换后端

- 换地址：改启动参数 `--prefill-backend` / `--decode-backend` 后重启，或让部署脚本按环境注入这两个变量。
- 换阈值：`--char-threshold`。长上下文只增不减时调大，交互延迟变差时调小（把更多请求推向 decode 双机）。
- 加第三条 leg：在 `router.py` 顶部 `LEGS` 与 `build_config()` 里加一个名字 + 参数，再在 `decide_leg()` 返回它，其余转发/统计/健康检查代码自动覆盖。

## 自验收

```bash
python selftest.py
```

进程内起 `PREFILL-MOCK` / `DECODE-MOCK` 两个 mock 后端与一个 router，覆盖：短 prompt 走 decode、5000 字符走 prefill、`-prefill` / `-decode` 后缀强制路由且转发体已去后缀、stream 逐 chunk 到达且带正确标记、非 stream 正常、`/v1/models` 转发、`/route-stats` 计数、`/health` 在两后端在线时 200 且停掉一个后变 503 并点名、后端不可达时转发 502 JSON。全部 PASS 时打印 `SELFTEST_ALL_PASS` 并以 0 退出；有失败项则打印失败清单并以 1 退出。
