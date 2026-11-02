# OVNative Web UI

A tiny, dependency-free local web frontend for the native engine.

```bash
# from the repository root
./cpp/build/ovweb output/weights/omnivoice-lm-int8.ovw \
  output/weights/omnivoice-at-fp32.ovw output/weights/tokenizer.ovt \
  --backend cuda --port 8080
```

Open <http://127.0.0.1:8080>.

## Endpoints

| Method | Path | Body | Response |
|---|---|---|---|
| GET | `/api/health` | — | `{"ok":true,"cuda":true,"sample_rate":24000}` |
| POST | `/api/ref` | raw WAV | `{"id":"r..."}` |
| POST | `/api/tts` | `application/x-www-form-urlencoded` | `audio/wav` |
| GET | `/*` | — | static files from `--web-root` |

TTS form fields: `text`, `language`, `instruct`, `steps`, `cfg`, `seed`,
`ref_id`, `ref_text`, `trim`, `max_pause_ms`, `keep_pause_ms`.

The server binds to `127.0.0.1` by default. It has no authentication; do not
expose it to an untrusted network.
