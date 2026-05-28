# Running & Deploying the Allergo Probe Analyzer

A practical, copy-paste guide to get the API running on **any machine** —
Windows, macOS, or Linux — for local development, and how to deploy it to a
server. No prior knowledge of the project is assumed.

If you only want the 30-second version, jump to [Quick start](#quick-start).

---

## Contents

1. [What you are running](#what-you-are-running)
2. [Prerequisites](#prerequisites)
3. [Quick start](#quick-start)
4. [Local setup (step by step)](#local-setup-step-by-step)
   - [Windows (PowerShell)](#windows-powershell)
   - [macOS / Linux (bash/zsh)](#macos--linux-bashzsh)
5. [Verify it works](#verify-it-works)
6. [Using the API](#using-the-api)
7. [Command-line batch tools](#command-line-batch-tools-no-server)
8. [Run with Docker](#run-with-docker)
9. [Deploy to a server](#deploy-to-a-server)
10. [Configuration & tuning](#configuration--tuning)
11. [Troubleshooting](#troubleshooting)

---

## What you are running

A small **FastAPI** web service (`api.py`) that detects dark-blue dots in
microscopy images and flags the "positive" large blobs. It is **stateless** —
image in, JSON or labeled image out. There is no database and nothing to
provision beyond Python (or Docker).

Endpoints: `GET /health`, `POST /analyze` (JSON), `POST /analyze/image` (labeled
JPEG). Full API reference is in the [README](../README.md).

---

## Prerequisites

| You need | Version | Check with | Get it |
|---|---|---|---|
| **Python** | 3.10+ (3.12 recommended) | `python --version` | <https://python.org/downloads> |
| **Git** | any | `git --version` | <https://git-scm.com> |
| **Docker** *(optional)* | any recent | `docker --version` | <https://docker.com> — only if using the container path |

> On Windows, tick **"Add Python to PATH"** in the installer. If `python`
> doesn't work, try `py` (the Python launcher) in its place everywhere below.

No GPU, no database, no cloud account is required to run locally.

---

## Quick start

```bash
git clone https://github.com/zhubek/allergo-probe-analyzer
cd allergo-probe-analyzer
```

Then pick your OS:

**Windows (PowerShell)**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn api:app --host 0.0.0.0 --port 8000
```

**macOS / Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn api:app --host 0.0.0.0 --port 8000
```

Open <http://localhost:8000/docs> in a browser. Done.

The rest of this document explains each step, the cross-platform gotchas, and
how to deploy.

---

## Local setup (step by step)

The only real cross-platform difference is **where the virtual-environment
executables live**: `\.venv\Scripts\` on Windows vs `.venv/bin/` on macOS/Linux.
Activating the venv hides that difference, so the examples below activate it.

### Windows (PowerShell)

```powershell
# 1. Clone and enter the project
git clone https://github.com/zhubek/allergo-probe-analyzer
cd allergo-probe-analyzer

# 2. Create an isolated virtual environment
python -m venv .venv

# 3. Activate it  (note: NOT .venv/bin — Windows uses .venv\Scripts)
.venv\Scripts\Activate.ps1

# 4. Install pinned dependencies
pip install -r requirements.txt

# 5. Start the server
uvicorn api:app --host 0.0.0.0 --port 8000
```

> **"running scripts is disabled on this system"** when activating? Run this
> once, then retry step 3:
> ```powershell
> Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
> ```
> Alternatively, skip activation and call the venv's tools by full path:
> `.venv\Scripts\python.exe -m uvicorn api:app --host 0.0.0.0 --port 8000`

### macOS / Linux (bash/zsh)

```bash
# 1. Clone and enter the project
git clone https://github.com/zhubek/allergo-probe-analyzer
cd allergo-probe-analyzer

# 2. Create an isolated virtual environment
python3 -m venv .venv

# 3. Activate it
source .venv/bin/activate

# 4. Install pinned dependencies
pip install -r requirements.txt

# 5. Start the server
uvicorn api:app --host 0.0.0.0 --port 8000
```

> **Why a venv?** On many Linux distros and Homebrew Python, the system pip is
> [PEP-668](https://peps.python.org/pep-0668/) "externally managed" and will
> refuse to install. The virtual environment sidesteps this and keeps the
> project's dependencies isolated. Always install into `.venv`, never globally.

To leave the virtual environment later: `deactivate`.

---

## Verify it works

With the server running, in a **second terminal**:

**Any OS (browser):** open <http://localhost:8000/docs> — interactive Swagger UI.
Click any endpoint → **Try it out** → upload `samples/sample_dense.jpg` →
**Execute**.

**macOS / Linux (curl):**
```bash
curl http://localhost:8000/health
# -> {"status":"ok"}

curl -F "file=@samples/sample_dense.jpg" http://localhost:8000/analyze
# -> {"width":1200,"height":804,"count":1,"has_points":true,"points":[...]}
```

**Windows (PowerShell):** use `curl.exe` — bare `curl` is an alias for
`Invoke-WebRequest` and won't accept these flags.
```powershell
curl.exe http://localhost:8000/health
curl.exe -F "file=@samples/sample_dense.jpg" http://localhost:8000/analyze
```

> **Expected: low counts on the samples.** The two `samples/` images are
> downscaled to 1200px for quick wiring tests, but the detector thresholds are
> calibrated for full-resolution (~5440px) microscopy images. So
> `sample_sparse.jpg` returns `count: 0` and `sample_dense.jpg` returns
> `count: 1`. **This is expected, not a bug.** Feed a full-resolution image to
> see realistic counts.

---

## Using the API

All examples assume the server is at `http://localhost:8000`. Both POST
endpoints accept the image four different ways.

| Mode | curl (macOS/Linux & `curl.exe` on Windows) |
|---|---|
| **File upload** | `curl -F "file=@image.png" http://localhost:8000/analyze` |
| **Raw bytes** | `curl --data-binary @image.png -H "Content-Type: image/png" http://localhost:8000/analyze` |
| **Image URL** | `curl -H "Content-Type: application/json" -d '{"url":"https://.../img.png"}' http://localhost:8000/analyze` |
| **Base64** | JSON body `{"image_b64":"<base64>"}` (use a script/client, not the shell — base64 strings overflow command-line length limits) |

**Get a labeled image back** (yellow circles = dots, red boxes = positives):
```bash
curl -F "file=@samples/sample_dense.jpg" http://localhost:8000/analyze/image -o labeled.jpg
# dot/positive counts are also returned in the X-Dot-Count / X-Positive-Count headers
```

A tiny Python client (works on every OS, handles base64 cleanly):
```python
import requests

# JSON analysis
r = requests.post("http://localhost:8000/analyze",
                  files={"file": open("samples/sample_dense.jpg", "rb")})
print(r.json())

# labeled image
r = requests.post("http://localhost:8000/analyze/image",
                  files={"file": open("samples/sample_dense.jpg", "rb")})
open("labeled.jpg", "wb").write(r.content)
print("dots:", r.headers["X-Dot-Count"], "positives:", r.headers["X-Positive-Count"])
```

---

## Command-line batch tools (no server)

To process a whole folder of images without running the API:

```bash
# (venv activated)  every dot -> per-image *_dots.csv + *_overlay.jpg
python detect_dots.py --dir path/to/images
python detect_dots.py one_image.png            # single file

# positive large blobs -> *_clusters.csv + *_clusters.jpg + clusters_summary.csv
python detect_clusters.py --dir path/to/images
```

Outputs land next to each source image. `--dir` globs `*.png`, skips the
generated overlays, so re-running is idempotent.

---

## Run with Docker

The repo ships a production-ready `Dockerfile` (Python 3.12-slim, non-root user,
built-in `/health` healthcheck). This is the most portable way to run on any
machine that has Docker — no Python setup needed.

```bash
# build the image
docker build -t allergo-probe-analyzer .

# run it, mapping container port 8000 to host port 8000
docker run --rm -p 8000:8000 allergo-probe-analyzer
```

Then verify exactly as above (<http://localhost:8000/docs>).

Useful variations:
```bash
# run detached in the background, auto-restart on crash/reboot
docker run -d --restart unless-stopped -p 8000:8000 --name allergo allergo-probe-analyzer

# view logs / stop
docker logs -f allergo
docker stop allergo && docker rm allergo
```

`docker-compose.yml` (optional convenience — create it if you like):
```yaml
services:
  api:
    build: .
    ports:
      - "8000:8000"
    restart: unless-stopped
```
Then: `docker compose up -d --build`.

---

## Deploy to a server

The service is stateless, so deployment is just "run the container (or the
uvicorn process) somewhere reachable, behind HTTPS." Pick whichever matches
your infrastructure.

### Option A — Any Linux VPS with Docker (simplest)

```bash
# on the server
git clone https://github.com/zhubek/allergo-probe-analyzer
cd allergo-probe-analyzer
docker build -t allergo-probe-analyzer .
docker run -d --restart unless-stopped -p 8000:8000 --name allergo allergo-probe-analyzer
```

Put a reverse proxy in front for TLS and a clean hostname. Minimal **nginx**:
```nginx
server {
    listen 80;
    server_name allergo.example.com;

    client_max_body_size 64M;        # images can be up to 60 MB (see MAX_BYTES)

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_read_timeout 120s;     # full-res analysis can take tens of seconds
    }
}
```
Then add HTTPS with [certbot](https://certbot.eff.org/) (`certbot --nginx`).

### Option B — systemd service (no Docker)

For a bare-metal/VM host without Docker. Set up the venv as in
[Local setup](#macos--linux-bashzsh), then create
`/etc/systemd/system/allergo.service`:
```ini
[Unit]
Description=Allergo Probe Analyzer API
After=network.target

[Service]
User=www-data
WorkingDirectory=/opt/allergo-probe-analyzer
ExecStart=/opt/allergo-probe-analyzer/.venv/bin/uvicorn api:app --host 0.0.0.0 --port 8000
Restart=always

[Install]
WantedBy=multi-user.target
```
```bash
sudo systemctl daemon-reload
sudo systemctl enable --now allergo
sudo systemctl status allergo
```
Front it with nginx as in Option A.

### Option C — Managed container platforms

The image runs unmodified on any container host. It listens on `0.0.0.0:8000`
and exposes `/health` for liveness/readiness probes.

- **Google Cloud Run** — `gcloud run deploy --source .` (Cloud Run sets `$PORT`;
  if you target it, change the `CMD`/`ExecStart` to `--port ${PORT:-8000}`).
- **AWS App Runner / ECS Fargate**, **Azure Container Apps**, **Fly.io**,
  **Render**, **Railway** — point them at this repo or the built image, expose
  port 8000, set the health check path to `/health`.

### Production notes

- **Scaling / concurrency.** Analysis is CPU-bound (σ=60 Gaussian + watershed).
  For real load, run multiple workers and size CPU accordingly:
  `uvicorn api:app --host 0.0.0.0 --port 8000 --workers 4`, or run several
  container replicas behind a load balancer. Each worker handles one image at a
  time.
- **Request size & timeouts.** Images may be up to **60 MB** (`MAX_BYTES` in
  `api.py`). Raise proxy body-size limits accordingly and allow generous read
  timeouts — a full-resolution image can take tens of seconds.
- **No state to back up.** Nothing is persisted; restarts are safe and
  instances are interchangeable.
- **Security.** There is no authentication built in. If exposing publicly, add
  auth at the proxy/gateway, and note the `url` input mode makes the server
  fetch arbitrary URLs (SSRF surface) — restrict it at the network layer if
  that matters for your environment.

---

## Configuration & tuning

There are **no environment variables** — behavior is set by constants in code.

| What | Where | Notes |
|---|---|---|
| Host / port | command line | `uvicorn api:app --host ... --port ...` |
| Max upload size | `MAX_BYTES` in `api.py` | default 60 MB |
| Download timeout (URL mode) | `DOWNLOAD_TIMEOUT` in `api.py` | default 15 s |
| Returned-image size | `OVERLAY_MAXDIM` in `api.py` | default 1800 px longest side |
| **Detection thresholds** | `allergo_core.py` | `MIN_BLOB_AREA`, `DARK_DROP`, `BLUE_EXCESS`, … — see [ALGORITHM.md](ALGORITHM.md) |

The detection constants are calibrated for full-resolution (~5440px) images.
If you analyze resized images, scale the `*_AREA` thresholds and the
`σ`/distance values roughly with resolution, or counts will run low.

---

## Troubleshooting

| Symptom | Cause & fix |
|---|---|
| `'python' is not recognized` (Windows) | Python not on PATH. Use `py` instead, or reinstall ticking "Add Python to PATH". |
| `running scripts is disabled on this system` when activating venv | PowerShell execution policy. Run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, or call tools by full path: `.venv\Scripts\python.exe -m uvicorn ...`. |
| `error: externally-managed-environment` on pip install | You're using the system Python (PEP-668). Create and activate the `.venv` first, then `pip install`. |
| `curl` flags rejected on Windows | PowerShell's `curl` is `Invoke-WebRequest`. Use **`curl.exe`** (it ships with Windows 10+). |
| `[Errno 48] / port already in use` | Something is on 8000. Pick another port: `--port 8001`. Find the culprit: `lsof -i :8000` (mac/Linux) / `netstat -ano \| findstr :8000` (Windows). |
| `count: 0` / very low counts | You fed a downscaled image (the `samples/` are 1200px). Expected — thresholds target full-res. Use a full-resolution image. |
| `Argument list too long` when sending base64 via shell | Shell command-length limit, not the app. Send base64 from a script/client (see the Python example), or use file-upload mode. |
| Server starts but `/docs` won't load remotely | You bound to `127.0.0.1`. Use `--host 0.0.0.0` to accept external connections (and open the firewall / security group for the port). |
| `ModuleNotFoundError: fastapi` (or numpy, scipy…) | venv not activated, or deps not installed. Activate it and re-run `pip install -r requirements.txt`. |
| Pillow / libjpeg errors at runtime in a container | Use the provided `Dockerfile` — it installs `libjpeg62-turbo` and `zlib1g`. |

Still stuck? Confirm versions with `python --version` (need 3.10+) and
`pip list` (should show the packages from `requirements.txt`), and check the
server log output in the terminal where uvicorn is running.
