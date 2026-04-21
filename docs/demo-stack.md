# Demo Stack

## Purpose

The current live stack is a demo surface, not the training lab itself.

It exists to prove that we can run:

- real telephony over a public endpoint
- speech-to-text and text-to-speech
- OCR and grounded visual detection
- local LLM, VLM, and embedding inference
- public ingress, TLS termination, and tunnel-based routing

In practical terms, this is the stack that lets us:

- receive real phone calls through a public voice endpoint such as `voice.example.com`
- run computer-use demos with OCR, detection, and vision reasoning
- keep a public-facing, demoable system online while we study the model-building side

The next phase is different. We are moving from "run interesting models" to "practice making, training, evaluating, packaging, and serving our own models."

## Current Inventory

### Sunny WSL2

| Port / Service | Role |
| --- | --- |
| `22` `ssh` | Remote shell access to WSL2 |
| `3478`, `5349` `coturn` | TURN / TURN-TLS for realtime traffic |
| `5432` `postgresql` | Local database for chat and app state |
| `8008` `synapse` | Matrix homeserver |
| `8080` `nginx` / Element | Web entry for chat stack |
| `8765` `voice-phone.service` | FastAPI phone server for Twilio webhook and media stream |
| `wg0` private lab address | WireGuard tunnel endpoint back to AWS |

### Sunny Windows

| Port | Service | Runtime | Purpose |
| --- | --- | --- | --- |
| `8013` | `gpt-oss-20b` | `llama-server.exe` | Local coding / chat model on Sunny |
| `8082` | `Qwen3VL-30B-A3B-Instruct` | `llama-server.exe` | Vision-language reasoning for demo flows |
| `8083` | `Qwen3-Embedding-4B` | `llama-server.exe` | Local embedding endpoint |
| `9001` | `stt_server.py` | Python 3.11 + CUDA | Speech-to-text |
| `9002` | `tts_server.py` | Python 3.11 | Text-to-speech |
| `9003` | `ocr_server.py` | Python 3.11 + CUDA | OCR with bounding boxes |
| `9004` | `detect_server.py` | Python 3.11 + CUDA | Grounded object detection |

### Mac Studio

| Port | Service | Runtime | Purpose |
| --- | --- | --- | --- |
| `8080` | `gpt-oss-120b` (or current default LLM) | `llama-server` | Heavy local coding / chat model (`/v1` OpenAI-compatible API) |

Live listener on this host was validated at `:8080` (not `:8013`; Sunny Windows uses `8013` for a separate local `llama-server` instance).

### AWS Edge

| Component | Purpose |
| --- | --- |
| Route53 | Public DNS for the lab domain |
| EC2 proxy | TLS termination, reverse proxy, WireGuard endpoint |
| Let's Encrypt | Public certs for chat and voice subdomains |

## Why This Matters

This stack is already useful, but it is not yet the main learning target.

The main learning target for this repo is:

- data pipelines
- training loops
- evaluation
- model packaging
- artifact promotion
- reproducible serving
- cost-aware deployment

So the live demo stack should be treated as the stable public surface and the practice repo should be treated as the place where we turn the operational shape into explicit, reproducible code.

## Where We Are Going

The intended split is:

- keep the demo stack alive and documented
- use this repo to practice training and shipping models
- move more of the boot logic and runtime assumptions into versioned code
- reduce dependence on undocumented host memory and manual restarts

That split now has an operational expression on Sunny:

- demo mode
  - WSL2 services stay up
  - Windows demo GPU services stay up
  - use `ops/sunny/reinit-demo-stack.sh`

- training mode
  - WSL2 services stay up
  - Windows demo GPU services on `8013`, `8082`, `8083`, `9001`, `9002`, `9003`, and `9004` are intentionally stopped
  - use `ops/sunny/training-mode.sh`

This transition is now runtime-validated on Sunny:

- demo mode sat around `23.7 GiB` to `23.9 GiB` of used VRAM on the 4090
- training mode reduced that to about `1.1 GiB` used and `23.5 GiB` free
- restore brought the full demo layer back online without disturbing the WSL2 service layer

That means the repo should become the source of truth for:

- what services are supposed to exist
- which ports they bind to
- which commands start them
- what counts as healthy
- what needs to come back after a reboot

It should not become the place for:

- committed secrets
- committed production credentials
- giant model weights
- host-specific one-off state that cannot be reconstructed

## Operational Rule

Going forward, the default should be:

- long-lived service definitions and boot commands live in the repo
- experiments happen on branches
- Sunny can check out a branch when we need to test shared in-progress work
- once something becomes part of the real operating surface, it should be codified here

Manual file copy is still acceptable for emergency testing, but it should be treated as temporary. If a process matters enough to survive reboot and matter to the system, it should have a repo-owned definition.

## Repo-Backed Entry Points

The repo now includes Sunny operations files under [`ops/sunny`](../ops/sunny/README.md):

- `audit-demo-stack.sh`: inspect the current WSL2 and Windows runtime state
- `reinit-demo-stack.sh`: start the WSL2 layer and then invoke the Windows-side start script
- `reinit-demo-stack.ps1`: start or verify the Windows-side demo processes
- `training-mode.sh`: free the 4090 for training without taking down the WSL2 service layer
- `training-mode.ps1`: stop or restore the Windows-side GPU demo processes

Those scripts do not replace model weights, secrets, or the existing source repos yet. They define the runtime contract around the current live stack so we can move toward reproducible state instead of hand-maintained state.
