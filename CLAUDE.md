# CLAUDE.md

Project constitution for coding agents working in this repository. It is loaded every session. Read all of it.

## 0. How to use this file

- These rules outrank a work order. If a work order conflicts with them, stop and report the conflict. Do not pick a side.
- When rules pull against each other, the order is: (1) the invariants in section 5, (2) the work order's scope and non-goals, (3) the conventions, (4) speed of delivery.
- Cite rules by number in reports, for example "would violate Invariant 4".
- If an `AGENTS.md` exists, it must say the same thing as this file. Report any drift.
- This file is law, not a chat log. Never paste task history into it. When the human has to repeat a correction, propose a new rule in your PR. Do not edit invariants silently.

## 1. Discovery Summary

- **Domain problem.** Show a live object-detection demo on a machine with no GPU. A browser page captures the camera, an image or a video file. A service on the same machine (WSL2 Ubuntu or Linux) detects objects with a pretrained COCO model. The page draws the boxes over the picture. There is no training.
- **Users.** A presenter who is not necessarily a programmer and who starts one command and opens a browser. An audience member who may open the page from another device on the same network. Other programs that call the HTTP endpoint directly.
- **Product shape.** One small HTTP service plus one self-contained HTML5 page. `POST /detect` takes a raw JPEG or PNG and returns JSON.
- **Release-one scope.** Camera, image and video-file input; adjustable confidence; boxes over the media; a readout of FPS, server milliseconds and detected classes; optional local HTTPS.
- **Priority.** The first use is a demo for a manager. A verified, working demo beats new features.
- **Status (verified 2026-09-21).** Run on this CPU with yolo26s.pt. scripts/smoke_http.py: 10 passed, 0 failed, 0 skipped. Not yet verified: browser flows (camera, image, video, slider) and performance numbers.

**Stack and why**

| Component | Choice | Why |
|---|---|---|
| Detector | Ultralytics YOLO26, COCO-pretrained; default `yolo26s.pt`, `yolo26m.pt` for more accuracy | Newest Ultralytics family, designed for CPU and edge use with an end-to-end (NMS-free) output. Pretrained weights cover 80 classes including person, so no training is needed. |
| Runtime | PyTorch, CPU build only | No multi-gigabyte CUDA download and no GPU code path. |
| Service | FastAPI + uvicorn | Small async HTTP server with automatic query validation. |
| Image handling | OpenCV + NumPy | Installed with Ultralytics; used to decode uploads. |
| Client | Vanilla JS and CSS in one HTML file | Nothing to build; works offline. |

**Rejected alternatives**

- GPU/CUDA: the target machine has no GPU.
- Training or fine-tuning: not needed; the pretrained model is the point.
- Older YOLO versions: the human asked for the newest, and they are slower on CPU.
- Multipart upload: needs an extra dependency and gives no benefit here.
- WebSocket or WebRTC streaming: more moving parts; plain HTTP is enough at a few FPS.
- Databases, authentication: out of demo scope.
- Frontend frameworks and bundlers: a build step and an offline risk.

## 2. Mission

A demo that runs on CPU only and starts with one command. Three promises must never be broken: no GPU is needed; the boxes on screen are what the model actually detected; every number shown (FPS, milliseconds) is a real measurement.

## 3. Repository layout

```text
server.py                   the service (the only Python module for now)
index.html                  the whole client: HTML, CSS, JS
CLAUDE.md                   this file
work-orders/                numbered work orders from the strategic layer
scripts/                    helper scripts, for example bench.py (created by a work order)
tests/                      pytest tests (created by a work order)
README.md                   user-facing documentation (created by a work order)
cert.pem, key.pem           optional local HTTPS certificate; never committed
```

Do not create other top-level folders without a work order.

## 4. Architecture

```text
browser (index.html) --POST /detect, JPEG--> server.py (FastAPI) --> Ultralytics YOLO26, CPU
                     <-- JSON, boxes normalised 0..1 --
```

### 4.1 Server (`server.py`)

- Hides all GPUs by setting `CUDA_VISIBLE_DEVICES=""` before torch is imported.
- At import: reads configuration, sets torch threads, loads the model with `task="detect"`, and runs one warm-up prediction on a blank image.
- `infer(img, conf)` takes the inference lock, calls `model.predict(img, imgsz=IMGSZ, conf=conf, verbose=False)`, and converts the boxes to normalised coordinates.
- `GET /` serves `index.html`. `GET /info` returns the model file name, `imgsz`, device and thread count. `POST /detect` decodes the body with OpenCV and runs `infer` in a worker thread so the event loop is never blocked.
- `python server.py` serves HTTPS if `cert.pem` and `key.pem` sit next to it, otherwise HTTP. It binds `0.0.0.0` on `PORT`.

### 4.2 Client (`index.html`)

- Three input modes: `camera`, `video` (file, looped, muted) and `image` (file, one shot).
- A sequential loop: grab a frame, scale it so the longest side is at most `SEND_MAX` (640 px), encode it as JPEG (quality 0.7), `POST` it to `/detect?conf=<slider minimum>`, keep the raw answer, draw only the detections at or above the slider value, then grab the next frame. The slider filters on the page, so moving it updates the boxes at once without a new request. The next frame is never grabbed before the previous answer arrives. A `runId` token cancels a running loop.
- The overlay canvas has the same size as the stage (times the device pixel ratio). Boxes are drawn from normalised coordinates, so any display size works. The live video stays smooth; the boxes are the latest detection.
- The readout shows FPS (smoothed), the server's milliseconds per frame, and a summary such as `person ×2, cup`.

### 4.3 Configuration (environment variables only)

| Variable | Default | Meaning | Notes |
|---|---|---|---|
| `YOLO_MODEL` | `yolo26s.pt` | Weights file, or an exported model folder | A `.pt` file is downloaded on first use. |
| `YOLO_IMGSZ` | `640` | Inference size in pixels | Must equal the export size for ONNX and OpenVINO models. Smaller is faster and worse on small objects. |
| `YOLO_CONF` | `0.35` | Default confidence when a request has no `conf` | The page slider sends its own value. |
| `YOLO_THREADS` | logical core count | PyTorch CPU threads | Affects PyTorch models only. |
| `PORT` | `8000` | Listening port | |

### 4.4 API contract

The contract is public. Changing it needs a work order and a README update.

`POST /detect?conf=0.35`

- Request body: raw JPEG or PNG bytes. `Content-Type: image/jpeg` is what the page sends. The query `conf` is a float clamped to 0.01-0.99.
- Response `200` (illustrative values):

```json
{"width": 640, "height": 480, "inference_ms": 92.4,
 "detections": [{"label": "person", "conf": 0.912, "box": [0.11, 0.21, 0.43, 0.98]}]}
```

- `width` and `height` are those of the received image, which the page has already downscaled.
- `box` is `[x1, y1, x2, y2]` normalised to 0..1 relative to that image.
- `label` is a COCO class name. `conf` has 3 decimals.
- `inference_ms` is the wall time of `infer()`. It includes waiting for the lock and excludes JPEG decoding and network time, so end-to-end time is larger.
- Errors: `400` for an undecodable body; `422` for a non-numeric `conf` (FastAPI validation).

```bash
curl -s -X POST --data-binary @photo.jpg -H "Content-Type: image/jpeg" "http://localhost:8000/detect?conf=0.35"
```

### 4.5 Networking and HTTPS

- The camera works only on `localhost` or over HTTPS. Opening the page from another device over plain HTTP silently blocks the camera.
- Another device reaches WSL2 either through mirrored networking (`networkingMode=mirrored` under `[wsl2]` in `C:\Users\<name>\.wslconfig`, then `wsl --shutdown`) or through `netsh interface portproxy` to the WSL address (from `hostname -I`, which changes after a restart). Allow inbound TCP on the port in the Windows firewall; Microsoft documents extra firewall considerations for mirrored mode.
- Self-signed certificate for a demo, created in the project folder (the server then switches to HTTPS on its own):

```bash
openssl req -x509 -newkey rsa:2048 -nodes -keyout key.pem -out cert.pem -days 30 \
  -subj "/CN=localhost" -addext "subjectAltName=DNS:localhost,IP:127.0.0.1"
```

## 5. Non-negotiable invariants

1. **CPU only.** No GPU code paths and no CUDA wheels. `server.py` sets `CUDA_VISIBLE_DEVICES=""` before importing torch, and PyTorch must be the CPU build (`torch.version.cuda is None`). Keep `/info` truthful about the device.
2. **Model family.** Ultralytics YOLO26 or newer. Never fall back silently to another family or version. If the weights cannot be downloaded or loaded, stop and report.
3. **Pretrained only.** No training, fine-tuning or dataset downloads.
4. **One model, one inference at a time.** One model instance, the inference lock stays, one uvicorn worker. Blocking model calls never run on the event loop.
5. **No request pile-up.** The page sends the next frame only after the previous response arrives.
6. **Privacy.** Uploaded images are never written to disk, logged or sent elsewhere. No telemetry. The only runtime network access is the one-time weights download.
7. **Works offline.** The page is a single file with no CDN scripts, web fonts or remote images.
8. **Boxes stay aligned.** Boxes are normalised and drawn on an overlay exactly the size of the displayed media. Do not add `object-fit` letterboxing under the overlay. Stop must release the camera.
9. **Measured numbers only.** Never state an FPS or latency you did not measure. Give the CPU model, thread count, model name and `imgsz` with every figure.
10. **Demo security posture.** No authentication, so LAN use only. Do not add public tunnels, port forwarding instructions for the internet, or "internet-ready" wording without a work order.
11. **Colour order.** Images decoded by OpenCV are BGR, and Ultralytics expects BGR for NumPy input. Do not convert to RGB.

## 6. Design decisions

| Decision | Why | Revisit when |
|---|---|---|
| Raw image bytes as the request body | Works with `curl`; needs no multipart dependency | Another client needs multipart |
| Boxes normalised to 0..1 | The client can draw at any display size, independent of downscaling | Not without a contract change |
| Client downscales to at most 640 px, JPEG 0.7 | Small uploads, fast decode, matches the default `imgsz` | A larger default `imgsz` is chosen |
| Sequential request loop | Bounded latency, no queue, natural frame dropping | A streaming work order exists |
| Inference lock and one worker | The Ultralytics predictor is not safe to share across threads; one model in memory | Multi-user load becomes a requirement |
| `CUDA_VISIBLE_DEVICES` set before the torch import | Guarantees CPU even if a CUDA build sneaks in | Never |
| Overlay canvas over live video | Smooth video at a low detection rate | Frame-exact sync becomes a requirement |
| Model loaded at import, with warm-up | The first demo frame is not slow, and startup fails fast | Tests need lazy loading (work order) |
| Single HTML file, no CDN | Works offline and on locked-down networks | A build step becomes necessary |
| Slider filters on the page | One request per frame at the slider's minimum; the page hides detections below the slider value, so moving the slider updates the boxes instantly in every mode, including a still image, and costs the server nothing | The server needs to return different results per threshold |

## 7. Forbidden actions

- Installing or referencing CUDA or `cu*` wheels, `nvidia-*` packages, TensorRT, or `device=0`.
- Committing weights (`*.pt`, `*.onnx`, `*_openvino_model/`), `cert.pem`, `key.pem`, `.venv/`, caches, or any captured frame, screenshot, photo or video of people.
- Adding a dependency without justifying it in the report, or adding frontend frameworks or bundlers.
- Changing the API contract, default model, default `imgsz`, default port or environment-variable names outside a work order.
- Enabling CORS, adding authentication, or opening extra endpoints outside a work order.
- Making tests pass by weakening them: deleting tests, using a broad `except`, or mocking away the very call a test claims to verify.
- Touching anything outside this repository, or installing packages globally (use the project venv).
- Running shell commands or choosing file paths from request data.
- Pushing to `main`, merging your own PR, or force-pushing a shared branch.
- Printing or committing secrets. If a secret or credential shows up in output or in the VM, stop and report.

## 8. Decide or escalate

**Decide alone:** names of local variables and functions inside the scope; how tests are structured; wording of messages inside the scope; small fixes on lines you already touch; choosing between equivalent standard-library options.

**Stop and report, with a recommendation:**

- Any change to the API contract, dependencies, defaults or environment variables.
- Any need to touch an invariant.
- A single defect that needs more than about 40 changed lines.
- Weights that will not download, or a model name that is not recognised.
- An ambiguous instruction where the two readings differ materially. If they do not differ materially, take the narrower reading and say which one you took.
- A test that cannot be written without refactoring. Propose the refactor as a follow-up; do not do it.

## 9. Workflow

- **Runtime boundary.** You run in a disposable VM or WSL2 distro that holds no production secrets and nothing irreplaceable. Inside it you may install what you need (venv packages, `apt` packages) and must document it.
- **Anti-pilot rule.** Do not ask the human to install packages, run commands or paste logs. Do it yourself and report. Ask only for decisions that belong to the human: scope, risk, release.
- **Start.** Read this file and the work order in full. Check live state with `git status` and `git log -5`. If it differs from the work order, report the difference. If the directory is not a git repository, run `git init`, add a `.gitignore` covering the items in section 7, commit the baseline, and report that you did.
- **Plan first.** Your first message states, in at most 10 lines, the files you will touch and any conflict with this file.
- **Unit of work.** One task, one branch, one PR. Branch from up-to-date `main`. Name branches `<type>/<short-kebab-name>` with type `feat`, `fix`, `chore`, `docs` or `test`. Commit only related files, in small commits with an imperative subject of at most 72 characters and a body that says why. Never rewrite or force-push published history.
- **PR.** Push the branch, open the PR, use the Agent Report as its description, do not merge. With no remote, commit on the branch and report "no remote". Never invent a PR URL. A PR much larger than about 400 changed lines should be split; stop and say so.
- **Scope.** Do only what the work order says. No drive-by refactors and no formatting-only diffs. List anything else you noticed under follow-ups.
- **Blockers.** Stop and report real blockers instead of working around an invariant.

## 10. Local setup and commands

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu   # CPU build first
pip install -U ultralytics fastapi "uvicorn[standard]" pytest httpx ruff
python -c "import torch; assert torch.version.cuda is None and not torch.cuda.is_available()"
python server.py                                    # http://localhost:8000
YOLO_MODEL=yolo26m.pt python server.py              # a different model
YOLO_IMGSZ=480 python server.py                     # a smaller inference size
```

- Install the CPU torch first; otherwise `ultralytics` may pull a CUDA build.
- Use the system Python 3 of the VM. If `pip install ultralytics` reports an unsupported Python, report it. Do not install another interpreter.
- If OpenCV fails to import because of a missing library, install `libgl1` and `libglib2.0-0` with `apt`.
- Exported models (build once, never commit): `yolo export model=yolo26s.pt format=openvino imgsz=640` creates `yolo26s_openvino_model/`; `format=onnx` creates `yolo26s.onnx`. Run them with `YOLO_MODEL=<path>` and the same `YOLO_IMGSZ`. Ultralytics may install exporter or runtime packages on first use; record what it installed.

## 11. Performance

- **Levers, in order of effect:** model size (n, then s, then m); `YOLO_IMGSZ` (640, 480, 416, 320); export to OpenVINO on Intel CPUs or ONNX; `YOLO_THREADS` (try the physical core count against the logical one); client `SEND_MAX` and JPEG quality, which mostly affect transfer and decode.
- **Client downscaling.** The server resizes every image to `imgsz`, so an upload smaller than `imgsz` is enlarged and costs the same. Lowering `SEND_MAX` below `imgsz` therefore loses accuracy without saving inference time. Change `YOLO_IMGSZ` instead, and measure before relying on this.
- **Measure properly.** Warm up first, then time at least 30 runs. Report mean and p95. Do not benchmark while a client is running detections. One number without its CPU model, thread count, model name and `imgsz` is not a result.
- **Two clients share one CPU.** With several browser tabs the FPS divides between them.
- **WSL2 limits.** If `nproc` shows fewer cores than the machine has, check `processors` under `[wsl2]` in `.wslconfig`.
- **Targets.** A few FPS is the goal. Prefer the largest model that keeps the mean at or below about 333 ms per frame (3 FPS) on the demo machine, as measured. The human decides; you report.

## 12. Coding conventions

**Python**

- Keep `server.py` one small module until a work order splits it.
- Import-time side effects are limited to what section 4.1 lists.
- Add type hints to functions that other code calls.
- Never use a bare `except`. Map decode failures to `400`; let other errors surface as `500`. Never swallow them.
- Use `asyncio.to_thread` for blocking work in async endpoints. Global state is limited to `model`, `lock` and configuration constants.
- Comments explain why, not what. Aim for lines of about 100 characters. `ruff check .` must be clean on lines you touch. Do not reformat untouched code.

**Frontend**

- Vanilla JavaScript in the one existing `<script>`. No frameworks, no modules, no external files.
- Element IDs are load-bearing: `video`, `still`, `overlay`, `stage`, `placeholder`, `status`, `startCam`, `openFile`, `file`, `stop`, `conf`, `confOut`, `fps`, `ms`, `found`, `model`. Renaming one means updating the JS in the same change.
- The `[hidden] { display: none !important; }` rule must stay, because other rules would otherwise override the `hidden` attribute.
- Colours, sizes and fonts come from CSS custom properties in `:root` and a system font stack.
- Any new input source must call `stopAll()` first and end by calling `begin()`. Stopping must always release the camera.
- UI copy is sentence case with plain verbs and consistent action names (Start camera, Open image or video, Stop). An error says what went wrong and what to do about it. No apologies, no all-caps labels, no emoji.
- Accessibility: real `<button>` elements, visible keyboard focus, touch targets of at least 44 px, `role="status"` on the status line, and a layout that works down to a phone width.

## 13. Testing

Commands: `ruff check .` and `python -m pytest -q`. Tests live in `tests/`.

| Layer | What it proves | How | Needs weights | Report as |
|---|---|---|---|---|
| Contract | API shape and status codes | pytest, `TestClient`, stubbed model | No | passed or failed |
| Model smoke | The real model finds a person | `pytest -m model` on a sample photo | Yes | skipped if it cannot run |
| Benchmark | Milliseconds per frame per model | `scripts/bench.py` | Yes | measurements |
| Browser | The UI flows work | Manual checklist below | Yes | not run unless done |
| Static | Lint | `ruff check .` | No | passed or failed |

**Contract tests to write** (names are suggestions): `test_info_reports_cpu`, `test_detect_returns_documented_shape`, `test_boxes_are_normalised`, `test_conf_is_clamped_low_and_high`, `test_non_numeric_conf_is_422`, `test_garbage_body_is_400`, `test_empty_body_is_400`, `test_index_is_served`, `test_detect_writes_no_files`.

**Stubbing.** `server.py` loads the model at import. Patch `ultralytics.YOLO` with a fake before importing `server`. The fake's `predict` must accept a NumPy image plus keyword arguments and return a list whose first item has `.names` (a dict) and `.boxes` (an iterable). Each box needs `.xyxy`, `.cls` and `.conf` as NumPy arrays, because the server calls `box.xyxy[0].tolist()`, `int(box.cls[0])` and `float(box.conf[0])`.

**Rules for tests**

- Every test must be able to fail. For each new test, say in the PR which code change would make it fail. Asserting a stub's own output is not evidence.
- Tests stay offline and never need weights, except the ones marked `model`. Register the `model` marker in the pytest configuration.
- A skipped test is not a passing test. A test you did not run is not evidence.

**Browser checklist (manual):** start and stop the camera (camera light off after Stop); open an image; open a video file; move the confidence slider and watch the box count change; stop the server and check that the page shows a readable error and recovers when the server returns. If you cannot drive a browser, report the checklist as not run and hand it to the human.

## 14. Security and privacy

- **Threat model.** A demo on a trusted local network. No authentication, no rate limiting, no TLS by default. It must not be exposed to the internet.
- **Data.** Camera frames go only from the browser to this server, over the same origin. Nothing is stored, logged or forwarded. Access logs may show method, path and status; never log request bodies.
- **Input.** The request body is untrusted. Decode it only with OpenCV. Never build paths or commands from it. The model path comes only from the environment, never from a request.
- **Cross-origin.** Same-origin only. Do not enable CORS without a work order.
- **Secrets.** `key.pem` is a secret even for a demo. Never commit or print it. Never paste real credentials into prompts, tests, docs or logs. If one appears, stop and report.
- **Dependencies.** Add none without a reason. Do not download or execute anything outside pip, apt and the Ultralytics weights.

## 15. Documentation

- `README.md` is the user-facing document. It covers setup, running, the environment variables, the endpoint contract, WSL2 networking and HTTPS notes, and known limitations. Update it in the same PR as any behavior change.
- State limits plainly: demo, no authentication, CPU only, FPS depends on the CPU. Never write "production ready" or "real-time" without a measured number next to it.
- Code comments explain why. Do not restate this file in comments.
- Keep this file current. Update the status line and the known-gaps list as facts change.

## 16. Definition of done

- [ ] The change matches the work order's scope and touches no unrelated files.
- [ ] Every acceptance criterion has a named check and a result.
- [ ] Tests and lint were run and reported with exact results; skipped or not-run items are labelled as such.
- [ ] The CPU-only check ran (`torch.version.cuda is None`).
- [ ] No weights, certificates, keys or photos are committed.
- [ ] The README and this file are updated where behavior or facts changed.
- [ ] The report says what you are least sure about.

## 17. Reporting

End every task with this report. It is an index into evidence, not proof. Write it for a reader who knows what the demo should do but may not know Python: lead with what changed in behavior, then give the evidence. Aim for under about 60 lines, and quote command output only where it is evidence (exit codes, counts, timings).

```markdown
## Agent Report
Branch / Commit / PR URL (or "no remote")

## Summary
## Files changed
## Tests run
- command: result (one line each)

## Measurements (performance work only)
CPU model, threads, model, imgsz, mean ms, p95 ms, FPS

## Documentation impact
## Local tools installed
## Safety confirmations
- CPU-only verified (`torch.version.cuda` is None).
- No weights, certificates, keys or photos committed.
- No unrelated files changed.
- No skipped test reported as passed.

## Least sure about
## Known limitations / follow-up
```

Use only these outcomes: passed, failed, skipped, not run, blocked, out of scope.

- Good: "Contract tests: 9 passed. Model smoke test: skipped, no network for weights. Browser checklist: not run."
- Bad: "All tests passed."
- Do not write "all tests passed" unless the full suite ran and passed. Never claim CI results you cannot see. Never fake a PR URL.

## 18. Known gaps (remove as they are fixed or disproved)

- `/detect` reads the whole request body with no size limit.
- `inference_ms` includes time spent waiting for the lock.
- `/info` reports the device as a constant string rather than a measured value.
- The first start needs internet to download weights.
- No tests or benchmark exist yet.
- Exported ONNX and OpenVINO models have a fixed input size: keep `YOLO_IMGSZ` equal to the export size.
- The overlay shows the latest detections over live video, so boxes lag by one inference. This is expected.

## 19. Planned work (not authorized; do not start without a work order)

- Work order 001: verify the baseline, `.gitignore`, `scripts/bench.py`, measurements, and the empty-body fix if the suspicion is confirmed.
- Work order 002: contract tests, README, a request size limit on `/detect` (rejecting oversized bodies).
- Later, only if asked: pinned requirements, an optional WebSocket route, a default OpenVINO or ONNX export, a Docker image.

## 20. Glossary

- **COCO:** the 80-class dataset the pretrained weights were trained on (person, car, dog, bottle and so on).
- **NMS-free:** the model outputs final detections directly, without the non-maximum-suppression post-processing step.
- **`imgsz`:** the square size the model runs at, in pixels.
- **`conf`:** the minimum confidence for a detection to be returned.
- **Normalised box:** coordinates from 0 to 1 relative to the image width and height.
- **Constitution:** this file. **Work order:** one bounded task from the strategic layer. **Executor:** the coding agent that carries it out.
