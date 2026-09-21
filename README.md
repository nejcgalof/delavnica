# Delavnica – YOLO Object Detection Demo

A CPU-only object detection demo using Ultralytics YOLO26. Start one command, open a browser, and detect objects in live camera feeds, images, or videos on a machine with no GPU.

## Overview

Delavnica captures video from your camera (or opens an image/video file) and detects objects in real-time using a pretrained COCO model. The browser displays bounding boxes over the media with live FPS and latency measurements. Perfect for presentations and demonstrations on any laptop.

**Key features:**
- Runs on CPU only—no GPU or CUDA required
- Camera, image file, and video file input
- Adjustable confidence threshold
- Live FPS and per-frame latency display
- Detects 80 COCO classes (person, car, dog, cup, etc.)
- Optional local HTTPS for network access
- Single-file client—no build step, works offline

## Quick Start

### Requirements

- Python 3.9+
- macOS, Linux, or Windows with WSL2
- No GPU required

### Install and Run

```bash
# Clone the repository
git clone https://github.com/nejcgalof/delavnica
cd delavnica

# Set up a Python virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies (CPU build first to avoid CUDA wheels)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -U ultralytics fastapi "uvicorn[standard]"

# Verify CPU-only PyTorch
python -c "import torch; assert torch.version.cuda is None"

# Start the server
python server.py
```

Open your browser to **http://localhost:8000** and follow the on-screen prompts to start the camera, upload an image, or open a video file.

## Configuration

The server reads configuration from environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `YOLO_MODEL` | `yolo26s.pt` | YOLO weights file or exported model folder |
| `YOLO_IMGSZ` | `640` | Inference size in pixels (smaller = faster, less accurate) |
| `YOLO_CONF` | `0.35` | Default confidence threshold (0.01–0.99) |
| `YOLO_THREADS` | auto | PyTorch CPU threads (default: logical core count) |
| `PORT` | `8000` | Server listening port |

### Examples

```bash
# Use a more accurate but slower model
YOLO_MODEL=yolo26m.pt python server.py

# Faster inference on weaker CPUs
YOLO_IMGSZ=480 python server.py

# Run on a different port
PORT=9000 python server.py

# Combine multiple settings
YOLO_IMGSZ=480 YOLO_THREADS=4 python server.py
```

## API Contract

The `/detect` endpoint accepts raw JPEG or PNG image data and returns detection results.

### `POST /detect?conf=0.35`

**Request:**
- **Body:** Raw JPEG or PNG image bytes
- **Header:** `Content-Type: image/jpeg` (or `image/png`)
- **Query:** `conf` (float, clamped to 0.01–0.99)

**Response (200 OK):**
```json
{
  "width": 640,
  "height": 480,
  "inference_ms": 92.4,
  "detections": [
    {"label": "person", "conf": 0.912, "box": [0.11, 0.21, 0.43, 0.98]},
    {"label": "cup", "conf": 0.845, "box": [0.65, 0.72, 0.89, 0.95]}
  ]
}
```

- `width`, `height`: dimensions of the received image
- `box`: `[x1, y1, x2, y2]` normalized to 0–1 relative to the image
- `conf`: confidence score (3 decimals)
- `inference_ms`: wall time of inference only (excludes decoding and network)

**Errors:**
- `400`: undecodable image
- `422`: invalid or missing `conf` parameter
- `500`: server error (model or other)

**Example:**
```bash
curl -s -X POST --data-binary @photo.jpg \
  -H "Content-Type: image/jpeg" \
  "http://localhost:8000/detect?conf=0.35"
```

## Networking and HTTPS

### Local Network Access

By default, the server binds to `0.0.0.0:8000`. From another device on the same network, access it at `http://<machine-ip>:8000`.

**On WSL2:** Windows Firewall may block inbound traffic. To allow it:
1. Open Settings → Privacy & Security → Windows Defender Firewall → Allow an app through firewall
2. Add your WSL2 application or open port 8000 manually

Alternatively, use port forwarding:
```powershell
netsh interface portproxy add v4tov4 listenport=8000 connectaddress=<wsl-ip> connectport=8000
```

### HTTPS (Self-Signed)

Camera access over HTTP from a remote device is blocked by browsers. Use HTTPS instead:

```bash
openssl req -x509 -newkey rsa:2048 -nodes \
  -keyout key.pem -out cert.pem -days 30 \
  -subj "/CN=localhost" -addext "subjectAltName=DNS:localhost,IP:127.0.0.1"

python server.py  # Now serves HTTPS
```

The server automatically detects `cert.pem` and `key.pem` in the project folder and switches to HTTPS. You will see a browser security warning—this is expected for self-signed certificates. Click "Advanced" and proceed.

**Note:** Certificates are **never** committed to the repository.

## Performance

A few FPS is typical on CPU. Performance depends on:

1. **Model size:** `yolo26n` (nano) < `yolo26s` (small) < `yolo26m` (medium) < `yolo26l` (large)
2. **Inference size:** lower `YOLO_IMGSZ` is faster (e.g., 480 px vs. 640 px)
3. **CPU cores:** more threads = faster
4. **Export format:** ONNX and OpenVINO are faster than PyTorch on CPU

For example, on a modern 8-core CPU:
- `yolo26s.pt` at 640×640: ~100–150 ms per frame (6–10 FPS)
- `yolo26s.pt` at 480×480: ~60–100 ms per frame (10–16 FPS)
- `yolo26n.pt` at 640×640: ~40–60 ms per frame (16–25 FPS)

See the [CLAUDE.md](CLAUDE.md#11-performance) for tuning guidance.

## Models

The default is `yolo26s.pt` (small). On first run, Ultralytics downloads weights (~47 MB). Subsequent runs use the cached weights.

Available models:
- `yolo26n.pt` (nano, ~8 MB)
- `yolo26s.pt` (small, ~47 MB)
- `yolo26m.pt` (medium, ~142 MB)
- `yolo26l.pt` (large, ~366 MB)

Smaller models are faster; larger models are more accurate on small objects.

### Exported Models

For better CPU performance, export a model once:

```bash
pip install onnx onnxruntime  # For ONNX
yolo export model=yolo26s.pt format=onnx imgsz=640

# Then use it:
YOLO_MODEL=yolo26s.onnx python server.py
```

OpenVINO export for Intel CPUs:
```bash
yolo export model=yolo26s.pt format=openvino imgsz=640
YOLO_MODEL=yolo26s_openvino_model python server.py
```

## Testing

Run the contract test suite (no weights required):
```bash
pip install pytest httpx
python -m pytest -q
```

Run model smoke tests (requires weights):
```bash
python -m pytest -q -m model
```

Lint:
```bash
pip install ruff
ruff check .
```

## Known Limitations

- **CPU-only:** No GPU or CUDA support by design
- **Demo scope:** No authentication, rate limiting, or persistent storage
- **LAN only:** Not designed for internet exposure
- **First run:** Requires internet to download pretrained weights (~50 MB)
- **One model at a time:** Cannot run multiple models concurrently
- **No training:** Weights are pretrained; fine-tuning is not supported
- **Overlay lag:** Boxes display one frame behind live video (expected behavior)

## Architecture

```
Browser (index.html)
  ↓ POST /detect (JPEG)
  ↓
Server (server.py + FastAPI)
  ↓ NumPy + OpenCV decode
  ↓
Ultralytics YOLO26 (CPU)
  ↓ boxes, classes, confidence
  ↓ JSON response
Browser draws boxes
```

## Troubleshooting

**"Cannot access camera from remote device"**  
Cameras require HTTPS over the network. Generate a self-signed certificate (see [HTTPS](#https-self-signed) section).

**"ImportError: libGL.so.1"**  
On headless Linux, install OpenCV dependencies:
```bash
sudo apt-get install libgl1 libglib2.0-0
```

**"CUDA detected" or `torch.cuda.is_available()` is `True`**  
This should not happen if you installed the CPU build first. Verify:
```bash
python -c "import torch; print(f'CUDA: {torch.version.cuda}'); print(torch.cuda.is_available())"
```

**Slow inference**  
Check the server's `/info` endpoint:
```bash
curl http://localhost:8000/info
```

Verify the model size and image size match your CPU. See [Performance](#performance) for tuning guidance.

## Code Structure

- `server.py` – FastAPI server, YOLO inference, API endpoints
- `index.html` – Browser client: camera, image/video input, canvas overlay
- `CLAUDE.md` – Project constitution and technical specification
- `tests/` – Contract and model smoke tests

## Development

Read [CLAUDE.md](CLAUDE.md) for the full project specification, including:
- Architecture details
- Configuration and API contract
- Non-negotiable invariants (CPU-only, privacy, etc.)
- Testing requirements
- Security considerations

## License

See [LICENSE](LICENSE) for details.

---

**Made with ❤️ for demos.**
