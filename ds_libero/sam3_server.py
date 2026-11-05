"""Isolated SAM3 image worker. Run with the SAM3 Python, never the simulator Python.

The loopback-only service loads local weights once and serializes inference.
Imports of torch/SAM3 are deferred so --help and the client check stay lightweight.
"""

import argparse
import base64
import hashlib
import io
import importlib.metadata
import json
import platform
import sys
import time
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from types import MethodType

from .config import Config, digest
from .errors import ExperimentError
from .sam3_tools import WIRE_VERSION, Sam3Tool, validate_arguments


DEFAULT_CODE = "/mnt/cpfs/users/mrq/package/sam3/sam3"
DEFAULT_CHECKPOINT = "/mnt/cpfs/users/mrq/package/sam3/checkpoints"
SAM2_NECK_PREFIX = "backbone.vision_backbone.sam2_convs."
MAX_REQUEST_BYTES = 4 * 1024 * 1024


def file_hash(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def resolve_paths(code, checkpoint, bpe=None):
    code = Path(code).resolve()
    if (code / "sam3" / "model_builder.py").is_file():
        code = code / "sam3"
    if not (code / "model_builder.py").is_file():
        raise ExperimentError(f"SAM3 model_builder.py not found in {code}")
    checkpoint = Path(checkpoint).resolve()
    if checkpoint.is_dir():
        preferred = checkpoint / "sam3.pt"
        candidates = sorted([*checkpoint.glob("*.pt"), *checkpoint.glob("*.pth")])
        if preferred.is_file():
            checkpoint = preferred
        elif len(candidates) == 1:
            checkpoint = candidates[0]
        else:
            raise ExperimentError("Checkpoint directory must contain sam3.pt or one .pt/.pth file; otherwise pass --checkpoint FILE")
    if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
        raise ExperimentError(f"Local SAM3 checkpoint is missing/empty: {checkpoint}")
    bpe = Path(bpe).resolve() if bpe else code / "assets" / "bpe_simple_vocab_16e6.txt.gz"
    if not bpe.is_file():
        raise ExperimentError(f"SAM3 tokenizer missing: {bpe}; pass --bpe FILE")
    return code, checkpoint, bpe


def png_bytes(image):
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def load_local_image_weights(model, checkpoint, torch):
    """Official SAM3 checkpoints bundle detector and tracker; only detector is used.

    Detector weights also contain an optional SAM2 neck. Exclude that branch only
    when the model explicitly disables it; strictly validate all active weights.
    Upstream's blanket non-strict load could accept partially initialized weights.
    """
    state = torch.load(str(checkpoint), map_location="cpu", weights_only=True)
    if isinstance(state, dict) and isinstance(state.get("model"), dict):
        state = state["model"]
    if not isinstance(state, dict) or not state or any(not isinstance(k, str) for k in state):
        raise ExperimentError("SAM3 checkpoint has no state dictionary")
    if any(key.startswith("detector.") for key in state):
        state = {key[len("detector."):]: value for key, value in state.items() if key.startswith("detector.")}
    neck = getattr(getattr(model, "backbone", None), "vision_backbone", None)
    disabled_sam2 = (getattr(neck, "sam2_convs", False) is None
                     and getattr(model, "inst_interactive_predictor", False) is None)
    expected_keys = set(model.state_dict())
    ignored = sorted(key for key in state if disabled_sam2
                     and key.startswith(SAM2_NECK_PREFIX) and key not in expected_keys)
    if ignored:
        state = dict(state)
        for key in ignored:
            del state[key]
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise ExperimentError(f"SAM3 image weights do not match the local code (strict load): {exc}") from exc
    return {"strict": True, "loaded_keys": len(state), "ignored_disabled_sam2_keys": ignored}


def _native_vit_mlp_forward(self, x):
    """Preserve the MLP layers without SAM3 addmm_act's unconditional BF16 cast."""
    x = self.act(self.fc1(x))
    x = self.norm(self.drop1(x))
    return self.drop2(self.fc2(x))


def configure_model_precision(model, precision):
    policy = {"autocast_enabled": precision == "bfloat16", "float32_mlp_modules": [],
              "vit_mlp_implementation": "upstream"}
    if precision == "float32":
        from sam3.model.vitdet import Mlp
        # Module.float() preserves complex RoPE buffers; .to(dtype=float32) would not.
        model.float()
        for name, module in model.named_modules():
            if type(module) is Mlp:
                module.forward = MethodType(_native_vit_mlp_forward, module)
                policy["float32_mlp_modules"].append(name)
        policy["vit_mlp_implementation"] = "pytorch_float32"
    return policy


def encode_predictions(image, masks, boxes, scores, max_instances):
    """Convert original-resolution predictions to bounded, auditable 2D results."""
    import numpy as np
    from PIL import Image, ImageDraw
    masks, boxes, scores = np.asarray(masks), np.asarray(boxes), np.asarray(scores)
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]
    width, height = image.size
    if (masks.shape != (len(scores), height, width) or boxes.shape != (len(scores), 4)
            or scores.ndim != 1 or not np.isfinite(scores).all() or not np.isfinite(boxes).all()
            or np.any(scores < 0) or np.any(scores > 1)):
        raise ValueError("Unexpected SAM3 masks/boxes/scores shapes or values")
    detections = []
    overlay = image.convert("RGBA")
    palette = [(255, 70, 70), (50, 220, 80), (65, 140, 255), (250, 190, 35), (210, 75, 230), (30, 220, 220)]
    # Stable score order; do not discard alternative instances or fuse cameras.
    for idx in np.argsort(-scores, kind="stable"):
        mask = masks[idx].astype(bool)
        ys, xs = np.nonzero(mask)
        if not len(xs):
            continue
        box = boxes[idx].astype(float).copy()
        box[[0, 2]] = np.clip(box[[0, 2]], 0, width)
        box[[1, 3]] = np.clip(box[[1, 3]], 0, height)
        if box[0] >= box[2] or box[1] >= box[3]:
            continue
        number = len(detections) + 1
        center = [float(xs.mean()), float(ys.mean())]
        binary = Image.fromarray(mask.astype("uint8") * 255)
        detections.append({"id": number, "score": float(scores[idx]), "bbox_xyxy": box.tolist(),
            "centroid_uv": center, "mask_area_pixels": int(mask.sum()),
            "mask_area_fraction": float(mask.mean()),
            "mask_png_base64": base64.b64encode(png_bytes(binary)).decode("ascii")})
        color = palette[(number - 1) % len(palette)]
        rgba = np.zeros((height, width, 4), dtype="uint8")
        rgba[mask] = (*color, 90)
        overlay = Image.alpha_composite(overlay, Image.fromarray(rgba))
        draw = ImageDraw.Draw(overlay)
        draw.rectangle(box.tolist(), outline=(*color, 255), width=2)
        u, v = center
        draw.line((u - 3, v, u + 3, v), fill="white", width=1)
        draw.line((u, v - 3, u, v + 3), fill="white", width=1)
        draw.text((float(box[0]) + 2, float(box[1]) + 2), str(number), fill="white", stroke_width=1, stroke_fill="black")
        if len(detections) == max_instances:
            break
    return {"width": width, "height": height, "detections": detections,
            "coordinate_system": "upright original-image pixels; origin top-left; u right, v down; xyxy boxes",
            "overlay_png_base64": base64.b64encode(png_bytes(overlay.convert("RGB"))).decode("ascii")}


class Sam3Backend:
    def __init__(self, code=DEFAULT_CODE, checkpoint=DEFAULT_CHECKPOINT, bpe=None,
                 device="cuda", precision="bfloat16"):
        code, checkpoint, bpe = resolve_paths(code, checkpoint, bpe)
        sys.path.insert(0, str(code.parent))  # Package parent, NOT the sam3/ package directory.
        import torch
        import sam3
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
        if Path(sam3.__file__).resolve().parent != code:
            raise ExperimentError("A different SAM3 installation was imported; use a fresh worker process")
        if device == "cuda" and not torch.cuda.is_available():
            raise ExperimentError("SAM3 CUDA is unavailable in this Python; use a GPU environment or explicit --device cpu --precision float32")
        if device == "cpu" and precision != "float32":
            raise ExperimentError("CPU worker requires --precision float32")
        if precision == "bfloat16" and not torch.cuda.is_bf16_supported():
            raise ExperimentError("This GPU does not support bfloat16; use --precision float32")
        self.torch, self.device, self.precision = torch, device, precision
        print(f"Using local SAM3 package: {code}", flush=True)
        print(f"Loading local SAM3 checkpoint: {checkpoint}", flush=True)
        # No downloads and no random-weight fallback when the local file is absent.
        self.model = build_sam3_image_model(checkpoint_path=None, bpe_path=str(bpe),
            device="cpu", eval_mode=True, load_from_HF=False, enable_segmentation=True,
            enable_inst_interactivity=False, compile=False)
        checkpoint_load = load_local_image_weights(self.model, checkpoint, torch)
        print(f"SAM3 image weights loaded (strict): {checkpoint_load['loaded_keys']} keys; "
              f"ignored {len(checkpoint_load['ignored_disabled_sam2_keys'])} disabled SAM2 neck keys", flush=True)
        precision_policy = configure_model_precision(self.model, precision)
        print(f"SAM3 precision: {precision}; ViT MLP: {precision_policy['vit_mlp_implementation']} "
              f"({len(precision_policy['float32_mlp_modules'])} float32 overrides)", flush=True)
        self.model.to(device).eval()
        self.processor = Sam3Processor(self.model, device=device, confidence_threshold=0.35)
        self.cache = OrderedDict()  # At most the two current camera embeddings.
        versions = {}
        for name in ("torchvision", "numpy", "Pillow", "timm", "triton"):
            try:
                versions[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                versions[name] = None
        self.identity = {"schema": WIRE_VERSION, "code_path": str(code), "checkpoint_path": str(checkpoint),
            "checkpoint_sha256": file_hash(checkpoint), "bpe_sha256": file_hash(bpe),
            "checkpoint_load": checkpoint_load,
            "precision_policy": precision_policy,
            "sam3_source_sha256": digest({p.relative_to(code).as_posix(): file_hash(p) for p in sorted(code.rglob("*.py"))}),
            "worker_source_sha256": file_hash(Path(__file__)), "torch": torch.__version__,
            "tool_source_sha256": file_hash(Path(__file__).with_name("sam3_tools.py")), "dependencies": versions,
            "python": platform.python_version(), "device": device, "precision": precision,
            "gpu": torch.cuda.get_device_name() if device == "cuda" else None}
        self.runtime_id = digest(self.identity)

    def health(self):
        return {"schema": WIRE_VERSION, "status": "ready", "identity": self.identity, "runtime_id": self.runtime_id}

    def segment(self, request):
        from PIL import Image
        if (not isinstance(request, dict) or set(request) != {
                "schema", "image_base64", "prompt", "confidence_threshold", "max_instances"}
                or request["schema"] != WIRE_VERSION):
            raise ValueError("Invalid segment request schema")
        prompt = validate_arguments({"camera": "agentview", "prompt": request["prompt"]})["prompt"]
        threshold, limit = request["confidence_threshold"], request["max_instances"]
        if (type(threshold) not in (float, int) or not 0 < threshold < 1
                or type(limit) is not int or not 1 <= limit <= 16):
            raise ValueError("Invalid threshold or instance limit")
        raw = base64.b64decode(request["image_base64"], validate=True)
        with Image.open(io.BytesIO(raw)) as source:
            if source.format != "PNG" or source.size != (256, 256):
                raise ValueError("Expected the benchmark's upright 256x256 PNG")
            image = source.convert("RGB")  # Already upright; NEVER flip again.
        key = hashlib.sha256(raw).hexdigest()
        started = time.perf_counter()
        autocast = (self.torch.autocast(device_type=self.device, dtype=self.torch.bfloat16)
                    if self.precision == "bfloat16"
                    else self.torch.autocast(device_type=self.device, enabled=False))
        with self.torch.inference_mode(), autocast:
            cached = key in self.cache
            if not cached:
                # Drop old embeddings BEFORE allocating a new one.
                while len(self.cache) >= 2:
                    self.cache.popitem(last=False)
                self.cache[key] = self.processor.set_image(image)
            self.cache.move_to_end(key)
            state = self.cache[key]
            self.processor.reset_all_prompts(state)
            self.processor.set_confidence_threshold(float(threshold))
            output = self.processor.set_text_prompt(prompt=prompt, state=state)
            masks = output["masks"].detach().cpu().numpy()
            boxes = output["boxes"].detach().float().cpu().numpy()
            scores = output["scores"].detach().float().cpu().numpy()
            # Retain only the image embeddings, not masks from the previous query.
            self.processor.reset_all_prompts(state)
        result = encode_predictions(image, masks, boxes, scores, limit)
        result.update(schema=WIRE_VERSION, runtime_id=self.runtime_id, image_sha256=key,
                      prompt=prompt, confidence_threshold=threshold, image_embedding_cached=cached,
                      inference_seconds=time.perf_counter() - started)
        return result


def make_server(backend, host="127.0.0.1", port=8765):
    if host not in ("127.0.0.1", "localhost"):
        raise ExperimentError("SAM3 worker binds only to loopback; use SSH forwarding across hosts")

    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup()
            self.connection.settimeout(30)

        def respond(self, status, value):
            data = json.dumps(value, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            self.respond(200, backend.health()) if self.path == "/health" else self.respond(404, {"error": "not found"})

        def do_POST(self):
            if self.path != "/segment":
                self.respond(404, {"error": "not found"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= MAX_REQUEST_BYTES:
                    raise ValueError("Invalid request size")
                body = self.rfile.read(length)
                if len(body) != length:
                    raise ValueError("Incomplete request")
                request = json.loads(body)
                result = backend.segment(request)
            except (ValueError, TypeError, ExperimentError) as exc:
                self.respond(400, {"schema": WIRE_VERSION, "error": str(exc)})
                return
            except Exception as exc:
                # Full traceback stays on the worker, no fabricated empty detections.
                import traceback
                traceback.print_exc()
                self.respond(500, {"schema": WIRE_VERSION, "error": type(exc).__name__})
                return
            self.respond(200, result)

    return HTTPServer((host, port), Handler)  # Single-threaded: processor state is mutable.


def check(args):
    from PIL import Image
    from .artifacts import Store
    from .protocol import CAMERAS, Observation
    from .report import build_report
    cfg = Config.load(args.config)
    if not cfg.sam3:
        raise ExperimentError("Check requires a SAM3-enabled config")
    store = Store(args.out, "sam3-check", cfg)
    result = {"status": "incomplete", "scored": False, "task_success": None, "model_actions_executed": 0}
    started = time.perf_counter()
    try:
        if args.image:
            with Image.open(args.image) as source:
                if source.size != (256, 256):
                    raise ExperimentError("Smoke image must be an upright 256x256 camera frame")
                image = source.convert("RGB")
        else:
            image = Image.new("RGB", (256, 256), (160, 160, 160))
        obs = Observation({}, {c: png_bytes(image) for c in CAMERAS})
        store.observe(0, obs)
        tool = Sam3Tool(cfg.sam3, store)
        tool.execute({"camera": "agentview", "prompt": args.prompt}, obs, 0, "smoke")
        result.update(status="pass", note="Real SAM3 inference/transport smoke only; not segmentation accuracy or task success.",
                      input="provided image" if args.image else "blank smoke image")
    except Exception as exc:
        result.update(status=getattr(exc, "status", "runtime_error"), reason=f"{type(exc).__name__}: {exc}")
    finally:
        result["command_wall_seconds"] = time.perf_counter() - started
        store.finish(result)
    build_report(store.path, replay=False)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0 if result["status"] == "pass" else 2


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    serve = sub.add_parser("serve", help="Run with your SAM3/GPU Python")
    serve.add_argument("--sam3-code", default=DEFAULT_CODE)
    serve.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    serve.add_argument("--bpe")
    serve.add_argument("--host", default="127.0.0.1", choices=("127.0.0.1", "localhost"))
    serve.add_argument("--port", type=int, default=8765)
    serve.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    serve.add_argument("--precision", choices=("bfloat16", "float32"), default="bfloat16")
    smoke = sub.add_parser("check", help="Real local SAM3 inference; no DeepSeek call or simulator")
    smoke.add_argument("--config", default="configs/deepseek_sam3_220.json")
    smoke.add_argument("--out", required=True)
    smoke.add_argument("--image", help="Optional upright camera PNG, 256x256")
    smoke.add_argument("--prompt", default="object")
    args = parser.parse_args(argv)
    try:
        if args.command == "check":
            return check(args)
        backend = Sam3Backend(args.sam3_code, args.checkpoint, args.bpe, args.device, args.precision)
        with make_server(backend, args.host, args.port) as server:
            print(json.dumps(backend.health(), indent=2), flush=True)
            print(f"SAM3 ready at http://{args.host}:{server.server_port}", flush=True)
            server.serve_forever()
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"SAM3 failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
