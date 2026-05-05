from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import torch
from PIL import Image, ImageOps


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from jax_geneval.color import JaxClipColorClassifier  # noqa: E402
from jax_geneval.config import DetectorConfig  # noqa: E402
from jax_geneval.convert import convert_mask2former_detector_state_dict  # noqa: E402
from jax_geneval.detector import mask2former_detector_forward, mask2former_detector_instances  # noqa: E402
from jax_geneval.evaluation import EvalOptions, evaluate_detector_outputs, evaluate_instance_outputs  # noqa: E402
from jax_geneval.preprocess import preprocess_image  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("imagedir")
    parser.add_argument("--outfile", default="jax_geneval_results.jsonl")
    parser.add_argument(
        "--checkpoint",
        default=os.environ.get("JAX_GENEVAL_DETECTOR_CKPT"),
        help="Path to mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.pth. "
        "Can also be set with JAX_GENEVAL_DETECTOR_CKPT.",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--input-height", type=int, default=800)
    parser.add_argument("--input-width", type=int, default=800)
    parser.add_argument("--output-height", type=int, default=0)
    parser.add_argument("--output-width", type=int, default=0)
    parser.add_argument("--max-prompts", type=int, default=0)
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--worker-id", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--counting-threshold", type=float, default=0.9)
    parser.add_argument("--max-objects", type=int, default=16)
    parser.add_argument("--max-overlap", type=float, default=1.0)
    parser.add_argument("--position-threshold", type=float, default=0.1)
    parser.add_argument("--compile", choices=["auto", "jit", "pjit"], default="auto")
    parser.add_argument("--host-instance-postprocess", action="store_true")
    parser.add_argument("--skip-clip", action="store_true")
    parser.add_argument("--eager-clip", action="store_true")
    parser.add_argument("--clip-model", default="ViT-L/14")
    parser.add_argument(
        "--clip-repo",
        default=os.environ.get("JAX_CLIP_REPO"),
        help="Optional path to legacy/jax-clip. Can also be set with JAX_CLIP_REPO.",
    )
    parser.add_argument("--clip-batch-size", type=int, default=16)
    parser.add_argument("--benchmark-out", default=None)
    return parser.parse_args()


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", file=sys.stderr, flush=True)


def load_params(checkpoint_path: str) -> dict[str, object]:
    if not checkpoint_path:
        raise ValueError(
            "Missing detector checkpoint. Pass --checkpoint /path/to/model.pth "
            "or set JAX_GENEVAL_DETECTOR_CKPT."
        )
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    return convert_mask2former_detector_state_dict(checkpoint["state_dict"])


def create_infer_fn(
    params: dict[str, object],
    *,
    detector_cfg: DetectorConfig,
    compile_mode: str,
    batch_size: int,
):
    def forward(p, images):
        if not detector_cfg.device_instance_postprocess:
            return mask2former_detector_forward(images, p, detector_cfg=detector_cfg)
        return mask2former_detector_instances(images, p, detector_cfg=detector_cfg)

    devices = jax.devices()
    use_pjit = compile_mode == "pjit" or (compile_mode == "auto" and len(devices) > 1)
    if use_pjit:
        if batch_size % len(devices) != 0:
            raise ValueError(
                f"batch_size={batch_size} must be divisible by device_count={len(devices)} for pjit"
            )
        from jax.experimental.pjit import pjit
        from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

        mesh = Mesh(np.asarray(devices), ("data",))
        replicated = NamedSharding(mesh, P())
        image_sharding = NamedSharding(mesh, P("data", None, None, None))
        out_sharding = (
            NamedSharding(mesh, P("data", None)),
            NamedSharding(mesh, P("data", None, None)),
            NamedSharding(mesh, P("data", None, None, None)),
        )
        if not detector_cfg.device_instance_postprocess:
            out_sharding = (
                NamedSharding(mesh, P("data", None, None)),
                NamedSharding(mesh, P("data", None, None, None)),
            )
        with mesh:
            params_dev = jax.device_put(params, replicated)
            compiled = pjit(
                forward,
                in_shardings=(replicated, image_sharding),
                out_shardings=out_sharding,
            )

        def infer(images: np.ndarray):
            with mesh:
                images_dev = jax.device_put(jnp.asarray(images), image_sharding)
                return compiled(params_dev, images_dev)

        return infer, "pjit"

    params_dev = jax.device_put(params)
    compiled = jax.jit(forward)

    def infer(images: np.ndarray):
        return compiled(params_dev, jnp.asarray(images))

    return infer, "jit"


def collect_jobs(args: argparse.Namespace):
    subfolders = [
        subfolder
        for subfolder in sorted(os.listdir(args.imagedir))
        if os.path.isdir(os.path.join(args.imagedir, subfolder)) and subfolder.isdigit()
    ]
    if args.num_workers > 1:
        subfolders = [
            subfolder
            for index, subfolder in enumerate(subfolders)
            if index % args.num_workers == args.worker_id
        ]
    jobs = []
    for subfolder in subfolders:
        if args.max_prompts and len(jobs) >= args.max_prompts:
            break
        folderpath = os.path.join(args.imagedir, subfolder)
        with open(os.path.join(folderpath, "metadata.jsonl")) as fp:
            metadata = json.load(fp)
        imagenames = [
            name
            for name in sorted(os.listdir(os.path.join(folderpath, "samples")))
            if os.path.isfile(os.path.join(folderpath, "samples", name)) and re.match(r"\d+\.png", name)
        ]
        if args.max_images:
            imagenames = imagenames[: args.max_images]
        for imagename in imagenames:
            jobs.append(
                {
                    "subfolder": subfolder,
                    "metadata": metadata,
                    "imagepath": os.path.join(folderpath, "samples", imagename),
                }
            )
    return jobs


def metadata_needs_color(metadata: dict[str, object]) -> bool:
    return any("color" in req for req in metadata.get("include", []))


def main() -> None:
    args = parse_args()
    if args.worker_id < 0 or args.worker_id >= args.num_workers:
        raise ValueError(f"worker-id must be in [0, {args.num_workers}), got {args.worker_id}")

    detector_cfg = DetectorConfig(
        input_height=args.input_height,
        input_width=args.input_width,
        output_height=args.output_height,
        output_width=args.output_width,
        device_instance_postprocess=not args.host_instance_postprocess,
    )
    eval_options = EvalOptions(
        threshold=args.threshold,
        counting_threshold=args.counting_threshold,
        max_objects=args.max_objects,
        nms_threshold=args.max_overlap,
        position_threshold=args.position_threshold,
    )

    load_t0 = time.perf_counter()
    params = load_params(args.checkpoint)
    load_dt = time.perf_counter() - load_t0
    infer, infer_mode = create_infer_fn(
        params,
        detector_cfg=detector_cfg,
        compile_mode=args.compile,
        batch_size=args.batch_size,
    )
    log(f"loaded checkpoint in {load_dt:.3f}s; using {infer_mode} on {len(jax.devices())} device(s)")

    jobs = collect_jobs(args)
    log(f"worker {args.worker_id}/{args.num_workers} evaluating {len(jobs)} images")

    color_classifier = None
    if not args.skip_clip and args.eager_clip and any(metadata_needs_color(job["metadata"]) for job in jobs):
        color_classifier = JaxClipColorClassifier(
            model_name=args.clip_model,
            batch_size=args.clip_batch_size,
            repo_root=args.clip_repo,
        )

    if os.path.dirname(args.outfile):
        os.makedirs(os.path.dirname(args.outfile), exist_ok=True)

    metrics = {
        "load_s": load_dt,
        "compile_s": 0.0,
        "preprocess_s": 0.0,
        "inference_s": 0.0,
        "postprocess_s": 0.0,
        "num_images": len(jobs),
        "batch_size": args.batch_size,
        "infer_mode": infer_mode,
    }
    correct = 0
    compile_done = False
    with open(args.outfile, "w") as out_fp:
        for batch_start in range(0, len(jobs), args.batch_size):
            batch_jobs = jobs[batch_start : batch_start + args.batch_size]
            prep_t0 = time.perf_counter()
            images = []
            metas = []
            pil_images = []
            for job in batch_jobs:
                arr, meta = preprocess_image(job["imagepath"], (args.input_height, args.input_width))
                images.append(arr)
                metas.append(meta)
                pil_images.append(ImageOps.exif_transpose(Image.open(job["imagepath"])).convert("RGB"))
            if len(images) < args.batch_size:
                images.extend([np.zeros_like(images[-1])] * (args.batch_size - len(images)))
            image_batch = np.stack(images, axis=0).astype(np.float32)
            metrics["preprocess_s"] += time.perf_counter() - prep_t0

            infer_t0 = time.perf_counter()
            outputs = infer(image_batch)
            outputs = jax.block_until_ready(outputs)
            infer_dt = time.perf_counter() - infer_t0
            if not compile_done:
                metrics["compile_s"] = infer_dt
                compile_done = True
            metrics["inference_s"] += infer_dt

            post_t0 = time.perf_counter()
            output_np = tuple(np.asarray(x) for x in outputs)
            for i, job in enumerate(batch_jobs):
                if metadata_needs_color(job["metadata"]):
                    if args.skip_clip:
                        raise RuntimeError("color metadata encountered but --skip-clip was set")
                    if color_classifier is None:
                        color_classifier = JaxClipColorClassifier(
                            model_name=args.clip_model,
                            batch_size=args.clip_batch_size,
                            repo_root=args.clip_repo,
                        )
                if args.host_instance_postprocess:
                    mask_cls_np, mask_pred_np = output_np
                    result = evaluate_detector_outputs(
                        job["imagepath"],
                        pil_images[i],
                        job["metadata"],
                        metas[i],
                        mask_cls_np[i],
                        mask_pred_np[i],
                        color_classifier=color_classifier,
                        options=eval_options,
                    )
                else:
                    labels_np, bboxes_np, masks_np = output_np
                    result = evaluate_instance_outputs(
                        job["imagepath"],
                        pil_images[i],
                        job["metadata"],
                        labels_np[i],
                        bboxes_np[i],
                        masks_np[i],
                        color_classifier=color_classifier,
                        options=eval_options,
                    )
                correct += int(result["correct"])
                out_fp.write(json.dumps(result) + "\n")
            out_fp.flush()
            metrics["postprocess_s"] += time.perf_counter() - post_t0
            done = min(batch_start + len(batch_jobs), len(jobs))
            log(f"done {done}/{len(jobs)} images; running score={correct}/{done}")

    metrics["score"] = correct / len(jobs) if jobs else 0.0
    metrics["steady_inference_s"] = max(metrics["inference_s"] - metrics["compile_s"], 0.0)
    denom = max(len(jobs) - args.batch_size, 1)
    metrics["steady_images_per_s"] = denom / metrics["steady_inference_s"] if metrics["steady_inference_s"] else 0.0
    if args.benchmark_out:
        if os.path.dirname(args.benchmark_out):
            os.makedirs(os.path.dirname(args.benchmark_out), exist_ok=True)
        with open(args.benchmark_out, "w") as fp:
            json.dump(metrics, fp, indent=2)
    log(f"score={metrics['score']:.6f}; metrics={json.dumps(metrics, sort_keys=True)}")


if __name__ == "__main__":
    main()
