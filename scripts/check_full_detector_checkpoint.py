from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch


PROJECT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT.parent
MMDET_ROOT = Path(os.environ.get("MMDET_ROOT", WORKSPACE / "mmdetection"))
GENEVAL_VENV_SITE = os.environ.get("GENEVAL_VENV_SITE")
if GENEVAL_VENV_SITE:
    sys.path.append(GENEVAL_VENV_SITE)
sys.path.insert(0, str(MMDET_ROOT))
sys.path.insert(0, str(PROJECT / "src"))

import jax.numpy as jnp  # noqa: E402
from mmcv import Config  # noqa: E402
from mmdet.models import build_detector  # noqa: E402
from mmcv.runner import load_checkpoint  # noqa: E402

from jax_geneval.config import DetectorConfig  # noqa: E402
from jax_geneval.convert import convert_mask2former_detector_state_dict  # noqa: E402
from jax_geneval.detector import mask2former_detector_forward  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=os.environ.get(
            "JAX_GENEVAL_MMDET_CONFIG",
            str(MMDET_ROOT / "configs/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.py"),
        ),
    )
    parser.add_argument(
        "--checkpoint",
        default=os.environ.get("JAX_GENEVAL_DETECTOR_CKPT"),
    )
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()
    if not args.checkpoint:
        raise ValueError(
            "Missing detector checkpoint. Pass --checkpoint or set JAX_GENEVAL_DETECTOR_CKPT."
        )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cfg = Config.fromfile(args.config)
    cfg.model.train_cfg = None
    model = build_detector(cfg.model, test_cfg=cfg.get("test_cfg"))
    load_checkpoint(model, args.checkpoint, map_location="cpu")
    model.eval()

    rng = np.random.default_rng(args.seed)
    x = rng.normal(size=(1, 3, args.height, args.width)).astype(np.float32)
    img_metas = [
        {
            "batch_input_shape": (args.height, args.width),
            "img_shape": (args.height, args.width, 3),
            "ori_shape": (args.height, args.width, 3),
        }
    ]
    with torch.no_grad():
        feats = model.extract_feat(torch.from_numpy(x))
        expected_cls, expected_mask = model.panoptic_head.simple_test(feats, img_metas)

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    params = convert_mask2former_detector_state_dict(checkpoint["state_dict"])
    detector_cfg = DetectorConfig(input_height=args.height, input_width=args.width)
    actual_cls, actual_mask = mask2former_detector_forward(
        jnp.asarray(x.transpose(0, 2, 3, 1)),
        params,
        detector_cfg=detector_cfg,
    )
    actual_cls_np = np.asarray(actual_cls)
    actual_mask_np = np.asarray(actual_mask)
    expected_cls_np = expected_cls.numpy()
    expected_mask_np = expected_mask.numpy()
    cls_diff = float(np.max(np.abs(actual_cls_np - expected_cls_np)))
    mask_diff = float(np.max(np.abs(actual_mask_np - expected_mask_np)))
    print(f"Full detector cls max_diff={cls_diff:.6g}")
    print(f"Full detector mask max_diff={mask_diff:.6g}")
    np.testing.assert_allclose(actual_cls_np, expected_cls_np, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(actual_mask_np, expected_mask_np, rtol=2e-4, atol=2e-4)
    print("[ok] full checkpoint detector parity passed")


if __name__ == "__main__":
    main()
