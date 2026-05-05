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

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from jax_geneval.convert import (  # noqa: E402
    convert_ms_deformable_attention_state_dict,
    convert_mask2former_head_state_dict,
    convert_pixel_decoder_state_dict,
    convert_swin_block_state_dict,
    convert_swin_transformer_state_dict,
    convert_window_msa_state_dict,
)
from jax_geneval.deformable_attention import multi_scale_deformable_attention_module  # noqa: E402
from jax_geneval.pixel_decoder import ms_deform_attn_pixel_decoder  # noqa: E402
from jax_geneval.mask2former_head import mask2former_head_forward  # noqa: E402
from jax_geneval.swin import (  # noqa: E402
    SwinBlockConfig,
    SwinTransformerConfig,
    WindowAttentionConfig,
    swin_transformer_forward,
    swin_block,
    window_attention,
)


def check_window_msa() -> None:
    from mmdet.models.backbones.swin import WindowMSA

    torch.manual_seed(0)
    np.random.seed(0)
    module = WindowMSA(embed_dims=12, num_heads=3, window_size=(4, 4))
    module.eval()
    x = torch.randn(5, 16, 12)
    with torch.no_grad():
        expected = module(x).numpy()
    params = convert_window_msa_state_dict(module.state_dict())
    cfg = WindowAttentionConfig(embed_dims=12, num_heads=3, window_size=4)
    actual = np.asarray(window_attention(jnp.asarray(x.numpy()), params, cfg))
    max_diff = float(np.max(np.abs(actual - expected)))
    print(f"WindowMSA max_diff={max_diff:.6g}")
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)


def check_swin_block() -> None:
    from mmdet.models.backbones.swin import SwinBlock

    torch.manual_seed(1)
    np.random.seed(1)
    module = SwinBlock(
        embed_dims=8,
        num_heads=2,
        feedforward_channels=32,
        window_size=4,
        shift=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
    )
    module.eval()
    x = torch.randn(2, 36, 8)
    with torch.no_grad():
        expected = module(x, (6, 6)).numpy()
    params = convert_swin_block_state_dict(module.state_dict())
    cfg = SwinBlockConfig(embed_dims=8, num_heads=2, window_size=4, shift=True)
    actual = np.asarray(swin_block(jnp.asarray(x.numpy()), params, cfg, (6, 6)))
    max_diff = float(np.max(np.abs(actual - expected)))
    print(f"SwinBlock max_diff={max_diff:.6g}")
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)


def check_swin_transformer() -> None:
    from mmdet.models.backbones.swin import SwinTransformer

    torch.manual_seed(5)
    np.random.seed(5)
    depths = (1, 1, 2, 1)
    num_heads = (1, 2, 4, 8)
    module = SwinTransformer(
        embed_dims=8,
        depths=depths,
        num_heads=num_heads,
        window_size=4,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        with_cp=False,
        init_cfg=None,
    )
    module.eval()
    x = torch.randn(1, 3, 31, 29)
    with torch.no_grad():
        expected = module(x)
    params = convert_swin_transformer_state_dict(
        module.state_dict(),
        depths=depths,
    )
    cfg = SwinTransformerConfig(
        embed_dims=8,
        depths=depths,
        num_heads=num_heads,
        window_size=4,
        patch_size=4,
        out_indices=(0, 1, 2, 3),
    )
    actual = swin_transformer_forward(jnp.asarray(x.numpy().transpose(0, 2, 3, 1)), params, cfg)
    for idx, (actual_level, expected_level) in enumerate(zip(actual, expected)):
        expected_np = expected_level.numpy().transpose(0, 2, 3, 1)
        max_diff = float(np.max(np.abs(np.asarray(actual_level) - expected_np)))
        print(f"SwinTransformer out[{idx}] max_diff={max_diff:.6g}")
        np.testing.assert_allclose(np.asarray(actual_level), expected_np, rtol=3e-5, atol=3e-5)


def check_ms_deformable_attention() -> None:
    from mmcv.ops.multi_scale_deform_attn import MultiScaleDeformableAttention

    torch.manual_seed(2)
    np.random.seed(2)
    module = MultiScaleDeformableAttention(
        embed_dims=16,
        num_heads=4,
        num_levels=2,
        num_points=3,
        dropout=0.0,
        batch_first=False,
    )
    module.eval()
    shapes = torch.tensor([[3, 2], [2, 3]], dtype=torch.long)
    starts = torch.cat((shapes.new_zeros((1,)), shapes.prod(1).cumsum(0)[:-1]))
    total = int((shapes[:, 0] * shapes[:, 1]).sum())
    query = torch.randn(total, 2, 16)
    query_pos = torch.randn(total, 2, 16) * 0.01
    reference_points = torch.rand(2, total, 2, 2)
    with torch.no_grad():
        expected = module(
            query,
            query_pos=query_pos,
            reference_points=reference_points,
            spatial_shapes=shapes,
            level_start_index=starts,
        ).numpy()
    params = convert_ms_deformable_attention_state_dict(module.state_dict())
    actual = np.asarray(
        multi_scale_deformable_attention_module(
            jnp.asarray(query.numpy()),
            params,
            value_spatial_shapes=jnp.asarray(shapes.numpy()),
            reference_points=jnp.asarray(reference_points.numpy()),
            level_start_index=jnp.asarray(starts.numpy()),
            query_pos=jnp.asarray(query_pos.numpy()),
            num_heads=4,
            num_levels=2,
            num_points=3,
            batch_first=False,
        )
    )
    max_diff = float(np.max(np.abs(actual - expected)))
    print(f"MultiScaleDeformableAttention max_diff={max_diff:.6g}")
    np.testing.assert_allclose(actual, expected, rtol=3e-5, atol=3e-5)


def check_pixel_decoder() -> None:
    from mmcv import ConfigDict
    from mmdet.models.plugins.msdeformattn_pixel_decoder import MSDeformAttnPixelDecoder

    torch.manual_seed(3)
    np.random.seed(3)
    encoder = ConfigDict(
        type="DetrTransformerEncoder",
        num_layers=2,
        transformerlayers=ConfigDict(
            type="BaseTransformerLayer",
            attn_cfgs=ConfigDict(
                type="MultiScaleDeformableAttention",
                embed_dims=8,
                num_heads=2,
                num_levels=3,
                num_points=2,
                im2col_step=64,
                dropout=0.0,
                batch_first=False,
                norm_cfg=None,
                init_cfg=None,
            ),
            ffn_cfgs=ConfigDict(
                type="FFN",
                embed_dims=8,
                feedforward_channels=32,
                num_fcs=2,
                ffn_drop=0.0,
                act_cfg=dict(type="ReLU", inplace=True),
            ),
            operation_order=("self_attn", "norm", "ffn", "norm"),
        ),
        init_cfg=None,
    )
    module = MSDeformAttnPixelDecoder(
        in_channels=[4, 8, 16, 32],
        strides=[4, 8, 16, 32],
        feat_channels=8,
        out_channels=8,
        num_outs=3,
        norm_cfg=dict(type="GN", num_groups=4),
        act_cfg=dict(type="ReLU"),
        encoder=encoder,
        positional_encoding=dict(type="SinePositionalEncoding", num_feats=4, normalize=True),
    )
    module.eval()
    feats = [
        torch.randn(1, 4, 8, 8),
        torch.randn(1, 8, 4, 4),
        torch.randn(1, 16, 2, 2),
        torch.randn(1, 32, 1, 1),
    ]
    with torch.no_grad():
        expected_mask, expected_multi = module(feats)
    state = module.state_dict()
    params = convert_pixel_decoder_state_dict(
        state,
        num_encoder_levels=3,
        num_encoder_layers=2,
        num_lateral_levels=1,
    )
    feats_jax = [jnp.asarray(f.numpy().transpose(0, 2, 3, 1)) for f in feats]
    actual_mask, actual_multi = ms_deform_attn_pixel_decoder(
        feats_jax,
        params,
        strides=(4, 8, 16, 32),
        num_encoder_levels=3,
        num_outs=3,
        num_heads=2,
        num_points=2,
        gn_groups=4,
    )
    expected_mask_np = expected_mask.numpy().transpose(0, 2, 3, 1)
    max_diff = float(np.max(np.abs(np.asarray(actual_mask) - expected_mask_np)))
    print(f"PixelDecoder mask_feature max_diff={max_diff:.6g}")
    np.testing.assert_allclose(np.asarray(actual_mask), expected_mask_np, rtol=5e-5, atol=5e-5)
    for idx, (actual, expected) in enumerate(zip(actual_multi, expected_multi)):
        expected_np = expected.numpy().transpose(0, 2, 3, 1)
        max_diff = float(np.max(np.abs(np.asarray(actual) - expected_np)))
        print(f"PixelDecoder multi[{idx}] max_diff={max_diff:.6g}")
        np.testing.assert_allclose(np.asarray(actual), expected_np, rtol=5e-5, atol=5e-5)


def _small_pixel_encoder_cfg():
    from mmcv import ConfigDict

    return ConfigDict(
        type="DetrTransformerEncoder",
        num_layers=2,
        transformerlayers=ConfigDict(
            type="BaseTransformerLayer",
            attn_cfgs=ConfigDict(
                type="MultiScaleDeformableAttention",
                embed_dims=8,
                num_heads=2,
                num_levels=3,
                num_points=2,
                im2col_step=64,
                dropout=0.0,
                batch_first=False,
                norm_cfg=None,
                init_cfg=None,
            ),
            ffn_cfgs=ConfigDict(
                type="FFN",
                embed_dims=8,
                feedforward_channels=32,
                num_fcs=2,
                ffn_drop=0.0,
                act_cfg=dict(type="ReLU", inplace=True),
            ),
            operation_order=("self_attn", "norm", "ffn", "norm"),
        ),
        init_cfg=None,
    )


def check_mask2former_head() -> None:
    from mmcv import ConfigDict
    from mmdet.models.dense_heads.mask2former_head import Mask2FormerHead

    torch.manual_seed(4)
    np.random.seed(4)
    pixel_decoder = ConfigDict(
        type="MSDeformAttnPixelDecoder",
        in_channels=[4, 8, 16, 32],
        strides=[4, 8, 16, 32],
        feat_channels=8,
        out_channels=8,
        num_outs=3,
        norm_cfg=dict(type="GN", num_groups=4),
        act_cfg=dict(type="ReLU"),
        encoder=_small_pixel_encoder_cfg(),
        positional_encoding=dict(type="SinePositionalEncoding", num_feats=4, normalize=True),
        init_cfg=None,
    )
    transformer_decoder = ConfigDict(
        type="DetrTransformerDecoder",
        return_intermediate=True,
        num_layers=2,
        transformerlayers=ConfigDict(
            type="DetrTransformerDecoderLayer",
            attn_cfgs=ConfigDict(
                type="MultiheadAttention",
                embed_dims=8,
                num_heads=2,
                attn_drop=0.0,
                proj_drop=0.0,
                dropout_layer=None,
                batch_first=False,
            ),
            ffn_cfgs=ConfigDict(
                embed_dims=8,
                feedforward_channels=16,
                num_fcs=2,
                act_cfg=dict(type="ReLU", inplace=True),
                ffn_drop=0.0,
                dropout_layer=None,
                add_identity=True,
            ),
            feedforward_channels=16,
            ffn_dropout=0.0,
            operation_order=("cross_attn", "norm", "self_attn", "norm", "ffn", "norm"),
        ),
        init_cfg=None,
    )
    head = Mask2FormerHead(
        in_channels=[4, 8, 16, 32],
        feat_channels=8,
        out_channels=8,
        num_things_classes=3,
        num_stuff_classes=0,
        num_queries=5,
        num_transformer_feat_level=3,
        pixel_decoder=pixel_decoder,
        enforce_decoder_input_project=False,
        transformer_decoder=transformer_decoder,
        positional_encoding=dict(type="SinePositionalEncoding", num_feats=4, normalize=True),
        loss_cls=ConfigDict(
            type="CrossEntropyLoss",
            use_sigmoid=False,
            loss_weight=2.0,
            reduction="mean",
            class_weight=[1.0, 1.0, 1.0, 0.1],
        ),
        loss_mask=ConfigDict(type="CrossEntropyLoss", use_sigmoid=True, reduction="mean", loss_weight=5.0),
        loss_dice=ConfigDict(type="DiceLoss", use_sigmoid=True, activate=True, reduction="mean", naive_dice=True, eps=1.0, loss_weight=5.0),
        train_cfg=None,
        test_cfg=ConfigDict(),
    )
    head.eval()
    feats = [
        torch.randn(1, 4, 8, 8),
        torch.randn(1, 8, 4, 4),
        torch.randn(1, 16, 2, 2),
        torch.randn(1, 32, 1, 1),
    ]
    with torch.no_grad():
        expected_cls, expected_masks = head(feats, img_metas=[{}])
    params = convert_mask2former_head_state_dict(
        head.state_dict(),
        num_encoder_layers=2,
        num_decoder_layers=2,
    )
    feats_jax = [jnp.asarray(f.numpy().transpose(0, 2, 3, 1)) for f in feats]
    actual_cls, actual_masks = mask2former_head_forward(
        feats_jax,
        params,
        num_heads=2,
        num_transformer_feat_level=3,
        num_decoder_layers=2,
        pixel_decoder_num_heads=2,
        pixel_decoder_num_points=2,
        gn_groups=4,
    )
    for idx, (actual, expected) in enumerate(zip(actual_cls, expected_cls)):
        expected_np = expected.numpy()
        max_diff = float(np.max(np.abs(np.asarray(actual) - expected_np)))
        print(f"Mask2FormerHead cls[{idx}] max_diff={max_diff:.6g}")
        np.testing.assert_allclose(np.asarray(actual), expected_np, rtol=8e-5, atol=8e-5)
    for idx, (actual, expected) in enumerate(zip(actual_masks, expected_masks)):
        expected_np = expected.numpy()
        max_diff = float(np.max(np.abs(np.asarray(actual) - expected_np)))
        print(f"Mask2FormerHead mask[{idx}] max_diff={max_diff:.6g}")
        np.testing.assert_allclose(np.asarray(actual), expected_np, rtol=8e-5, atol=8e-5)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--only",
        choices=["all", "window", "block", "swin", "deform", "pixel", "head"],
        default="all",
    )
    args = parser.parse_args()
    if args.only in ("all", "window"):
        check_window_msa()
    if args.only in ("all", "block"):
        check_swin_block()
    if args.only in ("all", "swin"):
        check_swin_transformer()
    if args.only in ("all", "deform"):
        check_ms_deformable_attention()
    if args.only in ("all", "pixel"):
        check_pixel_decoder()
    if args.only in ("all", "head"):
        check_mask2former_head()
    print("[ok] mmdet oracle checks passed")


if __name__ == "__main__":
    main()
