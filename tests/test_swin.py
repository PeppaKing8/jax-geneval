from __future__ import annotations

import numpy as np
import torch

from jax_geneval import ops
from jax_geneval.swin import (
    SwinBlockConfig,
    WindowAttentionConfig,
    relative_position_index,
    shifted_window_attention,
    swin_block,
    window_attention,
)


def torch_window_attention(x, params, cfg, mask=None):
    b, n, c = x.shape
    head_dim = c // cfg.num_heads
    scale = cfg.qk_scale or head_dim**-0.5
    qkv_w = torch.from_numpy(np.asarray(params["qkv"]["kernel"]).T)
    qkv_b = torch.from_numpy(np.asarray(params["qkv"]["bias"]))
    proj_w = torch.from_numpy(np.asarray(params["proj"]["kernel"]).T)
    proj_b = torch.from_numpy(np.asarray(params["proj"]["bias"]))
    qkv = torch.nn.functional.linear(x, qkv_w, qkv_b)
    qkv = qkv.reshape(b, n, 3, cfg.num_heads, head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]
    attn = (q * scale) @ k.transpose(-2, -1)
    rel_idx = torch.from_numpy(np.asarray(relative_position_index(cfg.window_size))).reshape(-1)
    table = torch.from_numpy(np.asarray(params["relative_position_bias_table"]))
    rel_bias = table[rel_idx].reshape(n, n, cfg.num_heads).permute(2, 0, 1)
    attn = attn + rel_bias.unsqueeze(0)
    if mask is not None:
        mask_t = torch.from_numpy(np.asarray(mask))
        num_windows = mask_t.shape[0]
        attn = attn.reshape(b // num_windows, num_windows, cfg.num_heads, n, n)
        attn = attn + mask_t.unsqueeze(1).unsqueeze(0)
        attn = attn.reshape(-1, cfg.num_heads, n, n)
    attn = torch.softmax(attn, dim=-1)
    out = (attn @ v).transpose(1, 2).reshape(b, n, c)
    return torch.nn.functional.linear(out, proj_w, proj_b)


def random_params(rng, cfg):
    c = cfg.embed_dims
    return {
        "qkv": {
            "kernel": rng.normal(size=(c, 3 * c)).astype(np.float32) * 0.02,
            "bias": rng.normal(size=(3 * c,)).astype(np.float32) * 0.02,
        },
        "proj": {
            "kernel": rng.normal(size=(c, c)).astype(np.float32) * 0.02,
            "bias": rng.normal(size=(c,)).astype(np.float32) * 0.02,
        },
        "relative_position_bias_table": rng.normal(
            size=((2 * cfg.window_size - 1) ** 2, cfg.num_heads)
        ).astype(np.float32) * 0.02,
    }


def test_window_attention_matches_torch_reference():
    rng = np.random.default_rng(1)
    cfg = WindowAttentionConfig(embed_dims=12, num_heads=3, window_size=4)
    params = random_params(rng, cfg)
    x = rng.normal(size=(5, cfg.window_size**2, cfg.embed_dims)).astype(np.float32)

    actual = np.asarray(window_attention(x, params, cfg))
    expected = torch_window_attention(torch.from_numpy(x), params, cfg).numpy()
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)


def test_shifted_window_attention_runs_fixed_shape():
    rng = np.random.default_rng(2)
    cfg = WindowAttentionConfig(embed_dims=8, num_heads=2, window_size=4)
    params = random_params(rng, cfg)
    x = rng.normal(size=(2, 6 * 6, cfg.embed_dims)).astype(np.float32)
    out = np.asarray(shifted_window_attention(x, params, cfg, (6, 6), shift_size=2))
    assert out.shape == x.shape
    assert np.isfinite(out).all()


def test_swin_block_runs_fixed_shape():
    rng = np.random.default_rng(3)
    cfg = SwinBlockConfig(embed_dims=8, num_heads=2, window_size=4, shift=True)
    params = {
        "norm1": {
            "scale": np.ones((8,), dtype=np.float32),
            "bias": np.zeros((8,), dtype=np.float32),
        },
        "attn": random_params(rng, WindowAttentionConfig(8, 2, 4)),
        "norm2": {
            "scale": np.ones((8,), dtype=np.float32),
            "bias": np.zeros((8,), dtype=np.float32),
        },
        "ffn": {
            "fc1": {
                "kernel": rng.normal(size=(8, 32)).astype(np.float32) * 0.02,
                "bias": rng.normal(size=(32,)).astype(np.float32) * 0.02,
            },
            "fc2": {
                "kernel": rng.normal(size=(32, 8)).astype(np.float32) * 0.02,
                "bias": rng.normal(size=(8,)).astype(np.float32) * 0.02,
            },
        },
    }
    x = rng.normal(size=(2, 6 * 6, 8)).astype(np.float32)
    out = np.asarray(swin_block(x, params, cfg, (6, 6)))
    assert out.shape == x.shape
    assert np.isfinite(out).all()
