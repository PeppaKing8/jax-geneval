from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from jax_geneval.deformable_attention import multi_scale_deformable_attn


def torch_ms_deform_attn(value, shapes, locations, weights):
    bs, _, num_heads, head_dim = value.shape
    value_list = value.split([int(h * w) for h, w in shapes], dim=1)
    grids = 2 * locations - 1
    sampled = []
    for level, (height, width) in enumerate(shapes):
        level_value = value_list[level].flatten(2).transpose(1, 2).reshape(
            bs * num_heads, head_dim, int(height), int(width)
        )
        level_grid = grids[:, :, :, level].transpose(1, 2).flatten(0, 1)
        sampled.append(
            F.grid_sample(
                level_value,
                level_grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )
        )
    weights = weights.transpose(1, 2).reshape(
        bs * num_heads, 1, locations.shape[1], locations.shape[3] * locations.shape[4]
    )
    out = (torch.stack(sampled, dim=-2).flatten(-2) * weights).sum(-1)
    out = out.view(bs, num_heads * head_dim, locations.shape[1])
    return out.transpose(1, 2).contiguous()


def test_multi_scale_deformable_attention_matches_torch_grid_sample():
    rng = np.random.default_rng(0)
    batch = 2
    num_heads = 3
    head_dim = 4
    num_queries = 5
    num_points = 4
    shapes = np.array([[3, 2], [2, 4]], dtype=np.int64)
    total = int(np.sum(shapes[:, 0] * shapes[:, 1]))

    value = rng.normal(size=(batch, total, num_heads, head_dim)).astype(np.float32)
    locations = rng.uniform(
        low=-0.25,
        high=1.25,
        size=(batch, num_queries, num_heads, len(shapes), num_points, 2),
    ).astype(np.float32)
    raw_weights = rng.normal(
        size=(batch, num_queries, num_heads, len(shapes), num_points)
    ).astype(np.float32)
    raw_weights = raw_weights.reshape(batch, num_queries, num_heads, -1)
    weights = np.exp(raw_weights - raw_weights.max(axis=-1, keepdims=True))
    weights = (weights / weights.sum(axis=-1, keepdims=True)).reshape(
        batch, num_queries, num_heads, len(shapes), num_points
    )

    expected = torch_ms_deform_attn(
        torch.from_numpy(value),
        torch.from_numpy(shapes),
        torch.from_numpy(locations),
        torch.from_numpy(weights),
    ).numpy()
    actual = np.asarray(
        multi_scale_deformable_attn(value, shapes, locations, weights)
    )

    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)

