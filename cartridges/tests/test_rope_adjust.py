"""Tests for RoPE-adjusted cache stacking.

Verifies that:
1. RoPE round-trip (apply then strip) recovers the original keys.
2. stack_caches_rope_adjusted produces correct sequential positions.
3. Single-cache adjustment is identity (positions unchanged).
"""
import pytest
import torch

from cartridges.cache import (
    AttnConfig,
    TrainableCache,
    _build_rope_cos_sin,
    apply_rope,
    strip_rope,
)

ROPE_THETA = 10000.0


def _make_dummy_cache(
    n_layers: int = 2,
    n_heads: int = 4,
    num_tokens: int = 8,
    head_dim: int = 32,
    num_frozen: int = 1,
) -> TrainableCache:
    """Create a TrainableCache with random keys that have RoPE baked in."""
    config = AttnConfig(n_layers=n_layers, n_heads=n_heads, head_dim=head_dim)

    positions = torch.arange(num_tokens)
    cos, sin = _build_rope_cos_sin(positions, head_dim, ROPE_THETA)

    init_keys = []
    init_values = []
    for _ in range(n_layers):
        # Start with random "bare" keys, then apply RoPE to simulate
        # what KVFromText produces.
        bare_k = torch.randn(1, n_heads, num_tokens, head_dim)
        k_with_rope = apply_rope(bare_k, cos, sin)
        v = torch.randn(1, n_heads, num_tokens, head_dim)
        init_keys.append(k_with_rope)
        init_values.append(v)

    return TrainableCache(
        config=config,
        init_keys=init_keys,
        init_values=init_values,
        num_frozen_tokens=num_frozen,
    )


class TestRopeRoundTrip:
    """Verify that apply_rope then strip_rope recovers the original tensor."""

    @pytest.mark.parametrize("head_dim", [32, 64, 128])
    @pytest.mark.parametrize("seq_len", [1, 8, 512])
    def test_roundtrip_exact(self, head_dim, seq_len):
        positions = torch.arange(seq_len)
        cos, sin = _build_rope_cos_sin(positions, head_dim, ROPE_THETA)

        bare = torch.randn(1, 4, seq_len, head_dim)
        rotated = apply_rope(bare, cos, sin)
        recovered = strip_rope(rotated, cos, sin)

        torch.testing.assert_close(recovered, bare, atol=1e-5, rtol=1e-5)

    def test_roundtrip_nonzero_start(self):
        """Round-trip with positions starting at an offset (e.g. 512..1023)."""
        head_dim = 64
        seq_len = 16
        offset = 512
        positions = torch.arange(offset, offset + seq_len)
        cos, sin = _build_rope_cos_sin(positions, head_dim, ROPE_THETA)

        bare = torch.randn(1, 4, seq_len, head_dim)
        rotated = apply_rope(bare, cos, sin)
        recovered = strip_rope(rotated, cos, sin)

        torch.testing.assert_close(recovered, bare, atol=1e-5, rtol=1e-5)

    def test_strip_then_reapply_different_positions(self):
        """Strip RoPE at old positions, re-apply at new positions, verify result
        matches directly applying at new positions."""
        head_dim = 64
        seq_len = 16

        old_positions = torch.arange(seq_len)
        new_positions = torch.arange(100, 100 + seq_len)

        old_cos, old_sin = _build_rope_cos_sin(old_positions, head_dim, ROPE_THETA)
        new_cos, new_sin = _build_rope_cos_sin(new_positions, head_dim, ROPE_THETA)

        bare = torch.randn(1, 4, seq_len, head_dim)
        with_old_rope = apply_rope(bare, old_cos, old_sin)

        # Strip old, apply new
        stripped = strip_rope(with_old_rope, old_cos, old_sin)
        with_new_rope = apply_rope(stripped, new_cos, new_sin)

        # Direct application of new positions to bare keys
        expected = apply_rope(bare, new_cos, new_sin)

        torch.testing.assert_close(with_new_rope, expected, atol=1e-5, rtol=1e-5)


class TestStackCachesRopeAdjusted:
    """Test the full stack_caches_rope_adjusted classmethod."""

    def test_single_cache_is_identity(self):
        """Stacking a single cache with RoPE adjustment should produce the same
        keys as the original (positions 0..N-1 are unchanged)."""
        cache = _make_dummy_cache(n_layers=2, num_tokens=8, num_frozen=1)
        stacked = TrainableCache.stack_caches_rope_adjusted([cache], rope_theta=ROPE_THETA)

        for layer_idx in range(cache.config.n_layers):
            # Frozen keys
            if cache._num_frozen_tokens > 0:
                torch.testing.assert_close(
                    stacked.frozen_keys[layer_idx].data,
                    cache.frozen_keys[layer_idx].data,
                    atol=1e-4, rtol=1e-4,
                )
            # Trainable keys
            torch.testing.assert_close(
                stacked.trainable_keys[layer_idx].data,
                cache.trainable_keys[layer_idx].data,
                atol=1e-4, rtol=1e-4,
            )

    def test_two_cache_sequential_positions(self):
        """Stacking two caches: verify the second cache's keys get shifted positions."""
        n_layers, n_heads, head_dim = 1, 2, 32
        num_tokens = 4
        num_frozen = 0  # no frozen tokens to simplify

        config = AttnConfig(n_layers=n_layers, n_heads=n_heads, head_dim=head_dim)

        # Create two caches from known bare keys
        positions_0 = torch.arange(num_tokens)
        cos_0, sin_0 = _build_rope_cos_sin(positions_0, head_dim, ROPE_THETA)

        bare_k1 = torch.randn(1, n_heads, num_tokens, head_dim)
        bare_k2 = torch.randn(1, n_heads, num_tokens, head_dim)

        cache1 = TrainableCache(
            config=config,
            init_keys=[apply_rope(bare_k1, cos_0, sin_0)],
            init_values=[torch.randn(1, n_heads, num_tokens, head_dim)],
            num_frozen_tokens=0,
        )
        cache2 = TrainableCache(
            config=config,
            init_keys=[apply_rope(bare_k2, cos_0, sin_0)],
            init_values=[torch.randn(1, n_heads, num_tokens, head_dim)],
            num_frozen_tokens=0,
        )

        stacked = TrainableCache.stack_caches_rope_adjusted(
            [cache1, cache2], rope_theta=ROPE_THETA
        )

        # Expected: cache1 keys get positions 0..3, cache2 keys get positions 4..7
        expected_pos_1 = torch.arange(0, num_tokens)
        expected_pos_2 = torch.arange(num_tokens, 2 * num_tokens)

        cos_1, sin_1 = _build_rope_cos_sin(expected_pos_1, head_dim, ROPE_THETA)
        cos_2, sin_2 = _build_rope_cos_sin(expected_pos_2, head_dim, ROPE_THETA)

        expected_k1 = apply_rope(bare_k1, cos_1, sin_1)
        expected_k2 = apply_rope(bare_k2, cos_2, sin_2)
        expected_all = torch.cat([expected_k1, expected_k2], dim=2)

        actual_all = stacked.trainable_keys[0].data
        torch.testing.assert_close(actual_all, expected_all, atol=1e-4, rtol=1e-4)

    def test_stacked_token_counts(self):
        """Verify that the stacked cache has the correct number of tokens."""
        c1 = _make_dummy_cache(num_tokens=8, num_frozen=2)
        c2 = _make_dummy_cache(num_tokens=8, num_frozen=2)
        c3 = _make_dummy_cache(num_tokens=8, num_frozen=2)

        stacked = TrainableCache.stack_caches_rope_adjusted(
            [c1, c2, c3], rope_theta=ROPE_THETA
        )
        assert stacked._num_frozen_tokens == 6  # 3 * 2
        assert stacked._num_trainable_tokens == 18  # 3 * 6
        assert stacked.num_cartridge_tokens() == 24  # 3 * 8

    def test_values_unchanged(self):
        """Values should be concatenated without modification."""
        c1 = _make_dummy_cache(n_layers=1, num_tokens=4, num_frozen=0)
        c2 = _make_dummy_cache(n_layers=1, num_tokens=4, num_frozen=0)

        stacked_naive = TrainableCache.stack_caches([c1, c2])
        stacked_rope = TrainableCache.stack_caches_rope_adjusted(
            [c1, c2], rope_theta=ROPE_THETA
        )

        # Values should be identical between naive and rope-adjusted stacking
        for layer_idx in range(1):
            torch.testing.assert_close(
                stacked_rope.trainable_values[layer_idx].data,
                stacked_naive.trainable_values[layer_idx].data,
            )
