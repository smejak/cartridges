import abc
from dataclasses import dataclass
import itertools
import json
from pathlib import Path
from typing import Optional

from pydrantic import ObjectConfig
import torch
import torch.nn as nn

from cartridges.utils import get_logger

logger = get_logger(__name__)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input (same as in modeling_qwen3.py)."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _build_rope_cos_sin(
    positions: torch.Tensor,
    head_dim: int,
    rope_theta: float = 10000.0,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute RoPE cos/sin embeddings for the given positions.

    Args:
        positions: 1-D tensor of integer positions, shape (seq_len,).
        head_dim: dimension of each attention head.
        rope_theta: base frequency for RoPE (default 10000.0).
        dtype: output dtype.

    Returns:
        cos, sin each of shape (1, 1, seq_len, head_dim), ready to broadcast
        over (batch, n_heads, seq_len, head_dim) key tensors.
    """
    # inv_freq: (head_dim / 2,)
    inv_freq = 1.0 / (
        rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=positions.device) / head_dim)
    )
    # freqs: (seq_len, head_dim / 2)
    freqs = torch.outer(positions.float(), inv_freq)
    # emb: (seq_len, head_dim)  -- duplicate for the two halves
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos().unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, head_dim)
    sin = emb.sin().unsqueeze(0).unsqueeze(0)
    return cos.to(dtype=dtype), sin.to(dtype=dtype)


def strip_rope(keys: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Remove RoPE from key states by applying the inverse rotation.

    RoPE applies:  k_rope = k * cos + rotate_half(k) * sin

    The inverse (since RoPE is an orthogonal rotation) is:
        k = k_rope * cos + rotate_half(k_rope) * (-sin)
          = k_rope * cos - rotate_half(k_rope) * sin

    Args:
        keys: (batch, n_heads, seq_len, head_dim) with RoPE baked in.
        cos: (1, 1, seq_len, head_dim) cosine embeddings for the original positions.
        sin: (1, 1, seq_len, head_dim) sine embeddings for the original positions.

    Returns:
        keys with RoPE removed, same shape.
    """
    return keys * cos - _rotate_half(keys) * sin


def apply_rope(keys: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply RoPE to key states.

    k_rope = k * cos + rotate_half(k) * sin

    Args:
        keys: (batch, n_heads, seq_len, head_dim) without RoPE.
        cos: (1, 1, seq_len, head_dim) cosine embeddings for the target positions.
        sin: (1, 1, seq_len, head_dim) sine embeddings for the target positions.

    Returns:
        keys with RoPE applied, same shape.
    """
    return keys * cos + _rotate_half(keys) * sin

@dataclass
class AttnConfig:
    n_layers: int
    n_heads: int
    head_dim: int

CARTRIDGE_SEQ_ID = -1

class TrainableCache(nn.Module):
    """A trainable packed cache for generation with FlexAttention.
    
    The cache must do two things, which a standard Hugging Face cache does not:

    - Keep track of sequence membership of the cache and expose it to the model via
    the seq_ids method. The model will use this once per forward pass to construct 
    the appropriate block mask. 
    - Keep track of keys and values and expose them to the model in a packed manner via 
    the update method.
    
    TODO (Sabri): Ensure that tokens from the same sequence are contiguous. Eventually,
    should just page the keys and values.

    Args:
        config: The attention configuration, which we use to construct the 
        init_keys (list[torch.Tensor], optional): A `config.n_layers` length list of 
            trainable keys for the cache, should be of shape (1, n_heads, num_trainable_tokens, head_dim).
        init_values (list[torch.Tensor]): A `config.n_layers` length list of 
            trainable values for the cache, should be of shape (1, n_heads, num_trainable_tokens, head_dim).
        num_frozen_tokens (int): The number of the trainable tokens to freeze at the 
            beginning of the cache.
    """
    def __init__(
        self,        
        config: AttnConfig,
        init_keys: list[torch.Tensor]=None,
        init_values: list[torch.Tensor]=None,
        num_frozen_tokens: int = 0,
    ):
        super().__init__()
        self.config = config
        self._keys = [None] * config.n_layers  # List of tensors per layer
        self._values = [None] * config.n_layers  # List of tensors per layer
        self._num_tokens = 0

        assert (init_keys is None) == (init_values is None)
        if init_keys is None:
            self._num_trainable_tokens, self._num_frozen_tokens = 0, 0
            self.frozen_keys, self.frozen_values = None, None
            self.trainable_keys, self.trainable_values = None, None
            self._seq_ids = None
            self._init_seq_ids = None
        else:
            self._num_init_tokens = init_keys[0].shape[2]
            self._num_frozen_tokens = num_frozen_tokens
            self._num_trainable_tokens = self._num_init_tokens - num_frozen_tokens
            assert len(init_keys) == config.n_layers == len(init_values)
            
            # we initialize the seq ids for the first 
            # `num_trainable_tokens + num_frozen_tokens` tokens to -1, which means that 
            # the tokens are part of the cartridge and should be attended to by 
            # all tokens.
            _seq_ids =torch.full(
                (self._num_init_tokens,),
                fill_value=CARTRIDGE_SEQ_ID, 
                dtype=torch.long,
            )
            self.register_buffer("_init_seq_ids", _seq_ids)
            self.register_buffer("_seq_ids", _seq_ids)  # .to moves the tensor to the correct device

            for vec in itertools.chain(init_keys, init_values):
                assert vec.shape == (1, config.n_heads, self._num_init_tokens, config.head_dim)

            self.frozen_keys = nn.ParameterList(
                [
                    nn.Parameter(keys_vec[:, :, :num_frozen_tokens].contiguous())
                    for keys_vec in init_keys
                ]
                if num_frozen_tokens
                else []
            )
            self.frozen_values = nn.ParameterList(
                [
                    nn.Parameter(values_vec[:, :, :num_frozen_tokens].contiguous())
                    for values_vec in init_values
                ]
                if num_frozen_tokens
                else []
            )

            for param in itertools.chain(self.frozen_keys, self.frozen_values):
                param.requires_grad = False

            self.trainable_keys = nn.ParameterList(
                [
                    nn.Parameter(keys_vec[:, :, num_frozen_tokens:].contiguous())
                    for keys_vec in init_keys
                ]
            )
            self.trainable_values = nn.ParameterList(
                [
                    nn.Parameter(values_vec[:, :, num_frozen_tokens:].contiguous())
                    for values_vec in init_values
                ]
            )
            logger.info(f"num_trainable_tokens: {self._num_trainable_tokens}")
            logger.info(f"num_frozen_tokens: {self._num_frozen_tokens}")
                
    def update(
        self, 
        new_keys: torch.Tensor,
        new_values: torch.Tensor,
        new_seq_ids: torch.Tensor,
        layer_idx: int,
        skip_append: bool = False,
    ):
        """Update the cache with new keys and values while maintaining sequence contiguity.
        
        Args:
            new_keys: (1, num_heads, seq_len, head_dim) tensor of new keys
            new_values: (1, num_heads, seq_len, head_dim) tensor of new values  
            new_seq_ids: (seq_len,) tensor of sequence ids for the new tokens
            layer_idx: index of the layer in the model.
            skip_append: if True, do not append the new keys and values to the cache, 
                just return the concatenation of the new_keys and values. 
        """
        assert new_seq_ids.shape[0] == new_keys.shape[2]
        assert new_seq_ids.shape[0] == new_values.shape[2]

        if layer_idx == 0 and not skip_append:
            # we assume the same seq ids at every layer. This allows us to create
            # a single block mask for the entire model. 
            if self._seq_ids is None:
                self._seq_ids = new_seq_ids
            else:
                self._seq_ids = torch.cat([self._seq_ids, new_seq_ids], dim=0)
            self._num_tokens += new_keys.shape[2]
        
        keys = [new_keys]
        values = [new_values]

        if self._keys[layer_idx] is not None:
            # Concatenate along sequence dimension while maintaining contiguous sequences
            keys = [self._keys[layer_idx]] + keys
            values = [self._values[layer_idx]] + values

        if not skip_append:
            self._keys[layer_idx] = torch.cat(keys, dim=2)
            self._values[layer_idx] = torch.cat(values, dim=2)
        
        if self._num_trainable_tokens > 0:
            keys = [self.trainable_keys[layer_idx]] + keys
            values = [self.trainable_values[layer_idx]] + values
        
        if self._num_frozen_tokens > 0:
            keys = [self.frozen_keys[layer_idx]] + keys
            values = [self.frozen_values[layer_idx]] + values
        
        if self._num_trainable_tokens == 0 and self._num_frozen_tokens == 0:
            return self._keys[layer_idx], self._values[layer_idx]

        return torch.cat(keys, dim=2), torch.cat(values, dim=2)
    
    def num_tokens(self) -> int:
        """Get the sequence length of the cache."""
        return self._num_frozen_tokens + self._num_trainable_tokens + self._num_tokens
    
    def num_cartridge_tokens(self) -> int:
        """Get the number of tokens in the cartridge."""
        return self._num_frozen_tokens + self._num_trainable_tokens
    
    def seq_ids(self) -> torch.Tensor:
        """Returns the sequence ids of the cache."""
        return self._seq_ids
       
    def clear(self):
        self._keys = [None] * self.config.n_layers
        self._values = [None] * self.config.n_layers
        self._num_tokens = 0
        self._seq_ids = self._init_seq_ids

    def save(self, path: str):
        """Saves the trainable keys and values to the specified path."""
        torch.save(
            {
                "trainable_keys": self.trainable_keys,
                "trainable_values": self.trainable_values,
                "frozen_keys": self.frozen_keys,
                "frozen_values": self.frozen_values,
            },
            path,
        )

    @classmethod
    def stack_caches(cls, caches: list["TrainableCache"]) -> "TrainableCache":
        """Stack multiple trained caches by concatenating their KV tensors.

        All caches must share the same AttnConfig (n_layers, n_heads, head_dim).
        The stacked cache preserves the [frozen | trainable] layout: all frozen
        tokens from all caches come first, then all trainable tokens.
        All tokens retain CARTRIDGE_SEQ_ID = -1.

        Args:
            caches: List of TrainableCache objects to stack.

        Returns:
            A new TrainableCache with concatenated KV tensors.
        """
        assert len(caches) > 0, "Need at least one cache to stack"
        config = caches[0].config
        for c in caches:
            assert c.config == config, (
                f"All caches must share the same AttnConfig, got {c.config} vs {config}"
            )

        total_frozen = sum(c._num_frozen_tokens for c in caches)
        has_frozen = any(c._num_frozen_tokens > 0 for c in caches)

        init_keys = []
        init_values = []
        for layer_idx in range(config.n_layers):
            layer_keys = []
            layer_values = []
            # Frozen tokens first (from all caches)
            if has_frozen:
                for c in caches:
                    if c._num_frozen_tokens > 0:
                        layer_keys.append(c.frozen_keys[layer_idx].data)
                        layer_values.append(c.frozen_values[layer_idx].data)
            # Then trainable tokens (from all caches)
            for c in caches:
                if c._num_trainable_tokens > 0:
                    layer_keys.append(c.trainable_keys[layer_idx].data)
                    layer_values.append(c.trainable_values[layer_idx].data)

            init_keys.append(torch.cat(layer_keys, dim=2))
            init_values.append(torch.cat(layer_values, dim=2))

        return cls(
            config=config,
            init_keys=init_keys,
            init_values=init_values,
            num_frozen_tokens=total_frozen,
        )

    @classmethod
    def stack_caches_rope_adjusted(
        cls,
        caches: list["TrainableCache"],
        rope_theta: float = 10000.0,
    ) -> "TrainableCache":
        """Stack multiple caches with RoPE position re-indexing.

        Each cache's key states have RoPE baked in for positions 0..num_tokens-1.
        When naively stacking, all caches share the same positions, which creates
        conflicts.  This method:

        1. Strips the original RoPE from each cache's key states (positions 0..L-1).
        2. Re-applies RoPE with sequential positions across all stacked caches:
           cache 0 -> 0..L0-1, cache 1 -> L0..L0+L1-1, etc.

        Values are not affected by RoPE and are simply concatenated.

        Args:
            caches: list of TrainableCache objects to stack.
            rope_theta: base frequency for RoPE (must match the model's
                config.rope_theta, default 10000.0 matches Qwen3's default).

        Returns:
            A new TrainableCache with re-indexed RoPE in key states.
        """
        assert len(caches) > 0, "Need at least one cache to stack"
        config = caches[0].config
        for c in caches:
            assert c.config == config, (
                f"All caches must share the same AttnConfig, got {c.config} vs {config}"
            )

        head_dim = config.head_dim
        total_frozen = sum(c._num_frozen_tokens for c in caches)
        has_frozen = any(c._num_frozen_tokens > 0 for c in caches)

        # Build the per-cache token counts in stacking order:
        # [frozen_0, frozen_1, ..., trainable_0, trainable_1, ...]
        # We need to know the size of each segment to compute positions.
        frozen_lengths = []
        trainable_lengths = []
        if has_frozen:
            for c in caches:
                if c._num_frozen_tokens > 0:
                    frozen_lengths.append(c._num_frozen_tokens)
        for c in caches:
            if c._num_trainable_tokens > 0:
                trainable_lengths.append(c._num_trainable_tokens)

        # Compute sequential position assignments for each segment.
        # Position counter runs across the full stacked sequence.
        frozen_positions = []  # list of (start_pos, length) for each frozen segment
        trainable_positions = []  # list of (start_pos, length) for each trainable segment
        pos = 0
        if has_frozen:
            for length in frozen_lengths:
                frozen_positions.append((pos, length))
                pos += length
        for length in trainable_lengths:
            trainable_positions.append((pos, length))
            pos += length

        # Also track original positions for stripping: each cache's frozen/trainable
        # tokens were at positions 0..total_tokens-1 during initialization.
        # frozen = 0..num_frozen-1, trainable = num_frozen..num_frozen+num_trainable-1

        init_keys = []
        init_values = []
        for layer_idx in range(config.n_layers):
            layer_keys = []
            layer_values = []

            frozen_seg_idx = 0
            trainable_seg_idx = 0

            # --- Frozen tokens first ---
            if has_frozen:
                for c in caches:
                    if c._num_frozen_tokens > 0:
                        k = c.frozen_keys[layer_idx].data  # (1, n_heads, n_frozen, head_dim)
                        v = c.frozen_values[layer_idx].data
                        n_frozen = c._num_frozen_tokens
                        compute_dtype = torch.float32

                        # Original positions for this cache's frozen tokens: 0..n_frozen-1
                        orig_positions = torch.arange(n_frozen, device=k.device)
                        old_cos, old_sin = _build_rope_cos_sin(
                            orig_positions, head_dim, rope_theta, dtype=compute_dtype
                        )

                        # New sequential positions
                        new_start, seg_len = frozen_positions[frozen_seg_idx]
                        new_positions = torch.arange(new_start, new_start + seg_len, device=k.device)
                        new_cos, new_sin = _build_rope_cos_sin(
                            new_positions, head_dim, rope_theta, dtype=compute_dtype
                        )

                        # Strip old RoPE, apply new
                        k_f32 = k.to(compute_dtype)
                        k_bare = strip_rope(k_f32, old_cos, old_sin)
                        k_new = apply_rope(k_bare, new_cos, new_sin)

                        layer_keys.append(k_new.to(k.dtype))
                        layer_values.append(v)
                        frozen_seg_idx += 1

            # --- Trainable tokens ---
            for c in caches:
                if c._num_trainable_tokens > 0:
                    k = c.trainable_keys[layer_idx].data  # (1, n_heads, n_train, head_dim)
                    v = c.trainable_values[layer_idx].data
                    n_train = c._num_trainable_tokens
                    n_frozen_in_cache = c._num_frozen_tokens
                    compute_dtype = torch.float32

                    # Original positions for trainable tokens: n_frozen..n_frozen+n_train-1
                    orig_positions = torch.arange(
                        n_frozen_in_cache, n_frozen_in_cache + n_train, device=k.device
                    )
                    old_cos, old_sin = _build_rope_cos_sin(
                        orig_positions, head_dim, rope_theta, dtype=compute_dtype
                    )

                    # New sequential positions
                    new_start, seg_len = trainable_positions[trainable_seg_idx]
                    new_positions = torch.arange(new_start, new_start + seg_len, device=k.device)
                    new_cos, new_sin = _build_rope_cos_sin(
                        new_positions, head_dim, rope_theta, dtype=compute_dtype
                    )

                    # Strip old RoPE, apply new
                    k_f32 = k.to(compute_dtype)
                    k_bare = strip_rope(k_f32, old_cos, old_sin)
                    k_new = apply_rope(k_bare, new_cos, new_sin)

                    layer_keys.append(k_new.to(k.dtype))
                    layer_values.append(v)
                    trainable_seg_idx += 1

            init_keys.append(torch.cat(layer_keys, dim=2))
            init_values.append(torch.cat(layer_values, dim=2))

        return cls(
            config=config,
            init_keys=init_keys,
            init_values=init_values,
            num_frozen_tokens=total_frozen,
        )

    @classmethod
    def from_pretrained(cls, path: str, device: Optional[str] = None):
        if not isinstance(path, str):
            raise TypeError(f"path must be a string, got {type(path)}")
        print(path)
        checkpoint = torch.load(path, map_location=device, weights_only=False)

        # Ensure necessary keys are in the checkpoint
        for key in ["trainable_keys", "trainable_values", "frozen_keys", "frozen_values"]:
            if key not in checkpoint:
                raise KeyError(f"Key '{key}' not found in checkpoint")

        n_layers = len(checkpoint["trainable_keys"])
        n_heads = checkpoint["trainable_keys"][0].size(1)
        num_tokens = checkpoint["trainable_keys"][0].size(2)
        head_dim = checkpoint["trainable_keys"][0].size(3)

        if len(checkpoint["frozen_keys"]) != n_layers:
            raise AssertionError(
                "Mismatch in number of layers between trainable and fixed keys"
            )
        if checkpoint["frozen_keys"]:
            if (
                checkpoint["frozen_keys"][0].size(1) != n_heads
                or checkpoint["frozen_keys"][0].size(3) != head_dim
            ):
                raise AssertionError(
                    "Mismatch in head configuration between trainable and fixed keys"
                )

        config = AttnConfig(n_layers=n_layers, n_heads=n_heads, head_dim=head_dim)
        # Here, num_tokens is inferred from trainable keys, but note that the total tokens may be different if fixed tokens exist.
        # The number of fixed tokens can be inferred from frozen_keys if available.
        num_frozen_tokens = (
            checkpoint["frozen_keys"][0].size(1) if checkpoint["frozen_keys"] else 0
        )

        return cls(
            config=config,
            init_keys=[
                (
                    torch.cat([fixed, trainable], dim=2).contiguous()
                    if num_frozen_tokens > 0
                    else trainable
                )
                for fixed, trainable in zip(
                    checkpoint["frozen_keys"], checkpoint["trainable_keys"]
                )
            ],
            init_values=[
                (
                    torch.cat([fixed, trainable], dim=2).contiguous()
                    if num_frozen_tokens > 0
                    else trainable
                )
                for fixed, trainable in zip(
                    checkpoint["frozen_values"], checkpoint["trainable_values"]
                )
            ],
            num_frozen_tokens=num_frozen_tokens,
        )


class KVCacheFactory(abc.ABC):
    class Config(ObjectConfig):
        _pass_as_config = True

        # SE (03/26): we freeze the first token to prevent forgetting
        num_frozen_tokens: int = 1

    def __init__(self, config: Config):
        self.config = config

    @abc.abstractmethod
    def initialize_kv_cache(
        self, tokenizer, model, attn_config: AttnConfig 
    ) -> TrainableCache:
        raise NotImplementedError()


class KVCacheFactoryWithStateSaving(abc.ABC):
    class Config(KVCacheFactory.Config):
        directory: str
        is_wandb: bool
        force_recreate: bool = False

    def __init__(self, config: Config):
        self.config = config

    @abc.abstractmethod
    def initalize_kv_cache_impl(
        self,
        tokenizer,
        model,
        attn_config: AttnConfig,
    ) -> tuple[TrainableCache, dict]:
        raise NotImplementedError()

    @property
    def local_kv_cache_path(self) -> Path:
        # TODO: better file extension
        return Path(self.config.directory) / "kv_cache.torch"

    @property
    def local_metadata_path(self) -> Path:
        # TODO: better file extension
        return Path(self.config.directory) / "metadata.json"

    def maybe_load_cached(self) -> Optional[TrainableCache]:
        if self.config.force_recreate:
            return

        if not self.config.is_wandb:
            if self.local_kv_cache_path.exists():
                logger.info(
                    f"State Saving KV initializer: loading KV cache from: {self.local_kv_cache_path}"
                )
                return TrainableCache.from_pretrained(
                    str(self.local_kv_cache_path.absolute()),
                )

            return

        raise NotImplementedError("Need to add saving to wanb")

    def initalize_kv_cache(
        self, tokenizer, model, attn_config: AttnConfig
    ) -> TrainableCache:
        maybe_cache = self.maybe_load_cached()
        if maybe_cache is not None:
            assert (
                maybe_cache._num_trainable_tokens + maybe_cache._num_frozen_tokens
                == self.config.num_tokens
            )
            assert maybe_cache.config == attn_config
            return maybe_cache

        cache, metadata = self.initalize_kv_cache_impl(
            tokenizer, model, attn_config
        )

        Path(self.config.directory).mkdir(parents=True, exist_ok=True)

        with open(self.local_metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        cache.save(str(self.local_kv_cache_path.absolute()))
        logger.info(
            f"State Saving KV initializer: saving KV cache to: {self.local_kv_cache_path}"
        )

        return cache
