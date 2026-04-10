"""Attention distribution capture for stacked cartridge evaluation.

Registers a forward hook on the last Qwen3Attention layer to capture query and
key states, then computes attention scores post-hoc.  Aggregates attention by
patient cartridge region so we can see whether the model attends to the "right"
patient's cartridge when answering a question.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F


@dataclass
class AttentionCapture:
    """Hook manager that captures Q/K from the last attention layer."""

    num_tokens_per_cartridge: int
    patient_ids: list[str]
    _query_states: Optional[torch.Tensor] = field(default=None, repr=False)
    _key_states: Optional[torch.Tensor] = field(default=None, repr=False)
    _scaling: Optional[float] = None
    _handle: Optional[object] = field(default=None, repr=False)

    @property
    def num_cartridge_tokens(self) -> int:
        return self.num_tokens_per_cartridge * len(self.patient_ids)

    def register(self, model) -> None:
        """Register a forward hook on the last decoder layer's self_attn."""
        layers = model.layers if hasattr(model, "layers") else model.model.layers
        last_attn = layers[-1].self_attn

        self._scaling = last_attn.scaling

        def hook_fn(module, args, output):
            # args[0] is the Qwen3Batch dataclass
            batch = args[0]
            hidden_states = batch.hidden_states
            input_shape = hidden_states.shape[:-1]
            hidden_shape = (*input_shape, -1, module.head_dim)

            with torch.no_grad():
                from cartridges.models.qwen.modeling_qwen3 import apply_rotary_pos_emb

                q = module.q_norm(module.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
                k = module.k_norm(module.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
                cos, sin = batch.position_embeddings
                q, k = apply_rotary_pos_emb(q, k, cos, sin)

                # Get full key states including cache
                past_kv = batch.past_key_values
                if past_kv is not None:
                    full_k, _ = past_kv.update(
                        k, k, batch.seq_ids, module.layer_idx, skip_append=True,
                    )
                else:
                    full_k = k

                # Expand KV heads to match query heads for GQA
                num_groups = q.shape[1] // full_k.shape[1]
                if num_groups > 1:
                    full_k = full_k.repeat_interleave(num_groups, dim=1)

                self._query_states = q.detach().cpu()
                self._key_states = full_k.detach().cpu()

        self._handle = last_attn.register_forward_hook(hook_fn)

    def remove(self) -> None:
        """Remove the forward hook."""
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def compute_attention_distribution(self) -> Optional[Dict[str, float]]:
        """Compute per-patient attention fractions from captured Q/K.

        Returns a dict mapping patient_id -> fraction of attention, plus
        "input" for attention to input tokens.  Fractions sum to 1.
        Returns None if no Q/K captured yet.
        """
        if self._query_states is None or self._key_states is None:
            return None

        q = self._query_states.float()   # (1, n_heads, q_len, head_dim)
        k = self._key_states.float()     # (1, n_heads, kv_len, head_dim)

        # Attention scores: (1, n_heads, q_len, kv_len)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self._scaling

        # Simple causal + cartridge mask: allow all KV positions for now
        # (the block mask in FlexAttention is more nuanced, but for analysis
        # purposes this gives a good approximation)
        attn_weights = F.softmax(scores, dim=-1)  # (1, n_heads, q_len, kv_len)

        # Average over heads and query positions to get a single distribution
        # over KV positions
        avg_weights = attn_weights.mean(dim=(0, 1, 2))  # (kv_len,)

        kv_len = avg_weights.shape[0]
        result = {}
        offset = 0

        # Cartridge regions
        for pid in self.patient_ids:
            end = offset + self.num_tokens_per_cartridge
            if end <= kv_len:
                result[pid] = avg_weights[offset:end].sum().item()
            offset = end

        # Input tokens region
        if offset < kv_len:
            result["input"] = avg_weights[offset:].sum().item()

        # Clear captured states
        self._query_states = None
        self._key_states = None

        return result

    def compute_per_query_attention(self) -> Optional[Dict[str, torch.Tensor]]:
        """Compute attention distribution per query token (not averaged).

        Returns a dict mapping patient_id -> tensor of shape (q_len,) with
        the fraction of attention from each query position to that patient's
        cartridge region.
        """
        if self._query_states is None or self._key_states is None:
            return None

        q = self._query_states.float()
        k = self._key_states.float()

        scores = torch.matmul(q, k.transpose(-2, -1)) * self._scaling
        attn_weights = F.softmax(scores, dim=-1)

        # Average over heads: (1, q_len, kv_len) -> (q_len, kv_len)
        avg_weights = attn_weights.mean(dim=1).squeeze(0)

        kv_len = avg_weights.shape[-1]
        result = {}
        offset = 0

        for pid in self.patient_ids:
            end = offset + self.num_tokens_per_cartridge
            if end <= kv_len:
                result[pid] = avg_weights[:, offset:end].sum(dim=-1)
            offset = end

        if offset < kv_len:
            result["input"] = avg_weights[:, offset:].sum(dim=-1)

        self._query_states = None
        self._key_states = None

        return result
