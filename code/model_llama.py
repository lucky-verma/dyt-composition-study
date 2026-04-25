"""
Llama-style Transformer with toggle flags for compositional modifications.
Based on karpathy/llama2.c (MIT license), adapted to match our composition
study interface (same forward signature, configure_optimizers, etc.).

Architecture: RoPE, SwiGLU FFN, RMSNorm (default), GQA, bias=False throughout.
Modifications ported from model.py:
  - DyT (Dynamic Tanh): replaces RMSNorm with tanh(alpha*x)*weight+bias
  - DiffAttn (Differential Attention): dual-softmax attention that cancels noise

References:
  - llama2.c: https://github.com/karpathy/llama2.c
  - DyT: Zhu et al., "Transformers without Normalization" (CVPR 2025)
  - DiffAttn: Ye et al., "Differential Transformer" (ICLR 2025)
"""

import math
import inspect
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.nn import functional as F


# ============================================================
# Normalization variants
# ============================================================

class RMSNorm(nn.Module):
    """RMSNorm: default normalization for Llama-style models."""
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class DynamicTanh(nn.Module):
    """DyT: replace RMSNorm with tanh(alpha*x)*weight+bias.
    From Zhu et al., "Transformers without Normalization" (CVPR 2025).
    """
    def __init__(self, dim: int, alpha_init: float = 2.0, eps: float = 1e-5):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1) * alpha_init)
        self.weight = nn.Parameter(torch.ones(dim))
        # No bias in Llama-style models (bias=False throughout)

    def forward(self, x):
        return torch.tanh(self.alpha * x) * self.weight


def make_norm(dim: int, config):
    """Factory: select normalization based on config."""
    if config.use_dyt:
        return DynamicTanh(dim, alpha_init=config.dyt_alpha_init, eps=config.norm_eps)
    else:
        return RMSNorm(dim, eps=config.norm_eps)


# ============================================================
# RoPE (Rotary Positional Embeddings)
# ============================================================

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """Precompute cos/sin frequencies for RoPE."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cos = torch.cos(freqs)  # (end, dim/2)
    freqs_sin = torch.sin(freqs)  # (end, dim/2)
    return freqs_cos, freqs_sin


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    """Reshape freq tensor for broadcasting against (B, T, nh, hd/2)."""
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cos: torch.Tensor,
    freqs_sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE to query and key tensors."""
    # reshape to complex pairs: (..., hd) -> (..., hd/2, 2)
    xq_r, xq_i = xq.float().reshape(xq.shape[:-1] + (-1, 2)).unbind(-1)
    xk_r, xk_i = xk.float().reshape(xk.shape[:-1] + (-1, 2)).unbind(-1)

    # broadcast freqs
    freqs_cos = reshape_for_broadcast(freqs_cos, xq_r)
    freqs_sin = reshape_for_broadcast(freqs_sin, xq_r)

    # rotation
    xq_out_r = xq_r * freqs_cos - xq_i * freqs_sin
    xq_out_i = xq_r * freqs_sin + xq_i * freqs_cos
    xk_out_r = xk_r * freqs_cos - xk_i * freqs_sin
    xk_out_i = xk_r * freqs_sin + xk_i * freqs_cos

    # flatten back
    xq_out = torch.stack([xq_out_r, xq_out_i], dim=-1).flatten(3)
    xk_out = torch.stack([xk_out_r, xk_out_i], dim=-1).flatten(3)

    return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Expand KV heads for GQA: (B, T, n_kv, hd) -> (B, T, n_heads, hd)."""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


# ============================================================
# Attention variants
# ============================================================

class Attention(nn.Module):
    """Multi-head attention with GQA and RoPE (Llama-style)."""

    def __init__(self, config, layer_idx: int = 0):
        super().__init__()
        self.config = config
        self.n_heads = config.n_heads
        # R5 ablate_gqa: force MHA (n_kv = n_heads) regardless of config
        self.n_kv_heads = config.n_heads if (config.n_kv_heads is None or config.ablate_gqa) else config.n_kv_heads
        assert config.n_heads % self.n_kv_heads == 0
        self.n_rep = self.n_heads // self.n_kv_heads
        self.head_dim = config.dim // config.n_heads

        self.wq = nn.Linear(config.dim, config.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(config.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(config.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(config.n_heads * self.head_dim, config.dim, bias=False)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.dropout = config.dropout

    def forward(self, x, freqs_cos, freqs_sin):
        B, T, _ = x.shape

        xq = self.wq(x).view(B, T, self.n_heads, self.head_dim)
        xk = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim)
        xv = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim)

        # RoPE — R5 ablate_rope: skip rotation (model becomes positionally blind)
        if not self.config.ablate_rope:
            xq, xk = apply_rotary_emb(xq, xk, freqs_cos, freqs_sin)

        # GQA: expand KV heads
        xk = repeat_kv(xk, self.n_rep)
        xv = repeat_kv(xv, self.n_rep)

        # (B, nh, T, hd)
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        # Flash attention
        y = F.scaled_dot_product_attention(
            xq, xk, xv, attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )

        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        y = self.resid_dropout(self.wo(y))
        return y


class DifferentialAttention(nn.Module):
    """Differential Attention with GQA and RoPE for Llama-style models.
    attn = softmax(Q1K1) - lambda * softmax(Q2K2).
    From Ye et al., "Differential Transformer" (ICLR 2025).

    Q/K projections are doubled to produce two sub-queries/sub-keys per head.
    V projection and GQA are unchanged. RoPE is applied to both sub-pairs.
    """

    def __init__(self, config, layer_idx: int = 0):
        super().__init__()
        self.config = config
        self.n_heads = config.n_heads
        # R5 ablate_gqa: force MHA regardless of config
        self.n_kv_heads = config.n_heads if (config.n_kv_heads is None or config.ablate_gqa) else config.n_kv_heads
        assert config.n_heads % self.n_kv_heads == 0
        self.n_rep = self.n_heads // self.n_kv_heads
        self.head_dim = config.dim // config.n_heads
        self.dropout = config.dropout

        # 2x Q/K for differential (two sub-attention maps per head)
        self.wq = nn.Linear(config.dim, 2 * config.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(config.dim, 2 * self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(config.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(config.n_heads * self.head_dim, config.dim, bias=False)

        self.resid_dropout = nn.Dropout(config.dropout)

        # Learnable lambda parameters
        self.lambda_q1 = nn.Parameter(torch.randn(self.head_dim) * 0.1)
        self.lambda_k1 = nn.Parameter(torch.randn(self.head_dim) * 0.1)
        self.lambda_q2 = nn.Parameter(torch.randn(self.head_dim) * 0.1)
        self.lambda_k2 = nn.Parameter(torch.randn(self.head_dim) * 0.1)
        self.lambda_init = 0.8 - 0.6 * math.exp(-0.3 * (layer_idx + 1))

        # GroupNorm for stabilizing differential attention
        dim = config.dim
        self.subln = nn.GroupNorm(
            num_groups=self.n_heads, num_channels=dim, affine=False,
        )

    def forward(self, x, freqs_cos, freqs_sin):
        B, T, C = x.shape
        nh = self.n_heads
        n_kv = self.n_kv_heads
        hd = self.head_dim

        # Project Q/K with 2x heads, V with standard heads
        q = self.wq(x).view(B, T, nh, 2, hd)       # (B, T, nh, 2, hd)
        k = self.wk(x).view(B, T, n_kv, 2, hd)     # (B, T, n_kv, 2, hd)
        v = self.wv(x).view(B, T, n_kv, hd)         # (B, T, n_kv, hd)

        q1, q2 = q[:, :, :, 0, :], q[:, :, :, 1, :]  # (B, T, nh, hd)
        k1, k2 = k[:, :, :, 0, :], k[:, :, :, 1, :]  # (B, T, n_kv, hd)

        # Apply RoPE to both sub-pairs — R5 ablate_rope: skip for both paths
        if not self.config.ablate_rope:
            q1, k1 = apply_rotary_emb(q1, k1, freqs_cos, freqs_sin)
            q2, k2 = apply_rotary_emb(q2, k2, freqs_cos, freqs_sin)

        # GQA: expand KV heads
        k1 = repeat_kv(k1, self.n_rep)
        k2 = repeat_kv(k2, self.n_rep)
        v = repeat_kv(v, self.n_rep)

        # Transpose to (B, nh, T, hd)
        q1, q2 = q1.transpose(1, 2), q2.transpose(1, 2)
        k1, k2 = k1.transpose(1, 2), k2.transpose(1, 2)
        v = v.transpose(1, 2)

        scale = 1.0 / math.sqrt(hd)

        # Two attention maps with causal masking
        attn1 = (q1 @ k1.transpose(-2, -1)) * scale
        attn2 = (q2 @ k2.transpose(-2, -1)) * scale
        causal_mask = torch.triu(
            torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1,
        )
        attn1 = attn1.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        attn2 = attn2.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        attn1 = F.softmax(attn1, dim=-1)
        attn2 = F.softmax(attn2, dim=-1)

        # Compute lambda
        lambda_val = (
            torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1))
            - torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2))
            + self.lambda_init
        )

        # Differential attention
        attn = attn1 - lambda_val * attn2
        y = attn @ v  # (B, nh, T, hd)

        # GroupNorm + scale
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.subln(y.transpose(1, 2)).transpose(1, 2)
        y = y * (1.0 - self.lambda_init)

        y = self.resid_dropout(self.wo(y))
        return y


def make_attn(config, layer_idx: int = 0):
    """Factory: select attention mechanism based on config."""
    if config.use_diff_attn:
        return DifferentialAttention(config, layer_idx=layer_idx)
    else:
        return Attention(config, layer_idx=layer_idx)


# ============================================================
# SwiGLU FFN
# ============================================================

class FeedForward(nn.Module):
    """SwiGLU FFN (default) or non-gated GELU if config.ablate_swiglu (R5)."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        dim = config.dim
        hidden_dim = config.hidden_dim
        if hidden_dim is None:
            hidden_dim = 4 * dim
            hidden_dim = int(2 * hidden_dim / 3)
            hidden_dim = config.multiple_of * (
                (hidden_dim + config.multiple_of - 1) // config.multiple_of
            )
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)  # gate projection
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)   # down projection
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)   # up projection
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        if self.config.ablate_swiglu:
            # R5: GPT-2-style non-gated GELU (w3 unused)
            return self.dropout(self.w2(F.gelu(self.w1(x))))
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


# ============================================================
# Block and Model
# ============================================================

class TransformerBlock(nn.Module):
    """Llama-style transformer block with optional DyT/DiffAttn."""

    def __init__(self, config, layer_idx: int = 0):
        super().__init__()
        self.attention_norm = make_norm(config.dim, config)
        self.attn = make_attn(config, layer_idx=layer_idx)
        self.ffn_norm = make_norm(config.dim, config)
        self.ffn = FeedForward(config)

    def forward(self, x, freqs_cos, freqs_sin):
        h = x + self.attn(self.attention_norm(x), freqs_cos, freqs_sin)
        out = h + self.ffn(self.ffn_norm(h))
        return out


@dataclass
class LlamaConfig:
    # Model architecture
    dim: int = 512
    n_layers: int = 12
    n_heads: int = 8
    n_kv_heads: Optional[int] = None     # None = MHA, else GQA
    vocab_size: int = 50304              # GPT-2 tokenizer, padded for efficiency
    hidden_dim: Optional[int] = None     # None = auto SwiGLU sizing
    multiple_of: int = 256               # SwiGLU hidden dim alignment
    norm_eps: float = 1e-5
    max_seq_len: int = 512
    dropout: float = 0.0
    # === Modification toggles (same interface as GPT-2 model) ===
    use_dyt: bool = False                # DyT: replace RMSNorm with Dynamic Tanh
    dyt_alpha_init: float = 2.0          # DyT alpha initialization value
    use_diff_attn: bool = False          # DiffAttn: differential attention
    # === R5 ablation toggles (2026-04-19) — isolate which Llama component causes DyT failure ===
    ablate_rope: bool = False            # skip RoPE rotation (tests positional encoding role)
    ablate_swiglu: bool = False          # FFN becomes non-gated GELU (GPT-2 style) not SwiGLU
    ablate_gqa: bool = False             # force n_kv_heads = n_heads (MHA not GQA)


class Llama(nn.Module):
    """Llama-style Transformer for the composition study.

    Interface-compatible with GPT class in model.py:
      - forward(idx, targets=None) -> (logits, loss)
      - get_num_params(non_embedding=True)
      - configure_optimizers(weight_decay, lr, betas, device_type)
      - estimate_mfu(fwdbwd_per_iter, dt)
      - generate(idx, max_new_tokens, temperature, top_k)
    """

    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.dim),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([
                TransformerBlock(config, layer_idx=i)
                for i in range(config.n_layers)
            ]),
            ln_f=make_norm(config.dim, config),
        ))
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)

        # Weight tying: share embedding and output projection
        self.transformer.wte.weight = self.lm_head.weight

        # Precompute RoPE frequencies
        freqs_cos, freqs_sin = precompute_freqs_cis(
            config.dim // config.n_heads, config.max_seq_len,
        )
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

        # Init weights
        self.apply(self._init_weights)
        # Scaled init for residual projections (wo and w2)
        for pn, p in self.named_parameters():
            if pn.endswith('wo.weight') or pn.endswith('w2.weight'):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layers),
                )

        # Log active modifications
        mods = []
        if config.use_dyt:
            mods.append("DyT")
        if config.use_diff_attn:
            mods.append("DiffAttn")
        mod_str = " + ".join(mods) if mods else "Vanilla (RMSNorm)"
        print(f"[CompositionStudy] Llama Config: {mod_str}")
        print(f"number of parameters: %.2fM" % (self.get_num_params() / 1e6,))

    def get_num_params(self, non_embedding=True):
        """Count parameters. No wpe to subtract (RoPE has no learned params)."""
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            # Subtract tied embedding (counted once in lm_head)
            # Since weights are tied, we subtract the embedding table
            n_params -= self.transformer.wte.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        """Forward pass. Same signature as GPT.forward()."""
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.max_seq_len, (
            f"Cannot forward sequence of length {t}, "
            f"max_seq_len is only {self.config.max_seq_len}"
        )

        # Token embeddings (no positional embeddings -- RoPE handles position)
        x = self.transformer.wte(idx)
        x = self.transformer.drop(x)

        # Slice RoPE frequencies to current sequence length
        freqs_cos = self.freqs_cos[:t]
        freqs_sin = self.freqs_sin[:t]

        # Transformer blocks
        for block in self.transformer.h:
            x = block(x, freqs_cos, freqs_sin)

        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-1,
            )
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        """Same interface as GPT.configure_optimizers()."""
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0},
        ]
        num_decay = sum(p.numel() for p in decay_params)
        num_nodecay = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay:,} parameters")
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas, **extra_args,
        )
        print(f"using fused AdamW: {use_fused}")
        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """Estimate model flops utilization (MFU) vs A100 bf16 peak."""
        N = self.get_num_params()
        cfg = self.config
        L = cfg.n_layers
        H = cfg.n_heads
        Q = cfg.dim // cfg.n_heads
        T = cfg.max_seq_len
        flops_per_token = 6 * N + 12 * L * H * Q * T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        flops_achieved = flops_per_iter * (1.0 / dt)
        flops_promised = 312e12  # A100 bf16 peak
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = (
                idx if idx.size(1) <= self.config.max_seq_len
                else idx[:, -self.config.max_seq_len:]
            )
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
