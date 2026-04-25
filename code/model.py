"""
Unified GPT model with toggle flags for recent architectural modifications.
Based on nanoGPT (Karpathy). Each modification can be independently enabled.

Modifications implemented:
  - DyT (Dynamic Tanh): replaces LayerNorm with tanh(alpha*x)*weight+bias
  - HardTanh: hard clipping control for activation bounding
  - DiffAttn (Differential Attention): dual-softmax attention that cancels noise
  - DiffAttn V2: sigmoid-bounded lambda variant
  - GatedAttn: learnable per-head scalar gate on attention output

References:
  - nanoGPT: https://github.com/karpathy/nanoGPT
  - DyT: Zhu et al., "Transformers without Normalization" (CVPR 2025)
  - DiffAttn: Ye et al., "Differential Transformer" (ICLR 2025)
  - GatedAttn: Qiu et al., "Unlocking the Potential of Dense Attention" (NeurIPS 2026)
"""

import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


# ============================================================
# Normalization variants
# ============================================================

class VanillaLayerNorm(nn.Module):
    """Standard LayerNorm (nanoGPT default)."""
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)


class DynamicTanh(nn.Module):
    """DyT: replace LayerNorm with tanh(alpha*x)*weight+bias.
    From Zhu et al., "Transformers without Normalization" (CVPR 2025).
    Core idea: 37 lines of code, drop-in replacement.
    """
    def __init__(self, ndim, bias, alpha_init=2.0):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1) * alpha_init)
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x):
        x = torch.tanh(self.alpha * x)
        x = x * self.weight
        if self.bias is not None:
            x = x + self.bias
        return x


class RMSNorm(nn.Module):
    """RMSNorm: used in Llama, Mistral, Qwen. Modern LLM standard."""
    def __init__(self, ndim, bias=False):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.eps = 1e-6

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class HardTanhNorm(nn.Module):
    """HardTanh normalization replacement: clips activations to [-1, 1].

    This tests whether activation bounding, independent of smooth tanh
    saturation, is sufficient to reproduce the DyT regime pattern.
    """
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x):
        x = F.hardtanh(x)
        x = x * self.weight
        if self.bias is not None:
            x = x + self.bias
        return x


def make_norm(ndim, bias, config):
    """Factory: select normalization based on config."""
    if config.use_rmsnorm:
        return RMSNorm(ndim, bias)
    elif config.use_dyt:
        return DynamicTanh(ndim, bias, alpha_init=config.dyt_alpha_init)
    elif config.use_hardtanh:
        return HardTanhNorm(ndim, bias)
    else:
        return VanillaLayerNorm(ndim, bias)


# ============================================================
# Attention variants
# ============================================================

class CausalSelfAttention(nn.Module):
    """Standard multi-head causal self-attention (nanoGPT default)."""

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None,
            dropout_p=self.dropout if self.training else 0,
            is_causal=True,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


class DifferentialAttention(nn.Module):
    """Differential Attention: attn = softmax(Q1K1) - lambda * softmax(Q2K2).
    From Ye et al., "Differential Transformer" (ICLR 2025).
    Splits each head into two sub-heads, computes two attention maps,
    subtracts to cancel noise. Uses half the heads but double sub-heads.
    """

    def __init__(self, config, layer_idx=0):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.dropout = config.dropout
        # Each head is split into 2 sub-heads for differential computation
        # Q, K get 2x projections; V stays the same
        self.q_proj = nn.Linear(config.n_embd, 2 * config.n_embd, bias=config.bias)
        self.k_proj = nn.Linear(config.n_embd, 2 * config.n_embd, bias=config.bias)
        self.v_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.resid_dropout = nn.Dropout(config.dropout)
        # Learnable lambda for balancing the two attention maps
        # Initialize near 0.5 as per the paper
        self.lambda_q1 = nn.Parameter(torch.randn(self.head_dim) * 0.1)
        self.lambda_k1 = nn.Parameter(torch.randn(self.head_dim) * 0.1)
        self.lambda_q2 = nn.Parameter(torch.randn(self.head_dim) * 0.1)
        self.lambda_k2 = nn.Parameter(torch.randn(self.head_dim) * 0.1)
        self.lambda_init = 0.8 - 0.6 * math.exp(-0.3 * (layer_idx + 1))
        # GroupNorm for stabilizing differential attention
        self.subln = nn.GroupNorm(num_groups=self.n_head, num_channels=self.n_embd, affine=False)
        self.v2 = getattr(config, 'diff_attn_v2', False)

    def forward(self, x):
        B, T, C = x.size()
        nh = self.n_head
        hd = self.head_dim

        q = self.q_proj(x).view(B, T, nh, 2, hd)  # (B, T, nh, 2, hd)
        k = self.k_proj(x).view(B, T, nh, 2, hd)
        v = self.v_proj(x).view(B, T, nh, hd)

        q1, q2 = q[:, :, :, 0, :], q[:, :, :, 1, :]  # (B, T, nh, hd)
        k1, k2 = k[:, :, :, 0, :], k[:, :, :, 1, :]

        # Transpose to (B, nh, T, hd)
        q1 = q1.transpose(1, 2)
        q2 = q2.transpose(1, 2)
        k1 = k1.transpose(1, 2)
        k2 = k2.transpose(1, 2)
        v = v.transpose(1, 2)

        scale = 1.0 / math.sqrt(hd)

        # Compute two attention maps with causal masking
        attn1 = (q1 @ k1.transpose(-2, -1)) * scale
        attn2 = (q2 @ k2.transpose(-2, -1)) * scale
        # Causal mask
        causal_mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        attn1 = attn1.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        attn2 = attn2.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        attn1 = F.softmax(attn1, dim=-1)
        attn2 = F.softmax(attn2, dim=-1)

        # Compute lambda. V2 bounds the two learned lambda components with
        # sigmoid, avoiding the exponential overflow mode observed at Scale 5.
        if self.v2:
            lambda_val = (torch.sigmoid(torch.sum(self.lambda_q1 * self.lambda_k1))
                         - torch.sigmoid(torch.sum(self.lambda_q2 * self.lambda_k2))
                         + self.lambda_init)
        else:
            lambda_val = (torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1))
                         - torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2))
                         + self.lambda_init)

        # Differential attention
        attn = attn1 - lambda_val * attn2
        y = attn @ v  # (B, nh, T, hd)

        # GroupNorm (reshape for group norm)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.subln(y.transpose(1, 2)).transpose(1, 2)
        y = y * (1.0 - self.lambda_init)

        y = self.resid_dropout(self.c_proj(y))
        return y


class GatedAttention(nn.Module):
    """Gated Attention: learnable per-head scalar gate on attention output."""

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.gate = nn.Parameter(torch.ones(config.n_head))

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None,
            dropout_p=self.dropout if self.training else 0,
            is_causal=True,
        )
        y = y * self.gate.view(1, self.n_head, 1, 1)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


def make_attn(config, layer_idx=0):
    """Factory: select attention mechanism based on config."""
    if config.use_diff_attn:
        return DifferentialAttention(config, layer_idx=layer_idx)
    elif config.use_gated_attn:
        return GatedAttention(config)
    else:
        return CausalSelfAttention(config)


# ============================================================
# MLP (unchanged across modifications)
# ============================================================

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


# ============================================================
# Block and Model
# ============================================================

class Block(nn.Module):
    """Transformer block with optional modifications."""

    def __init__(self, config, layer_idx=0):
        super().__init__()
        self.ln_1 = make_norm(config.n_embd, config.bias, config)
        self.attn = make_attn(config, layer_idx=layer_idx)
        self.ln_2 = make_norm(config.n_embd, config.bias, config)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True
    # === Modification toggles ===
    use_dyt: bool = False           # DyT: replace LayerNorm with Dynamic Tanh
    use_hardtanh: bool = False      # HardTanh: hard clipping to [-1, 1]
    use_rmsnorm: bool = False       # RMSNorm: modern LLM baseline (Llama-style)
    dyt_alpha_init: float = 2.0     # DyT alpha initialization value
    use_diff_attn: bool = False     # DiffAttn: differential attention
    diff_attn_v2: bool = False      # DiffAttn V2: sigmoid-bounded lambda
    use_gated_attn: bool = False    # GatedAttn: learnable per-head gate


class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([Block(config, layer_idx=i) for i in range(config.n_layer)]),
            ln_f=make_norm(config.n_embd, config.bias, config),
        ))
        self.lm_head = nn.Linear(config.vocab_size if False else config.n_embd, config.vocab_size, bias=False)
        # Weight tying
        self.transformer.wte.weight = self.lm_head.weight

        # Init weights
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        # Log active modifications
        mods = []
        if config.use_dyt: mods.append("DyT")
        if config.use_hardtanh: mods.append("HardTanh")
        if config.use_rmsnorm: mods.append("RMSNorm")
        if config.use_diff_attn: mods.append("DiffAttnV2" if config.diff_attn_v2 else "DiffAttn")
        if config.use_gated_attn: mods.append("GatedAttn")
        mod_str = " + ".join(mods) if mods else "Vanilla"
        print(f"[TransformerStudy] Config: {mod_str}")
        print(f"number of parameters: %.2fM" % (self.get_num_params() / 1e6,))

    def get_num_params(self, non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device)

        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)

        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-1)
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss

    def crop_block_size(self, block_size):
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
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
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")
        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd // cfg.n_head, cfg.block_size
        flops_per_token = 6 * N + 12 * L * H * Q * T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        flops_achieved = flops_per_iter * (1.0 / dt)
        flops_promised = 312e12  # A100 bfloat16 peak
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
