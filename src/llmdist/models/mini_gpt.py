"""MiniGPT: the reference transformer used throughout the course.

Deliberately plain: no FlashAttention, no fused kernels, no tricks — so that
every byte of memory and every FLOP is attributable to a line you can read.
Chapter 03 derives the parameter/activation/FLOP formulas below; this module
implements them as methods so experiments can check theory against measurement.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 8192
    block_size: int = 256      # max sequence length
    n_layer: int = 4
    n_head: int = 4
    d_model: int = 256
    dropout: float = 0.0
    bias: bool = True

    @property
    def d_head(self) -> int:
        assert self.d_model % self.n_head == 0
        return self.d_model // self.n_head


class CausalSelfAttention(nn.Module):
    """Standard multi-head causal attention, written out explicitly."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        # One fused projection for Q, K, V: (d_model -> 3*d_model).
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=cfg.bias)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=cfg.bias)
        self.drop = nn.Dropout(cfg.dropout)
        mask = torch.tril(torch.ones(cfg.block_size, cfg.block_size, dtype=torch.bool))
        self.register_buffer("causal_mask", mask.view(1, 1, cfg.block_size, cfg.block_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        H, Dh = self.cfg.n_head, self.cfg.d_head
        q, k, v = self.qkv(x).split(C, dim=2)                 # each (B, T, C)
        q = q.view(B, T, H, Dh).transpose(1, 2)               # (B, H, T, Dh)
        k = k.view(B, T, H, Dh).transpose(1, 2)
        v = v.view(B, T, H, Dh).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(Dh)       # (B, H, T, T)
        att = att.masked_fill(~self.causal_mask[:, :, :T, :T], float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.drop(att)
        y = att @ v                                           # (B, H, T, Dh)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.drop(self.proj(y))


class MLP(nn.Module):
    """Position-wise feed-forward: d_model -> 4*d_model -> d_model."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.fc = nn.Linear(cfg.d_model, 4 * cfg.d_model, bias=cfg.bias)
        self.proj = nn.Linear(4 * cfg.d_model, cfg.d_model, bias=cfg.bias)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.proj(F.gelu(self.fc(x))))


class Block(nn.Module):
    """Pre-LN transformer block: x + Attn(LN(x)), then x + MLP(LN(x))."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class MiniGPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight  # weight tying
        self.apply(self._init)

    @staticmethod
    def _init(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx: torch.Tensor,
                targets: torch.Tensor | None = None
                ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    # ---- The accountant's methods (derived in chapter 03) ----------------

    def num_params(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.pos_emb.weight.numel()
        return n

    def param_formula(self) -> int:
        """Closed-form parameter count; must equal num_params().

        Per block: attention 4*d^2 (+4d bias: 3d in qkv, d in proj)
                   mlp       8*d^2 (+5d bias: 4d in fc, d in proj)
                   2 layernorms: 4d
        Embeddings: (V + T_max) * d ; final LN: 2d ; head tied to tok_emb.
        """
        c = self.cfg
        d, V, T = c.d_model, c.vocab_size, c.block_size
        bias = 9 * d if c.bias else 0
        per_block = 12 * d * d + bias + 4 * d
        return (V + T) * d + c.n_layer * per_block + 2 * d

    def flops_per_token(self) -> int:
        """Training FLOPs per token, ~6*N plus attention's 12*T*d per layer
        (the T-dependent term that dominates long-context training)."""
        c = self.cfg
        N = self.num_params(non_embedding=True)
        return 6 * N + c.n_layer * 12 * c.block_size * c.d_model
