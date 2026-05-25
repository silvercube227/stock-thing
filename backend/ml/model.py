"""PatchTST-style encoder for multi-horizon trend classification.

A single shared transformer over the whole ticker universe. Each ticker is
identified by its append-only `embedding_idx` (see schema.sql), looked up in a
learned embedding table and broadcast onto every patch token — so attention can
condition on identity at every step rather than via a CLS concat.

Pipeline (see plan §4, with the variable-selection gate added per the
PatchTST-vs-TFT discussion):
    input (B, 252, 12)
      → FeatureGate: per-feature variable selection (softmax weights), the one
        idea worth borrowing from TFT for our heterogeneous features
      → reshape into 12 non-overlapping 21-day patches, flatten each to 252 dims
      → Linear(252, d_model)                       patch embedding
      → + learned positional embedding (12, d_model)
      → + ticker embedding (16 → d_model), broadcast to all 12 patches
      → 4 × pre-norm transformer encoder layers (8 heads, FFN 512, dropout 0.2,
        DropPath 0.1 on layers 2–4)
      → mean-pool over patches → (B, d_model)
      → shared MLP Linear(d_model, 64) + GELU
      → 4 horizon heads Linear(64, 2)   {1M, 3M, 6M, 1Y}

Output is per-horizon 2-class logits (down / up). direction_prob downstream is
softmax(logits)[..., 1]. ~0.84M params at the default config.

This module is pure architecture — no I/O, no DB, no training logic. That keeps
it trivially unit-testable on CPU (see tests/test_model.py).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn

from backend.ingestion.calendar import HORIZON_TRADING_DAYS
from backend.ml.features import FEATURE_DIM, SEQUENCE_LENGTH

# Horizon ordering is fixed here and reused by heads, loss weights, and the
# predictions writer. Sort by trading-day length so the order is deterministic.
HORIZONS: tuple[str, ...] = tuple(
    sorted(HORIZON_TRADING_DAYS, key=lambda h: HORIZON_TRADING_DAYS[h])
)


@dataclass
class ModelOutput:
    """Forward output. `returns` is per-horizon predicted log-return (B,), in
    real return space (predicted price = last_close * exp(r̂)); None if the
    model is classification-only. `gate` is the (B, n_features) feature-
    importance vector when requested, else None."""

    logits: dict[str, torch.Tensor]
    returns: dict[str, torch.Tensor] | None = None
    gate: torch.Tensor | None = None


@dataclass
class PatchTSTConfig:
    """Hyperparameters. Serialized verbatim into model_versions.config for
    reproducibility, so keep it JSON-friendly (plain scalars + lists)."""

    seq_len: int = SEQUENCE_LENGTH        # 252 trading days
    n_features: int = FEATURE_DIM         # 12 per-timestep features
    patch_len: int = 21                   # one trading month
    d_model: int = 128
    n_heads: int = 8
    d_ff: int = 512
    n_layers: int = 4
    dropout: float = 0.2
    drop_path: float = 0.1                # stochastic depth, applied to layers 2..n
    ticker_embed_dim: int = 16
    head_hidden: int = 64
    n_classes: int = 2                    # binary direction per horizon
    predict_returns: bool = True          # add a regression head per horizon (log-return)
    use_feature_gate: bool = True         # variable-selection gate before patching
    gate_hidden: int = 32
    horizons: tuple[str, ...] = field(default_factory=lambda: HORIZONS)

    @property
    def n_patches(self) -> int:
        if self.seq_len % self.patch_len != 0:
            raise ValueError(
                f"seq_len ({self.seq_len}) must be divisible by patch_len "
                f"({self.patch_len}); got remainder {self.seq_len % self.patch_len}"
            )
        return self.seq_len // self.patch_len

    @property
    def patch_dim(self) -> int:
        return self.patch_len * self.n_features


# =============================================================
# Feature gate (compact variable selection — TFT's best idea, minus the baggage)
# =============================================================


class FeatureGate(nn.Module):
    """Input-dependent per-feature variable selection.

    Summarizes each feature across the time window (mean + std), then a small
    shared MLP emits a softmax distribution over the features. The window is
    scaled per-feature by those weights, renormalized to mean 1 so total signal
    magnitude is preserved (softmax alone would shrink everything by ~1/F).

    The softmax weights double as interpretable per-feature importance for a
    given input window — surfaceable on the dashboard ("why is this bullish").
    ~1k params at the default config.
    """

    def __init__(self, n_features: int, hidden: int):
        super().__init__()
        self.n_features = n_features
        self.net = nn.Sequential(
            nn.Linear(n_features * 2, hidden),
            nn.GELU(),
            nn.Linear(hidden, n_features),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x: (B, L, F) → (gated x: (B, L, F), weights: (B, F))."""
        summary = torch.cat([x.mean(dim=1), x.std(dim=1)], dim=1)  # (B, 2F)
        weights = torch.softmax(self.net(summary), dim=1) * self.n_features  # mean 1
        return x * weights.unsqueeze(1), weights


# =============================================================
# Stochastic depth (DropPath)
# =============================================================


class DropPath(nn.Module):
    """Per-sample drop of a residual branch (stochastic depth).

    During training, zeroes the entire residual contribution for a random
    subset of the batch and rescales the survivors by 1/keep_prob so the
    expected value is unchanged. A no-op at eval or when p == 0.
    """

    def __init__(self, p: float = 0.0):
        super().__init__()
        if not 0.0 <= p < 1.0:
            raise ValueError(f"DropPath p must be in [0, 1); got {p}")
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.p == 0.0 or not self.training:
            return x
        keep = 1.0 - self.p
        # broadcast mask over all dims except batch
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.empty(shape, dtype=x.dtype, device=x.device).bernoulli_(keep)
        return x / keep * mask


# =============================================================
# Pre-norm transformer encoder layer
# =============================================================


class EncoderLayer(nn.Module):
    """Pre-norm self-attention + FFN block with optional stochastic depth.

    Custom (rather than nn.TransformerEncoderLayer) so DropPath wraps each
    residual branch — the stock layer applies dropout only inside the
    sublayers, not to the residual path.
    """

    def __init__(self, cfg: PatchTSTConfig, drop_path: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.d_model)
        self.attn = nn.MultiheadAttention(
            cfg.d_model, cfg.n_heads, dropout=cfg.dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(cfg.d_model)
        self.ff = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ff),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_ff, cfg.d_model),
        )
        self.drop_path = DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + self.drop_path(attn_out)
        x = x + self.drop_path(self.ff(self.norm2(x)))
        return x


# =============================================================
# Full model
# =============================================================


class PatchTST(nn.Module):
    """Shared multi-ticker PatchTST encoder with 4 horizon heads.

    Args:
        cfg:        hyperparameter bundle.
        num_tickers: size of the ticker embedding table. Must be > the largest
                     embedding_idx in use. embedding_idx is 1-based and
                     append-only, so pass (max_embedding_idx + 1). Index 0 is
                     reserved/unused padding.
    """

    def __init__(self, cfg: PatchTSTConfig, num_tickers: int):
        super().__init__()
        self.cfg = cfg
        self.num_tickers = num_tickers

        self.feature_gate = (
            FeatureGate(cfg.n_features, cfg.gate_hidden) if cfg.use_feature_gate else None
        )
        self.patch_embed = nn.Linear(cfg.patch_dim, cfg.d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, cfg.n_patches, cfg.d_model))
        self.ticker_embed = nn.Embedding(num_tickers, cfg.ticker_embed_dim)
        self.ticker_proj = nn.Linear(cfg.ticker_embed_dim, cfg.d_model)
        self.input_dropout = nn.Dropout(cfg.dropout)

        # DropPath schedule: 0 on the first layer, cfg.drop_path on layers 2..n.
        drop_paths = [0.0] + [cfg.drop_path] * (cfg.n_layers - 1)
        self.layers = nn.ModuleList(EncoderLayer(cfg, dp) for dp in drop_paths)
        self.norm = nn.LayerNorm(cfg.d_model)

        self.head_trunk = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.head_hidden),
            nn.GELU(),
        )
        self.heads = nn.ModuleDict(
            {h: nn.Linear(cfg.head_hidden, cfg.n_classes) for h in cfg.horizons}
        )
        # Regression heads predict the horizon log-return (one scalar each).
        self.return_heads = (
            nn.ModuleDict({h: nn.Linear(cfg.head_hidden, 1) for h in cfg.horizons})
            if cfg.predict_returns
            else None
        )

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.ticker_embed.weight, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _patchify(self, x: torch.Tensor) -> torch.Tensor:
        """(B, seq_len, n_features) → (B, n_patches, patch_len * n_features)."""
        b, seq_len, n_feat = x.shape
        if seq_len != self.cfg.seq_len or n_feat != self.cfg.n_features:
            raise ValueError(
                f"expected input (B, {self.cfg.seq_len}, {self.cfg.n_features}), "
                f"got (B, {seq_len}, {n_feat})"
            )
        # (B, n_patches, patch_len, n_features) → flatten the patch contents.
        x = x.reshape(b, self.cfg.n_patches, self.cfg.patch_len, n_feat)
        return x.reshape(b, self.cfg.n_patches, self.cfg.patch_dim)

    def forward(
        self,
        x: torch.Tensor,
        ticker_idx: torch.Tensor,
        return_gate: bool = False,
    ) -> ModelOutput:
        """
        Args:
            x:           (B, seq_len, n_features) float feature window.
            ticker_idx:  (B,) long tensor of embedding_idx values.
            return_gate: attach the (B, n_features) feature-importance weights
                         to the output (None if the gate is disabled).

        Returns:
            ModelOutput with per-horizon class logits (B, n_classes), per-horizon
            predicted log-returns (B,) when enabled, and optionally the gate.
        """
        gate_weights: torch.Tensor | None = None
        if self.feature_gate is not None:
            x, gate_weights = self.feature_gate(x)

        patches = self._patchify(x)                       # (B, P, patch_dim)
        tokens = self.patch_embed(patches)                # (B, P, d_model)
        tokens = tokens + self.pos_embed                  # broadcast over batch
        ticker = self.ticker_proj(self.ticker_embed(ticker_idx))  # (B, d_model)
        tokens = tokens + ticker.unsqueeze(1)             # broadcast over patches
        tokens = self.input_dropout(tokens)

        for layer in self.layers:
            tokens = layer(tokens)
        tokens = self.norm(tokens)

        pooled = tokens.mean(dim=1)                        # (B, d_model)
        trunk = self.head_trunk(pooled)                    # (B, head_hidden)
        logits = {h: head(trunk) for h, head in self.heads.items()}
        returns = (
            {h: head(trunk).squeeze(-1) for h, head in self.return_heads.items()}
            if self.return_heads is not None
            else None
        )
        return ModelOutput(
            logits=logits,
            returns=returns,
            gate=gate_weights if return_gate else None,
        )

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
