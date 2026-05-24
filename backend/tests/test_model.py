"""PatchTST architecture tests — pure, CPU-only, no DB.

Covers: patch arithmetic, forward shapes, the feature gate, parameter budget,
and a learning-sanity check (overfit one fixed batch) so we know the trunk +
heads can fit *something* before we ever wire up real data.
"""

from __future__ import annotations

import torch

from backend.ml.model import (
    HORIZONS,
    DropPath,
    FeatureGate,
    PatchTST,
    PatchTSTConfig,
)


def _batch(cfg: PatchTSTConfig, b: int, num_tickers: int, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(b, cfg.seq_len, cfg.n_features, generator=g)
    ticker_idx = torch.randint(1, num_tickers, (b,), generator=g)
    return x, ticker_idx


# =============================================================
# Config / patch arithmetic
# =============================================================


def test_horizons_order_is_ascending():
    assert HORIZONS == ("1M", "3M", "6M", "1Y")


def test_patch_arithmetic():
    cfg = PatchTSTConfig()
    assert cfg.n_patches == 12          # 252 / 21
    assert cfg.patch_dim == 252         # 21 * 12


def test_seq_len_must_divide_by_patch_len():
    cfg = PatchTSTConfig(seq_len=250, patch_len=21)
    try:
        _ = cfg.n_patches
    except ValueError:
        return
    raise AssertionError("expected ValueError for non-divisible seq_len")


# =============================================================
# Forward shapes
# =============================================================


def test_forward_shapes():
    cfg = PatchTSTConfig()
    model = PatchTST(cfg, num_tickers=36).eval()
    x, idx = _batch(cfg, b=4, num_tickers=36)
    out = model(x, idx)
    assert set(out) == set(HORIZONS)
    for h in HORIZONS:
        assert out[h].shape == (4, cfg.n_classes)


def test_return_gate_shape_and_normalization():
    cfg = PatchTSTConfig()
    model = PatchTST(cfg, num_tickers=36).eval()
    x, idx = _batch(cfg, b=4, num_tickers=36)
    _, gate = model(x, idx, return_gate=True)
    assert gate.shape == (4, cfg.n_features)
    # softmax * n_features → each row sums to n_features (mean weight 1).
    assert torch.allclose(gate.sum(dim=1), torch.full((4,), float(cfg.n_features)), atol=1e-4)
    assert (gate >= 0).all()


def test_gate_can_be_disabled():
    cfg = PatchTSTConfig(use_feature_gate=False)
    model = PatchTST(cfg, num_tickers=36).eval()
    assert model.feature_gate is None
    x, idx = _batch(cfg, b=2, num_tickers=36)
    _, gate = model(x, idx, return_gate=True)
    assert gate is None


def test_patchify_rejects_wrong_shape():
    cfg = PatchTSTConfig()
    model = PatchTST(cfg, num_tickers=36)
    bad = torch.randn(2, cfg.seq_len - 1, cfg.n_features)
    try:
        model(bad, torch.tensor([1, 2]))
    except ValueError:
        return
    raise AssertionError("expected ValueError for wrong sequence length")


# =============================================================
# Parameter budget (~1M, per plan §4)
# =============================================================


def test_param_count_near_one_million():
    model = PatchTST(PatchTSTConfig(), num_tickers=36)
    n = model.num_parameters()
    assert 0.5e6 < n < 1.2e6, f"param count {n} outside expected ~1M band"


# =============================================================
# DropPath behavior
# =============================================================


def test_droppath_noop_when_eval_or_zero():
    x = torch.randn(8, 12, 16)
    dp_eval = DropPath(0.5).eval()
    assert torch.equal(dp_eval(x), x)              # eval → passthrough
    dp_zero = DropPath(0.0).train()
    assert torch.equal(dp_zero(x), x)              # p=0 → passthrough


def test_droppath_drops_whole_samples_in_train():
    torch.manual_seed(0)
    x = torch.ones(1000, 4)
    out = DropPath(0.5).train()(x)
    # Each row is either all-zero (dropped) or all-equal (kept & rescaled).
    row_is_zero = (out == 0).all(dim=1)
    row_is_kept = torch.isclose(out, torch.full_like(out, 2.0)).all(dim=1)  # 1/0.5
    assert (row_is_zero | row_is_kept).all()
    frac_dropped = row_is_zero.float().mean().item()
    assert 0.4 < frac_dropped < 0.6                # ≈ p


# =============================================================
# Learning sanity: overfit one fixed batch
# =============================================================


def test_overfits_single_batch():
    """A small, dropout-free model should drive a fixed batch's loss to ~0.

    This is the 'can the trunk + heads learn anything' gate from the build
    sequence — run before trusting it on real data.
    """
    torch.manual_seed(0)
    cfg = PatchTSTConfig(
        d_model=32, n_heads=4, d_ff=64, n_layers=2,
        dropout=0.0, drop_path=0.0, gate_hidden=16,
    )
    model = PatchTST(cfg, num_tickers=5).train()

    b = 16
    x, idx = _batch(cfg, b=b, num_tickers=5, seed=1)
    labels = {
        h: torch.randint(0, cfg.n_classes, (b,), generator=torch.Generator().manual_seed(i))
        for i, h in enumerate(HORIZONS)
    }

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()

    first_loss = None
    for _ in range(400):
        opt.zero_grad()
        out = model(x, idx)
        loss = sum(loss_fn(out[h], labels[h]) for h in HORIZONS)
        if first_loss is None:
            first_loss = loss.item()
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        out = model(x, idx)
    # Loss collapsed and every horizon is perfectly classified on the train batch.
    assert loss.item() < 0.05 * first_loss
    for h in HORIZONS:
        acc = (out[h].argmax(dim=1) == labels[h]).float().mean().item()
        assert acc == 1.0, f"{h} train accuracy {acc} != 1.0"


def test_feature_gate_unit():
    gate = FeatureGate(n_features=12, hidden=16)
    x = torch.randn(3, 252, 12)
    gated, w = gate(x)
    assert gated.shape == x.shape
    assert w.shape == (3, 12)
    assert torch.allclose(w.sum(dim=1), torch.full((3,), 12.0), atol=1e-4)
