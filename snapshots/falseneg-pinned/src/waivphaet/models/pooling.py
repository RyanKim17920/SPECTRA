"""Token-pooling heads for the contrastive loss (RESULTS 9's "the fix").

WHY THIS FILE EXISTS
--------------------
``--split-heads`` (commit ``3509c65``) scores ``mean(patch_tokens)`` through its own
projector. ``mean`` is **linear**, so::

    d(mean)/d(t_i) = (1/N) * I      for every i

-- the direct gradient reaching every patch token is the *identical vector*. The loss can
**translate** the token cloud but cannot express any preference about the tokens' relative
arrangement. And THUNDER's segmentation decoder is ``proj_dec = nn.Linear(d_encoder,
d_model)`` **with a bias**, so for any constant ``c``::

    proj_dec(t_i + c) = W*t_i + (W*c + b)

a trained decoder simply absorbs ``c`` into its bias. **A uniform translation of the token
cloud is exactly in the null space of the segmentation decoder, and a uniform translation
is precisely what the mean head's first-order gradient requests.** That is the standing
candidate mechanism for classification improving (32/36 pairs) while segmentation does not
(3/12). The fix is a pooling whose gradient is token-DEPENDENT.

THE SIGN PROBLEM -- WHY PLAIN CLAMP-GeM IS NOT THE DEFAULT
-----------------------------------------------------------
GeM, ``(1/N * sum x_i^p)^(1/p)``, was designed for **post-ReLU CNN feature maps**, where
``x >= 0`` holds by construction. The textbook implementation is literally
``x.clamp(min=eps).pow(p).mean(1).pow(1/p)``, and the clamp is normally a no-op there.

ViT tokens are **not** post-ReLU. The last operation before the token sequence leaves the
backbone is a LayerNorm, which makes every token vector *exactly* zero-mean across its
channels. So **about half of every token vector is negative by construction**, and
``clamp(x, min=eps)`` does not "stabilise" anything -- it **deletes roughly half the
signal**, silently, with the right shape and no warning. Worse, every deleted entry has
gradient exactly 0, so the very tokens the pooling was supposed to start discriminating
between become invisible to it in half their coordinates.

So the shipped default is GeM over a **non-negative transform** (``softplus``), which is
smooth, strictly positive, strictly monotone (no information destroyed, no dead gradient)
and asymptotically the identity for large positive x. ``gem_clamp`` is implemented too --
but only so the cost is **measured and reported** (``zero_fraction`` in
:meth:`GeMPool.extra_metrics`) rather than hidden. It is never the default.

``attn`` and ``lse`` have no sign problem at all: attention weights signed values, and the
LSE softmax is over token *positions*, so a negative entry is a low weight, not a deleted
one.

WHERE THE PREFIX-TOKEN SLICE LIVES
----------------------------------
Nowhere in this file. Every pooling module takes an already-sliced ``(B, N, D)`` **patch**
tensor; the slice is ``tokens[:, encoder.num_prefix_tokens:, :]`` and is done by
:class:`~waivphaet.models.encoder.WaivEncoder`, which reads that count off the loaded
backbone -- 1 on phikon-v2, **5 on Virchow2** ([CLS] + 4 register tokens). Hardcoding
``tokens[:, 1:]`` would feed four register tokens into the pooling on Virchow2: right
shape, right dtype, no warning, just a worse number.

PRECISION
---------
Every module here computes in **float32** regardless of autocast and returns float32.
``pow(p)``, ``softplus`` and a 196-way softmax in bfloat16 (8 mantissa bits) would put the
pooling's own rounding error on the order of the token-to-token differences the pooling
exists to resolve. The projector's first ``nn.Linear`` re-casts under autocast anyway, so
this costs one cast and nothing else. :class:`MeanPool` is the exception -- it is the
control, and must stay bit-identical to the ``patches.mean(dim=1)`` it replaces.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

#: Selectable values of ``--pool-head`` / ``EncoderConfig.pool_head``.
#:
#: ``mean`` is the incumbent (and the arm currently running as ``headcmp``); it is listed
#: so the flag can be passed explicitly without changing behaviour. ``gem`` is the
#: softplus variant -- ``gem_clamp`` is the textbook one and is a **diagnostic**, kept so
#: the sign problem can be measured instead of argued about.
POOL_HEAD_NAMES: tuple[str, ...] = ("mean", "gem", "gem_clamp", "attn", "lse")

#: Poolings whose gradient w.r.t. an individual token is token-DEPENDENT, i.e. the ones
#: that can express a preference about the tokens' relative arrangement. Gate G3 asserts
#: exactly this partition numerically; the tuple is here so the claim has one definition
#: rather than one per test.
TOKEN_DEPENDENT_POOLS: tuple[str, ...] = ("gem", "gem_clamp", "attn", "lse")


class MeanPool(nn.Module):
    """``patches.mean(dim=1)``. The incumbent, and the CONTROL for gate G3.

    Parameterless and deliberately **not** upcast to float32: it must reproduce the inline
    ``patches.mean(dim=1)`` in :meth:`WaivEncoder._pool_parts` bit-for-bit, because that is
    the arm already running.

    Its whole point is the defect: ``d(mean)/d(t_i) = (1/N) I`` for every ``i``, so the
    per-token gradients are identical and G3 asserts they are.
    """

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        return patches.mean(dim=1)

    def extra_metrics(self) -> dict[str, float]:
        return {}


class GeMPool(nn.Module):
    r"""Generalized mean over a NON-NEGATIVE transform of the tokens.

    .. math::  \mathrm{GeM}(t)_d = \Big(\tfrac{1}{N}\sum_i \phi(t_{i,d})^p\Big)^{1/p}

    with a single learnable scalar ``p`` (shared across channels, init 3.0). The gradient
    is :math:`\partial \mathrm{GeM}/\partial t_i \propto \phi(t_i)^{p-1}\,\phi'(t_i)`, i.e.
    **token-dependent** -- large-activation tokens get a larger share, which is the whole
    reason for the change.

    ``mode``:

    * ``"softplus"`` (default) -- :math:`\phi = \mathrm{softplus}`. Smooth, strictly
      positive, strictly monotone: no entry is destroyed and no entry has zero gradient.
    * ``"clamp"`` -- :math:`\phi = \max(x, \varepsilon)`, the textbook CNN implementation.
      **Only for measuring the sign problem.** On LayerNorm'd ViT tokens it zeroes about
      half of every token vector; :meth:`extra_metrics` reports the measured fraction so
      the cost lands in ``history.json`` rather than in a comment.

    ``p`` is clamped to ``[p_min, p_max] = [1, 8]`` in the forward pass. The lower bound
    keeps the map a mean-to-max interpolation rather than an inverted one; the upper bound
    is an overflow guard -- Virchow2 has attention-sink tokens with entries in the hundreds,
    and ``1e3 ** 8 = 1e24`` is the largest thing we are willing to hand to a float32
    reduction. Note ``p`` also sees AdamW's decoupled weight decay (it is an ordinary
    parameter in the single param group); at ``lr=1e-4, wd=0.05`` that is a 5e-6 relative
    pull per step, i.e. <1% over 1500 steps, which is why it is left in the group rather
    than given a bespoke one.
    """

    def __init__(
        self,
        p_init: float = 3.0,
        mode: str = "softplus",
        eps: float = 1e-6,
        p_min: float = 1.0,
        p_max: float = 8.0,
    ):
        super().__init__()
        if mode not in ("softplus", "clamp"):
            raise ValueError(f"GeMPool mode must be 'softplus' or 'clamp', got {mode!r}")
        self.mode = mode
        self.eps = float(eps)
        self.p_min, self.p_max = float(p_min), float(p_max)
        self.p = nn.Parameter(torch.tensor(float(p_init)))
        #: Fraction of token entries the clamp destroyed on the most recent forward.
        #: Always 0.0 in softplus mode (softplus is strictly positive), which is what
        #: makes it a fair side-by-side number rather than a mode-specific one.
        self.last_zero_fraction: float = 0.0

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        x = patches.float()
        if self.mode == "clamp":
            # THE SIGN PROBLEM, made visible. Everything at or below eps is flattened to
            # eps: no signal, and -- because clamp's gradient is 0 on the clamped side --
            # no gradient either. On LayerNorm'd tokens this is ~half of every vector.
            zeroed = (x <= self.eps)
            phi = x.clamp_min(self.eps)
            with torch.no_grad():
                self.last_zero_fraction = float(zeroed.float().mean())
        else:
            phi = F.softplus(x).clamp_min(self.eps)
            self.last_zero_fraction = 0.0
        p = self.p.float().clamp(self.p_min, self.p_max)
        return phi.pow(p).mean(dim=1).clamp_min(self.eps).pow(1.0 / p)

    def extra_metrics(self) -> dict[str, float]:
        return {
            "pool_gem_p": float(self.p.detach().clamp(self.p_min, self.p_max)),
            "pool_zero_fraction": float(self.last_zero_fraction),
        }


class AttnPool(nn.Module):
    """Attention pooling: ONE learned query, a single cross-attention over the tokens.

    ``out = sum_i softmax_i(<W_k t_i + b_k, q> / sqrt(D)) * t_i``.

    Native to a ViT, correct on **signed** features (a negative entry lowers a weight, it
    is never deleted), and the attention map is a genuine per-token saliency that can be
    inspected -- :attr:`last_attn` holds the most recent one, detached.

    The **values are the raw tokens**, with no value projection. That is deliberate: the
    pooled output then lives in the same space the mean head's output lives in, so the two
    arms' projector inputs are directly comparable and ``assert_split_head_inputs``'s
    cls-vs-pool separation keeps its old meaning. The key projection is initialised to the
    identity with a zero bias, so at step 0 the scores are exactly ``<t_i, q>/sqrt(D)`` --
    an inspectable saliency over the raw tokens rather than over a random remix of them.

    QUERY INIT IS LOAD-BEARING, AND WAS MEASURED
    --------------------------------------------
    The query is drawn ``N(0, 1)`` **per component**, not scaled to unit norm. Getting
    this wrong makes the head silently useless: with a unit-NORM query the scores are
    ``<t_i, q>/sqrt(D)``, and a LayerNorm'd token has ``||t_i|| = sqrt(D)`` exactly, so
    the logits land at std ~ ``1/sqrt(D)`` = 0.03. A softmax over 196 logits that tight is
    uniform to four decimals, i.e. **the attention pooling is the mean pooling**, which is
    the exact defect this file exists to remove. Measured on real phikon-v2 tokens, that
    init gave attention entropy 0.99999 and a G3 token-gradient spread of 6.0e-3, an order
    of magnitude below every other variant. Unit-VARIANCE per component restores the
    standard scaling (``<t_i, q>`` ~ ``N(0, D)``, divided by ``sqrt(D)`` -> std 1) and a
    genuinely non-uniform init. G3 is what caught it, on real tokens, before a GPU.

    There is deliberately **no separate temperature parameter**: the score is linear in
    ``q``, so ``||q||`` already *is* the sharpness knob and the head can sharpen its own
    attention by growing it. A second scalar would be an exactly degenerate direction.
    Measured at init on real phikon-v2 tokens: normalised attention entropy 0.995 and a
    pooled output only 2.3% (relative) away from the plain mean -- so this head starts
    near the mean in OUTPUT space and departs by learning, while its per-token GRADIENT is
    already token-dependent (G3 spread 0.19-0.30), which is the property that matters.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = int(dim)
        self.to_k = nn.Linear(dim, dim, bias=True)
        nn.init.eye_(self.to_k.weight)
        nn.init.zeros_(self.to_k.bias)
        self.query = nn.Parameter(torch.randn(dim))
        #: ``(B, N)`` attention weights from the most recent forward, detached. A plain
        #: attribute, not a buffer, so it never enters ``state_dict``.
        self.last_attn: torch.Tensor | None = None

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        k = self.to_k(patches).float()
        scores = (k @ self.query.float()) * (self.dim**-0.5)  # (B, N)
        w = torch.softmax(scores, dim=1)
        with torch.no_grad():
            self.last_attn = w.detach()
        return (w.unsqueeze(-1) * patches.float()).sum(dim=1)

    def extra_metrics(self) -> dict[str, float]:
        if self.last_attn is None:
            return {}
        w = self.last_attn
        n = w.shape[1]
        # Normalised entropy: 1.0 == uniform == the mean pooling it replaces, 0.0 == a
        # single token. The number that says whether the head is actually doing anything.
        ent = -(w.clamp_min(1e-12).log() * w).sum(dim=1).mean()
        import math

        return {
            "pool_attn_entropy": float(ent / math.log(n)),
            "pool_attn_max": float(w.max(dim=1).values.mean()),
        }


class LSEPool(nn.Module):
    """Softmax-weighted (log-sum-exp family) pooling with a learnable temperature.

    ``out_d = sum_i softmax_i(tau * t_{i,d}) * t_{i,d}``, i.e. the softmax runs over token
    POSITIONS, independently per channel. ``tau -> 0`` recovers the mean exactly and
    ``tau -> inf`` recovers the max, so like GeM it interpolates mean<->max -- but with
    **no sign problem whatsoever**: a very negative entry receives a small weight, it is
    not clamped away, and its gradient is not zero.

    The gradient is token-dependent:
    ``d out_d / d t_{i,d} = w_i * (1 + tau * (t_{i,d} - out_d))``, which varies with the
    token's own value both through ``w_i`` and through the bracket.

    ``tau`` is parameterised as ``exp(log_tau)`` so it stays strictly positive under any
    optimiser step, and ``log_tau`` is initialised to 0 (``tau = 1``). Token entries after
    LayerNorm are O(1), so ``tau = 1`` over ~196 positions is a mild, non-degenerate
    weighting -- neither the mean nor a hard max.
    """

    def __init__(self, tau_init: float = 1.0, log_tau_max: float = 4.0):
        super().__init__()
        if tau_init <= 0.0:
            raise ValueError(f"LSEPool tau_init must be > 0, got {tau_init}")
        self.log_tau = nn.Parameter(torch.tensor(float(tau_init)).log())
        #: Upper bound on log(tau) in the forward pass: at tau = e^4 ~ 55 the softmax over
        #: O(1) token entries is already effectively a hard max, and letting it run away
        #: turns the pooling into a max whose gradient reaches exactly one token per
        #: channel -- which is a different (and much sparser) failure than the one we set
        #: out to fix.
        self.log_tau_max = float(log_tau_max)

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        x = patches.float()
        tau = self.log_tau.float().clamp(max=self.log_tau_max).exp()
        w = torch.softmax(x * tau, dim=1)
        return (w * x).sum(dim=1)

    def extra_metrics(self) -> dict[str, float]:
        return {"pool_lse_tau": float(self.log_tau.detach().clamp(max=self.log_tau_max).exp())}


def build_pool_head(name: str, dim: int) -> nn.Module:
    """``name`` -> a pooling module over an already-sliced ``(B, N, dim)`` patch tensor.

    ``"mean"`` returns :class:`MeanPool`. The encoder does **not** use it: on
    ``--pool-head mean`` it keeps the literal ``patches.mean(dim=1)`` already inside
    ``_pool_parts``, so that arm is the *same code*, not merely the same arithmetic. The
    module exists so gate G3 can measure the incumbent's token-gradient spread with the
    same instrument it measures the alternatives with.
    """
    if name == "mean":
        return MeanPool()
    if name == "gem":
        return GeMPool(mode="softplus")
    if name == "gem_clamp":
        return GeMPool(mode="clamp")
    if name == "attn":
        return AttnPool(dim)
    if name == "lse":
        return LSEPool()
    raise ValueError(f"unknown pool head {name!r}; valid names are {list(POOL_HEAD_NAMES)}")


def token_gradient_spread(
    pool: nn.Module, patches: torch.Tensor, direction: torch.Tensor | None = None
) -> dict[str, float]:
    """GATE G3's instrument: how much does ``d(pool)/d(t_i)`` vary with ``i``?

    Backpropagates a fixed linear functional ``<direction, pool(patches)>`` and measures,
    per sample, the dispersion of the per-token gradient ``g_i`` about its own token-mean::

        spread_b = || g_i - mean_i(g) ||_F / || mean_i(g) ||_F     (Frobenius, over i and d)

    For ``mean`` this is **exactly 0** by construction -- every ``g_i`` equals
    ``direction / N`` -- which is the defect RESULTS 9 identifies, stated as a number.
    For a token-dependent pooling it is O(1).

    The scale-free normalisation matters: an unnormalised std would let a pooling look
    "differentiated" merely by having a larger gradient overall.

    Args:
        pool: the pooling module (or any ``(B, N, D) -> (B, D)`` callable).
        patches: ``(B, N, D)`` **real** patch tokens, already prefix-sliced.
        direction: optional ``(D,)`` functional. Defaults to all-ones, which is the
            harshest choice for the alternatives (it is exactly the direction under which
            an unweighted mean is stationary) and costs the variants nothing to beat.

    Returns:
        ``{"spread_mean", "spread_min", "spread_max", "max_abs_grad"}``.
    """
    x = patches.detach().clone().float().requires_grad_(True)
    out = pool(x)
    if out.ndim != 2 or out.shape[0] != x.shape[0] or out.shape[1] != x.shape[2]:
        raise ValueError(
            f"pool returned {tuple(out.shape)} for a {tuple(x.shape)} input; a pooling "
            "must map (B, N, D) -> (B, D)"
        )
    d = torch.ones(x.shape[2], dtype=out.dtype) if direction is None else direction.to(out)
    (out @ d).sum().backward()
    g = x.grad  # (B, N, D)
    mean_g = g.mean(dim=1, keepdim=True)  # (B, 1, D)
    num = (g - mean_g).flatten(1).norm(dim=1)
    den = mean_g.flatten(1).norm(dim=1).clamp_min(1e-30) * (g.shape[1] ** 0.5)
    spread = (num / den).detach()
    return {
        "spread_mean": float(spread.mean()),
        "spread_min": float(spread.min()),
        "spread_max": float(spread.max()),
        "max_abs_grad": float(g.abs().max()),
    }
