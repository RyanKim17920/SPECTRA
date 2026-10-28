"""The in-run SELECTION eval. Uses ONLY PLISM tiles (+ the GTEx guard cache) -- never
PathoROB / HEST / THUNDER / CPTAC data.

Eval set: ``eval.locations_per_core`` locations from each of the 7 HELD-OUT tissue cores
(never in a training batch of a selection run) imaged under every condition except
``eval.exclude_conditions`` (HRH_S60 is defocused). Per exported readout r in eval.readouts
(cls / mean / clsmean):

* ``r/ret_top1_all``    cross-condition retrieval top-1: for every ordered condition pair
  (a, b), a != b, query a's tile at location t against all N eval locations of b; hit iff
  the nearest is t. Also ``_xscanner`` (same stain, other scanner = same glass), ``_xstain``
  (same scanner, other stain), ``_xboth`` and ``_heldaxis`` (pairs touching a held-out
  scanner / stain).
* ``r/croma_median`` / ``r/croma_ltm10`` / ``r/croma_f0`` / ``r/croma_q10``: CRoMa with
  label = tissue core, confounder = condition, group = location (same-location rows excluded
  from SO/OS typing). ``CRoMa_i = (d_OS - d_SO)/(d_OS + d_SO)`` with d = mean cosine distance
  to the m nearest SO (same core, other condition) / OS (other core, same condition) tiles.
  Also ``r/croma_heldaxis_*`` restricted to the held-out-axis conditions.
* ``r/erank``; ``r/retention_cka|cos|knn10`` vs the frozen base on the GTEx guard cache's
  retention split (when the guard cache exists).

PRE-REGISTERED PRIMARY SCALAR (fixed before any training result was looked at):

    primary = mean_{r in readouts} [ ret_top1_xscanner(r) + (croma_median(r) + croma_ltm10(r)) / 2 ] / 2

i.e. equal weight to SAME-GLASS cross-scanner retrieval on the held-out cores and the
core-level CRoMa margin, median and lower-tail mean equally; ``primary_se`` = SD of the same
quantity over ``eval.bootstrap`` core-level bootstrap resamples of the eval cores (seeded).
The stopping rule (:mod:`spectra.sweep.stopping`) reads this series.

GTEx UTILITY GUARD (:class:`GuardEval`): ``guard_lin_bacc`` = balanced accuracy of a
40-tissue linear probe fit on donor-disjoint GTEx tiles, reported alongside the primary.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .config import cache_root
from .data import to_float01

GUARD_CACHE_NAME = "gtex_guard_v1.npy"


# --------------------------------------------------------------------------------------
# embedding
# --------------------------------------------------------------------------------------


@torch.no_grad()
def embed(model, tiles, readouts, batch_size, device, base: bool = False) -> dict[str, torch.Tensor]:
    """``tiles`` (N,224,224,3) uint8 array-like -> {readout: (N, D) float32 on device}."""
    was = model.training
    model.eval()
    out: dict[str, list] = {r: [] for r in readouts}
    try:
        for i in range(0, len(tiles), batch_size):
            x = torch.from_numpy(np.array(tiles[i:i + batch_size])).to(device, non_blocking=True)
            x = model.prep(to_float01(x))
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                f = model.base_features(x, readouts) if base else \
                    model.pool_tokens(model.tokens(x), readouts)
            for r in readouts:
                out[r].append(f[r].float())
    finally:
        model.train(was)
    return {r: torch.cat(v) for r, v in out.items()}


# --------------------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------------------


def retrieval_hits(e: torch.Tensor) -> torch.Tensor:
    """e: (K, N, D) -> hits (K, K, N) bool: query (a, t) finds t among b's N tiles.
    The diagonal a == b is meaningless and set False."""
    k, n, _ = e.shape
    en = F.normalize(e.float(), dim=-1)
    flat = en.reshape(k * n, -1)
    tgt = torch.arange(n, device=e.device)
    hits = torch.zeros(k, k, n, dtype=torch.bool, device=e.device)
    for a in range(k):
        s = (en[a] @ flat.t()).view(n, k, n)          # (t, b, s)
        hits[a] = (s.argmax(-1) == tgt[:, None]).t()  # (b, t)
    hits[torch.arange(k), torch.arange(k)] = False
    return hits


def retrieval_top1(e: torch.Tensor) -> dict[str, float]:
    k = e.shape[0]
    h = retrieval_hits(e).float()
    off = ~torch.eye(k, dtype=torch.bool, device=e.device)
    per_pair = h.mean(-1)[off]
    return {"ret_top1": float(per_pair.mean()), "ret_top1_min": float(per_pair.min())}


def croma_values(feats: torch.Tensor, label, conf, group, m: int = 5) -> np.ndarray:
    """Exact brute-force per-sample CRoMa (same definition as croma.CRoMa, design 'all')."""
    x = F.normalize(feats.float(), dim=-1)
    dev = x.device
    lab = torch.as_tensor(np.asarray(label), device=dev)
    cf = torch.as_tensor(np.asarray(conf), device=dev)
    gr = torch.as_tensor(np.asarray(group), device=dev)
    vals = []
    for i in range(0, x.shape[0], 1024):
        d = 1.0 - x[i:i + 1024] @ x.t()
        sl, sc, sg = lab[i:i + 1024, None], cf[i:i + 1024, None], gr[i:i + 1024, None]
        other_g = gr[None, :] != sg
        so = (lab[None, :] == sl) & (cf[None, :] != sc) & other_g
        os_ = (lab[None, :] != sl) & (cf[None, :] == sc) & other_g
        dso = d.masked_fill(~so, math.inf).topk(m, dim=1, largest=False).values.mean(1)
        dos = d.masked_fill(~os_, math.inf).topk(m, dim=1, largest=False).values.mean(1)
        vals.append(((dos - dso) / (dos + dso)).cpu())
    v = torch.cat(vals).numpy()
    if not np.isfinite(v).all():
        raise RuntimeError("CRoMa: some samples lack m SO/OS neighbours")
    return v


def croma_summary(v: np.ndarray, alpha: float = 0.1) -> dict[str, float]:
    tail = np.sort(v)[: max(1, int(math.ceil(alpha * v.size)))]
    return {"croma_median": float(np.median(v)), "croma_ltm10": float(tail.mean()),
            "croma_f0": float(np.mean(v <= 0.0)), "croma_q10": float(np.quantile(v, alpha))}


def croma_torch(feats, label, conf, group, m=5, alpha=0.1) -> dict[str, float]:
    return croma_summary(croma_values(feats, label, conf, group, m), alpha)


def effective_rank(e: torch.Tensor) -> float:
    x = e.float() - e.float().mean(0, keepdim=True)
    s = torch.linalg.svdvals(x)
    p = s / s.sum().clamp_min(1e-12)
    return float(torch.exp(-(p * torch.log(p.clamp_min(1e-12))).sum()))


def linear_cka(x: torch.Tensor, y: torch.Tensor) -> float:
    x = x.double() - x.double().mean(0)
    y = y.double() - y.double().mean(0)
    hsic = (x.t() @ y).pow(2).sum()
    return float(hsic / ((x.t() @ x).pow(2).sum().sqrt() * (y.t() @ y).pow(2).sum().sqrt()))


def knn_overlap(x: torch.Tensor, y: torch.Tensor, k: int = 10) -> float:
    def nn(z):
        zn = F.normalize(z.float(), dim=-1)
        s = zn @ zn.t()
        s.fill_diagonal_(-2)
        return s.topk(k, dim=1).indices
    a, b = nn(x), nn(y)
    return float((a[:, :, None] == b[:, None, :]).any(-1).float().mean())


def primary_component(ret_xscanner: float, croma_med: float, croma_ltm: float) -> float:
    return (ret_xscanner + 0.5 * (croma_med + croma_ltm)) / 2.0


# --------------------------------------------------------------------------------------
# GTEx utility guard
# --------------------------------------------------------------------------------------


def guard_eval_path(cfg) -> Path | None:
    v = cfg.eval.guard
    if not v:
        return None
    p = cache_root() / GUARD_CACHE_NAME if v == "auto" else Path(v)
    if not p.exists() or not p.with_suffix(".json").exists():
        if v != "auto":
            raise FileNotFoundError(p)
        return None
    return p


def balanced_acc(pred: torch.Tensor, y: torch.Tensor, n_cls: int) -> float:
    rec = []
    for c in range(n_cls):
        m = y == c
        if bool(m.any()):
            rec.append(float((pred[m] == c).float().mean()))
    return float(np.mean(rec))


def linear_probe_predict(xtr, ytr, xva, n_cls: int, wd: float, iters: int = 500) -> torch.Tensor:
    """Multinomial logistic regression on standardised features (train-split stats),
    full-batch LBFGS, fixed L2 ``wd`` (pre-registered; no tuning on the scoring split)."""
    xtr, xva = xtr.float(), xva.float()
    mu, sd = xtr.mean(0, keepdim=True), xtr.std(0, keepdim=True) + 1e-6
    xtr, xva = (xtr - mu) / sd, (xva - mu) / sd
    w = torch.zeros(xtr.shape[1], n_cls, device=xtr.device, requires_grad=True)
    b = torch.zeros(n_cls, device=xtr.device, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], lr=1.0, max_iter=iters, history_size=20,
                            line_search_fn="strong_wolfe", tolerance_grad=1e-6, tolerance_change=1e-9)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(xtr @ w + b, ytr) + wd * w.pow(2).sum()
        loss.backward()
        return loss

    with torch.enable_grad():
        opt.step(closure)
    with torch.no_grad():
        return (xva @ w + b).argmax(1)


def knn_predict(xtr, ytr, xva, n_cls: int, k: int = 20) -> torch.Tensor:
    a, q = F.normalize(xtr.float(), dim=-1), F.normalize(xva.float(), dim=-1)
    s = q @ a.t()
    v, i = s.topk(k, dim=1)
    votes = torch.zeros(q.shape[0], n_cls, device=q.device)
    votes.scatter_add_(1, ytr[i], torch.exp(v / 0.07))
    return votes.argmax(1)


def guard_geometry(e: torch.Tensor, tissue: torch.Tensor, slide: torch.Tensor, r: str) -> dict:
    """Collapse geometry on the GTEx guard val split:
    cosine-distance ratios within-slide / between-tissue and within-tissue-across-slide / between-tissue,
    effective rank of the L2-normalised features. Reported, never gating."""
    z = torch.nn.functional.normalize(e.float(), dim=-1)
    d = 1 - z @ z.t()
    st = tissue[:, None] == tissue[None, :]
    ss = slide[:, None] == slide[None, :]
    eye = torch.eye(len(z), dtype=torch.bool, device=z.device)
    bt = d[~st].mean()
    return {f"geo/{r}/ratio_tissue_between": float(d[st & ~ss].mean() / bt),
            f"geo/{r}/ratio_slide_between": float(d[ss & ~eye].mean() / bt),
            f"geo/{r}/erank": _erank_cov(z)}


def _erank_cov(z: torch.Tensor) -> float:
    """exp(entropy) of the covariance eigenvalue distribution."""
    ev = torch.linalg.svdvals(z - z.mean(0, keepdim=True)).pow(2)
    p = ev / ev.sum().clamp_min(1e-12)
    p = p[p > 0]
    return float(torch.exp(-(p * p.log()).sum()))


class GuardEval:
    """UTILITY GUARD (pre-registered alongside the primary).

    GTEx tissue-type (40 classes) linear probe + kNN on the exported readouts of the
    CURRENT checkpoint, fit on the guard cache's ``train`` split and scored (balanced
    accuracy) on its donor-disjoint ``val`` split. GTEx is in no test benchmark. Keys:
    ``guard/<r>/lin_bacc``, ``guard/<r>/knn_bacc`` and the readout means
    ``guard_lin_bacc`` (THE guard), ``guard_knn_bacc``, plus ``guard_lin_bacc_se`` =
    SD over ``eval.bootstrap`` val-SLIDE bootstrap resamples (predictions fixed).
    """

    def __init__(self, cfg, device, path: Path):
        self.cfg, self.device = cfg, device
        meta = json.loads(path.with_suffix(".json").read_text())
        split = np.asarray(meta["split"])
        self.tr, self.va = np.where(split == "train")[0], np.where(split == "val")[0]
        arr = np.load(path, mmap_mode="r")
        self.tiles = np.asarray(arr[np.concatenate([self.tr, self.va])])
        lab = np.asarray(meta["label"])
        self.n_cls = int(lab.max()) + 1
        self.ytr = torch.as_tensor(lab[self.tr], device=device)
        self.yva = torch.as_tensor(lab[self.va], device=device)
        slides = np.asarray(meta["slide_id"])[self.va]
        _, self.va_slide = np.unique(slides, return_inverse=True)
        self.slide_rows = [np.where(self.va_slide == s)[0] for s in range(self.va_slide.max() + 1)]
        self.va_slide_t = torch.as_tensor(self.va_slide, device=device)
        self.rng_seed = cfg.eval.seed + 23
        #: donor-disjoint retention split (CKA / cosine / kNN overlap to the frozen base)
        self.ret_tiles = np.asarray(arr[np.where(split == "ret")[0][:cfg.eval.n_retention]])

    def __call__(self, model) -> dict:
        t0 = time.time()
        ro = list(dict.fromkeys(self.cfg.eval.readouts))
        emb = embed(model, self.tiles, ro, self.cfg.eval.batch_size, self.device)
        ntr = len(self.tr)
        rec: dict = {}
        preds = {}
        for r, e in emb.items():
            pl = linear_probe_predict(e[:ntr], self.ytr, e[ntr:], self.n_cls, self.cfg.eval.guard_wd)
            pk = knn_predict(e[:ntr], self.ytr, e[ntr:], self.n_cls, self.cfg.eval.guard_knn_k)
            rec[f"guard/{r}/lin_bacc"] = balanced_acc(pl, self.yva, self.n_cls)
            rec[f"guard/{r}/knn_bacc"] = balanced_acc(pk, self.yva, self.n_cls)
            preds[r] = pl
            rec.update(guard_geometry(e[ntr:], self.yva, self.va_slide_t, r))
        rec["guard_lin_bacc"] = float(np.mean([rec[f"guard/{r}/lin_bacc"] for r in emb]))
        rec["guard_knn_bacc"] = float(np.mean([rec[f"guard/{r}/knn_bacc"] for r in emb]))
        # val-slide cluster bootstrap, vectorised: per-slide x class (correct, total) counts
        rng = np.random.default_rng(self.rng_seed)
        n_s = len(self.slide_rows)
        y = self.yva.cpu().numpy()
        tot = np.zeros((n_s, self.n_cls))
        np.add.at(tot, (self.va_slide, y), 1)
        cors = []
        for p_ in preds.values():
            c = np.zeros((n_s, self.n_cls))
            np.add.at(c, (self.va_slide, y), (p_.cpu().numpy() == y).astype(float))
            cors.append(c)
        mult = np.stack([np.bincount(rng.integers(0, n_s, n_s), minlength=n_s)
                         for _ in range(max(self.cfg.eval.bootstrap, 2))])        # (B, S)
        tb = mult @ tot
        with np.errstate(invalid="ignore", divide="ignore"):
            boots = np.mean([np.nanmean((mult @ c) / tb, axis=1) for c in cors], axis=0)
        rec["guard_lin_bacc_se"] = float(np.std(boots, ddof=1))
        rec["guard_n_train"], rec["guard_n_val"] = int(ntr), int(len(self.va))
        rec["guard_s"] = time.time() - t0
        return rec


# --------------------------------------------------------------------------------------
# the hook
# --------------------------------------------------------------------------------------


class Evaluator:
    def __init__(self, cfg, device, eval_tiles: np.ndarray, eval_conds, eval_locs, labels: np.ndarray,
                 run_dir: Path, heldaxis=None, cores_in_train: bool = False, guard: "GuardEval | None" = None):
        self.cfg, self.device = cfg, device
        self.guard = guard
        self.eval_tiles = eval_tiles          # (K, N, 224, 224, 3)
        self.conds = list(eval_conds)
        self.locs = eval_locs
        #: retention tiles = the guard cache's donor-disjoint "ret" split (None: no retention metrics)
        self.ret_tiles = guard.ret_tiles if guard is not None else None
        self.cores_in_train = cores_in_train
        k, n = eval_tiles.shape[:2]
        self.loc_core = labels[eval_locs]                       # (N,)
        self.cores = np.unique(self.loc_core)
        self.flat_label = np.tile(self.loc_core, k)
        self.flat_conf = np.repeat(np.arange(k), n)
        self.flat_group = np.tile(eval_locs, k)
        sc = np.array([c.scanner for c in self.conds])
        st = np.array([c.stain for c in self.conds])
        self.same_sc = torch.as_tensor(sc[:, None] == sc[None, :], device=device)
        self.same_st = torch.as_tensor(st[:, None] == st[None, :], device=device)
        ha = np.zeros(k, bool) if heldaxis is None else np.asarray(heldaxis, bool)
        self.heldaxis = torch.as_tensor(ha, device=device)
        self.ro = list(dict.fromkeys(cfg.eval.readouts))
        per_core = np.bincount(self.loc_core)
        min_so = int((per_core[per_core > 0].min() - 1) * (k - 1))
        self.m = max(1, min(cfg.eval.croma_m, min_so))
        if self.m < cfg.eval.croma_m:
            print(f"[eval] CRoMa m capped {cfg.eval.croma_m} -> {self.m}", flush=True)
        self.base_ret = None
        self.base_path = run_dir / "base_retention_emb.pt"
        if self.base_path.exists():
            self.base_ret = torch.load(self.base_path, map_location="cpu")
        self.rng = np.random.default_rng(cfg.eval.seed)

    def ensure_base(self, model):
        """Frozen-base retention embeddings, computed once (before any update) and cached."""
        if self.base_ret is not None or self.ret_tiles is None:
            return
        e = embed(model, self.ret_tiles, self.ro, self.cfg.eval.batch_size, self.device, base=True)
        self.base_ret = {k: v.cpu() for k, v in e.items()}
        torch.save(self.base_ret, self.base_path)

    def _croma_vals(self, e: torch.Tensor, cond_sel: np.ndarray | None = None) -> np.ndarray:
        lab, conf, grp = self.flat_label, self.flat_conf, self.flat_group
        if cond_sel is not None:
            keep = np.isin(conf, np.where(cond_sel)[0])
            e, lab, conf, grp = e[torch.as_tensor(keep, device=e.device)], lab[keep], conf[keep], grp[keep]
        return croma_values(e, lab, conf, grp, self.m), lab

    def __call__(self, model, step: int) -> dict:
        t0 = time.time()
        cfg = self.cfg
        k, n = self.eval_tiles.shape[:2]
        flat = self.eval_tiles.reshape(k * n, 224, 224, 3)
        emb = embed(model, flat, self.ro, cfg.eval.batch_size, self.device)
        t_emb = time.time() - t0
        rec: dict = {"kind": "eval", "step": step, "n_conditions": int(k), "n_locations": int(n),
                     "eval_cores": self.cores.tolist(), "cores_in_train": self.cores_in_train,
                     "croma_m": self.m}
        off = ~torch.eye(k, dtype=torch.bool, device=self.device)
        cats = {"all": off, "xscanner": off & self.same_st & ~self.same_sc,
                "xstain": off & self.same_sc & ~self.same_st, "xboth": off & ~self.same_sc & ~self.same_st}
        if bool(self.heldaxis.any()):
            cats["heldaxis"] = off & (self.heldaxis[:, None] | self.heldaxis[None, :])
        boot_parts = []
        for r, e in emb.items():
            ek = e.view(k, n, -1)
            hits = retrieval_hits(ek).float()                 # (K, K, N)
            for name, msk in cats.items():
                if bool(msk.any()):
                    rec[f"{r}/ret_top1_{name}"] = float(hits[msk].mean())
            vals, vlab = self._croma_vals(e)
            for kk, v in croma_summary(vals, cfg.eval.croma_alpha).items():
                rec[f"{r}/{kk}"] = v
            if "heldaxis" in cats:
                hv, _ = self._croma_vals(e, self.heldaxis.cpu().numpy())
                for kk, v in croma_summary(hv, cfg.eval.croma_alpha).items():
                    rec[f"{r}/{kk.replace('croma_', 'croma_heldaxis_')}"] = v
            rec[f"{r}/erank"] = effective_rank(e)
            rec[f"{r}/primary"] = primary_component(rec[f"{r}/ret_top1_xscanner"],
                                                    rec[f"{r}/croma_median"], rec[f"{r}/croma_ltm10"])
            # per-location pieces for the core bootstrap (same-glass pairs, as in primary)
            hit_core = hits[cats["xscanner"]].mean(0).cpu().numpy()
            boot_parts.append((hit_core, vals, vlab))
        rec["primary"] = float(np.mean([rec[f"{r}/primary"] for r in emb]))
        rec["primary_se"] = self._bootstrap(boot_parts)
        if self.base_ret is not None:
            er = embed(model, self.ret_tiles, self.ro, cfg.eval.batch_size, self.device)
            for r in self.ro:
                a, b = er[r], self.base_ret[r].to(self.device)
                rec[f"{r}/retention_cka"] = linear_cka(a, b)
                rec[f"{r}/retention_cos"] = float(F.cosine_similarity(a, b, dim=-1).mean())
                rec[f"{r}/retention_knn10"] = knn_overlap(a, b, 10)
        if self.guard is not None:
            rec.update(self.guard(model))
        rec["eval_embed_s"] = t_emb
        rec["eval_s"] = time.time() - t0
        return rec

    def _bootstrap(self, parts) -> float:
        b_n = self.cfg.eval.bootstrap
        if b_n <= 1 or len(self.cores) < 2:
            return float("nan")
        alpha = self.cfg.eval.croma_alpha
        loc_idx = {c: np.where(self.loc_core == c)[0] for c in self.cores}
        val_idx = [{c: np.where(vl == c)[0] for c in self.cores} for _, _, vl in parts]
        out = []
        for _ in range(b_n):
            pick = self.rng.choice(self.cores, size=len(self.cores), replace=True)
            comps = []
            for (hit_core, vals, _), vi in zip(parts, val_idx):
                li = np.concatenate([loc_idx[c] for c in pick])
                vv = vals[np.concatenate([vi[c] for c in pick])]
                tail = np.sort(vv)[: max(1, int(math.ceil(alpha * vv.size)))]
                comps.append(primary_component(float(hit_core[li].mean()), float(np.median(vv)),
                                               float(tail.mean())))
            out.append(np.mean(comps))
        return float(np.std(out, ddof=1))
