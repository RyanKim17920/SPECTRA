"""The Q8 trainer: one config -> one run directory.

    python -m spectra.sweep.train --config configs/v2/q8_selection.yaml [key.sub=value ...]
    python -m spectra.sweep.train --config configs/v2/q8_selection.yaml --dry-run
    python -m spectra.sweep.train --run-dir <runs root>/<sweep>/<run>
        (re-run / resume from the config.json already stored there -- what sbatch does)

One step: C conditions x T shared locations -> CLS -> projector -> multi-temperature grid
InfoNCE, plus the queue adversary (stain / scanner) on the CLS feature through a gradient
reversal layer. The in-run PLISM selection eval (``spectra.sweep.evaluate``) runs on
``eval.every`` / ``eval.schedule`` and at the last step; checkpoints hold the trainable
tensors only (LoRA, projector, adversary heads).

Run directory layout::

    config.json        resolved config (+ _meta.hash / run_name)
    metrics.jsonl      {"kind": "train"|"eval"|"info", "step": ...} one JSON per line
    state.json         {"status": running|done|failed|preempted, "step", "job_id", ...}
    resume.pt          latest resumable state (trainable params, optimizer, RNG, adversary queue)
    ckpt/step_XXXXXXX.safetensors   trainable params only
    base_retention_emb.pt           frozen-base retention embeddings (step-0 cache)
    DONE               written on successful completion
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import signal
import socket
import sys
import time
from pathlib import Path

import numpy as np
import torch

from spectra.data.grid import assert_grid_batch

from . import losses as L
from .config import (SweepConfig, config_hash, dump, load_config, load_dumped, run_name, runs_root)
from .data import (build_train_loader, eval_locations, eval_tile_set, heldout_location_mask,
                   load_group_labels, pick_eval_conditions, pick_eval_cores, resolve_conditions,
                   scanner_stain_labels, to_float01, train_location_pool)
from .evaluate import Evaluator, GuardEval, guard_eval_path
from .model import SweepModel

_STOP = {"flag": False, "sig": None}


def _on_signal(sig, _frame):
    _STOP["flag"], _STOP["sig"] = True, sig
    print(f"[train] caught signal {sig}: will save resume state and exit", flush=True)


# ======================================================================================
# small utils
# ======================================================================================


class JsonlLog:
    def __init__(self, path: Path):
        self.path = path

    def write(self, rec: dict) -> None:
        with open(self.path, "a") as f:
            f.write(json.dumps(rec, default=_jsonable) + "\n")

    def truncate_after(self, step: int) -> None:
        if not self.path.exists():
            return
        keep = []
        for line in self.path.read_text().splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("kind") == "info" or int(r.get("step", 0)) <= step:
                keep.append(line)
        self.path.write_text("".join(x + "\n" for x in keep))


def _jsonable(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, torch.Tensor):
        return o.item() if o.numel() == 1 else o.tolist()
    return str(o)


def write_state(run_dir: Path, **kw) -> None:
    p = run_dir / "state.json"
    cur = json.loads(p.read_text()) if p.exists() else {}
    cur.update(kw, updated=time.strftime("%Y-%m-%dT%H:%M:%S"),
               job_id=os.environ.get("SLURM_JOB_ID"), host=socket.gethostname(),
               restart=int(os.environ.get("SLURM_RESTART_COUNT", "0")))
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(cur, indent=2))
    tmp.replace(p)


def lr_factor(step: int, cfg: SweepConfig) -> float:
    """Linear warmup lr * (step + 1) / warmup, then cosine to 0 at optim.steps."""
    o = cfg.optim
    if step < o.warmup_steps:
        return (step + 1) / max(o.warmup_steps, 1)
    prog = min((step - o.warmup_steps) / max(o.steps - o.warmup_steps, 1), 1.0)
    return 0.5 * (1.0 + math.cos(math.pi * prog))


def param_groups(model: SweepModel, cfg: SweepConfig) -> list[dict]:
    """Model params (no weight decay on biases / 1-d tensors) + the adversary heads' own group at
    a CONSTANT lr."""
    o = cfg.optim
    decay, no_decay, adv = [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("adv_heads."):
            adv.append(p)
        elif p.ndim <= 1 or name.endswith(".bias"):
            no_decay.append(p)
        else:
            decay.append(p)
    # group order = first appearance in named_parameters (LoRA weights come before any bias)
    groups = [{"params": decay, "weight_decay": o.weight_decay, "lr": o.lr},
              {"params": no_decay, "weight_decay": 0.0, "lr": o.lr}]
    first = {id(p): i for i, (_, p) in enumerate(model.named_parameters())}
    groups = sorted((g for g in groups if g["params"]), key=lambda g: first[id(g["params"][0])])
    groups.append({"params": adv, "weight_decay": cfg.adv.weight_decay, "lr": cfg.adv.lr,
                   "lr_fixed": cfg.adv.lr, "adv": True})
    return groups


def set_lrs(opt, cfg: SweepConfig, f: float) -> None:
    """Scheduled lr for every group, except the fixed-lr adversary group."""
    for g in opt.param_groups:
        g["lr"] = g["lr_fixed"] if "lr_fixed" in g else cfg.optim.lr * f


def clip_groups(opt, max_norm: float) -> tuple[float, float]:
    """Clip model params and adversary params SEPARATELY (a fresh adversary's large CE gradient
    must not shrink the model's update). Returns (model grad norm, adversary grad norm)."""
    main = [p for g in opt.param_groups if not g.get("adv") for p in g["params"]]
    adv = [p for g in opt.param_groups if g.get("adv") for p in g["params"]]
    gn = float(torch.nn.utils.clip_grad_norm_(main, max_norm))
    gna = float(torch.nn.utils.clip_grad_norm_(adv, max_norm))
    return gn, gna


def trainable_state(model: SweepModel) -> dict[str, torch.Tensor]:
    sd = {n: p.detach() for n, p in model.named_parameters() if p.requires_grad}
    for n, b in model.named_buffers():
        if not n.startswith("backbone."):
            sd[n] = b.detach()
    return sd


def load_trainable(model: SweepModel, sd: dict) -> None:
    missing = [n for n, p in model.named_parameters() if p.requires_grad and n not in sd]
    if missing:
        raise RuntimeError(f"state lacks {len(missing)} trainable tensors, e.g. {missing[:3]}")
    res = model.load_state_dict(sd, strict=False)
    if res.unexpected_keys:
        raise RuntimeError(f"unexpected keys in state: {res.unexpected_keys[:5]}")


def save_safetensors(sd: dict, path: Path, dtype: str) -> None:
    from safetensors.torch import save_file

    cast = torch.bfloat16 if dtype == "bfloat16" else None
    out = {k: (v.to(cast) if (cast is not None and v.is_floating_point()) else v).contiguous().cpu()
           for k, v in sd.items()}
    tmp = path.with_suffix(".tmp")
    save_file(out, str(tmp))
    tmp.replace(path)


def capture_rng() -> dict:
    st = {"torch": torch.get_rng_state(), "numpy": np.random.get_state(), "python": random.getstate()}
    if torch.cuda.is_available():
        st["cuda"] = torch.cuda.get_rng_state_all()
    return st


def restore_rng(st: dict) -> None:
    torch.set_rng_state(st["torch"].cpu())
    np.random.set_state(st["numpy"])
    random.setstate(st["python"])
    if "cuda" in st and torch.cuda.is_available() and len(st["cuda"]) == torch.cuda.device_count():
        torch.cuda.set_rng_state_all([s.cpu() for s in st["cuda"]])


# ======================================================================================
# the objective
# ======================================================================================


def adv_lambda(cfg: SweepConfig, step: int) -> float:
    """GRL strength: adv.lam x DANN ramp over optim.steps."""
    return float(cfg.adv.lam) * L.dann_ramp(step, cfg.optim.steps)


class AdvQueue:
    """FIFO memory bank (ring buffer) of the last ``size`` detached adversary features and their
    nuisance labels (one label vector per target; -100 = unlabelled). Lives on the features'
    device, allocated on the first push; entries [0, n) are valid. Saved in resume.pt."""

    def __init__(self, size: int, targets):
        self.size, self.targets = int(size), list(targets)
        self.f, self.y, self.ptr, self.n = None, None, 0, 0

    def push(self, f: torch.Tensor, ys: dict) -> None:
        f = f.detach().float()
        if f.shape[0] > self.size:
            f, ys = f[-self.size:], {t: y[-self.size:] for t, y in ys.items()}
        if self.f is None:
            self.f = torch.zeros(self.size, f.shape[1], device=f.device)
            self.y = {t: torch.full((self.size,), -100, dtype=torch.long, device=f.device) for t in self.targets}
        idx = (self.ptr + torch.arange(f.shape[0], device=f.device)) % self.size
        self.f[idx] = f
        for t in self.targets:
            self.y[t][idx] = ys[t].to(self.f.device).long()
        self.ptr = int((self.ptr + f.shape[0]) % self.size)
        self.n = min(self.n + f.shape[0], self.size)

    def get(self, t: str):
        if self.n == 0:
            return None, None
        return self.f[:self.n], self.y[t][:self.n]

    def state_dict(self) -> dict:
        return {"size": self.size, "targets": self.targets, "ptr": self.ptr, "n": self.n,
                "f": None if self.f is None else self.f.cpu(),
                "y": None if self.y is None else {t: v.cpu() for t, v in self.y.items()}}

    def load_state_dict(self, d: dict, device) -> None:
        assert d["size"] == self.size and list(d["targets"]) == self.targets, "adv queue shape mismatch"
        self.ptr, self.n = int(d["ptr"]), int(d["n"])
        self.f = None if d["f"] is None else d["f"].to(device)
        self.y = None if d["y"] is None else {t: v.to(device) for t, v in d["y"].items()}


def _frozen_call(head: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    """head(x) with its parameters detached: gradient reaches x only."""
    return torch.func.functional_call(head, {n: p.detach() for n, p in head.named_parameters()}, (x,))


def adv_loss(model, cfg: SweepConfig, f: torch.Tensor, targets: dict, step: int,
             queue: AdvQueue) -> tuple[torch.Tensor, dict]:
    """Queue adversary on the pooled feature ``f`` (rows labelled by ``targets[t]``). Per target:

      head term     CE(head_t([queue_f ; f.detach()]), [queue_y ; y])  -> head params only
      reversed term CE(frozen head_t(GRL(f, lambda_p)), y)             -> backbone only (batch rows)

    so the head is a multi-class classifier over many conditions while the backbone's reversed
    gradient has the batch-only magnitude. Metrics: ``acc`` = head accuracy on the current batch
    BEFORE it enters the queue, ``qacc`` = accuracy on the queue entries, ``qmaj`` = majority-class
    share of the queue (chance), ``qcls`` = classes present. The batch is enqueued after the loss
    is built."""
    av = cfg.adv
    lam = adv_lambda(cfg, step)
    f = f.float()
    x = L.grad_reverse(f, lam)
    total, mets = None, {"adv/lambda": lam}
    for t in av.targets:
        y = targets[t]
        head = model.adv_heads[t]
        qf, qy = queue.get(t)
        hx = f.detach() if qf is None else torch.cat([qf, f.detach()])
        hy = y if qy is None else torch.cat([qy.to(y.device), y])
        logits = head(hx)
        lh, mh = L.adversarial_ce(logits, hy)
        lr_, mr = L.adversarial_ce(_frozen_call(head, x), y)
        la = lh + lr_
        mets[f"adv/{t}_loss"], mets[f"adv/{t}_rev_loss"] = mh["adv_loss"], mr["adv_loss"]
        mets[f"adv/{t}_acc"] = mr["adv_acc"]
        if qf is not None:
            with torch.no_grad():
                ql, keep = logits[:qf.shape[0]], qy != -100
                if bool(keep.any()):
                    yy = qy[keep]
                    mets[f"adv/{t}_qacc"] = float((ql[keep].argmax(1) == yy).float().mean())
                    cnt = torch.bincount(yy)
                    mets[f"adv/{t}_qmaj"] = float(cnt.max()) / float(yy.numel())
                    mets[f"adv/{t}_qcls"] = int((cnt > 0).sum())
        total = av.weight * la if total is None else total + av.weight * la
    queue.push(f.detach(), {t: targets[t] for t in av.targets})
    mets["adv/queue_n"] = queue.n
    return total, mets


class Objective:
    """multi-tau grid InfoNCE on the projected CLS + the queue adversary on the raw CLS."""

    def __init__(self, cfg: SweepConfig, model: SweepModel, device):
        self.cfg, self.model, self.device = cfg, model, device
        self.adv_queue = AdvQueue(cfg.adv.queue, cfg.adv.targets)

    def autocast(self):
        return torch.autocast(self.device.type, dtype=torch.bfloat16)

    def __call__(self, cls: torch.Tensor, meta: dict) -> tuple[torch.Tensor, dict]:
        obj = self.cfg.objective
        C, T = meta["n_cond"], meta["n_tiles"]
        with self.autocast():
            z = self.model.projector(cls)
        with torch.autocast(self.device.type, enabled=False):
            loss, m = L.multi_tau(lambda t_: L.infonce_grid(z, C, T, t_), obj.taus, obj.tau_weights)
            mets = {f"h0_cls/{k}": v for k, v in m.items()}
            mets["loss_primary"] = float(loss.detach())
            la, ma = adv_loss(self.model, self.cfg, cls, meta["adv_targets"], meta["step"], self.adv_queue)
            total = loss + la
            mets.update(ma)
        mets["loss"] = float(total.detach())
        return total, mets


# ======================================================================================
# the split (shared by training, the dry run and post-hoc scoring)
# ======================================================================================


def resolve_split(cfg: SweepConfig) -> dict:
    """Everything the run's data split depends on, resolved exactly as training does (CPU only)."""
    train_conds, held = resolve_conditions(cfg)
    labels = load_group_labels(cfg.eval.labels)
    held_mask = heldout_location_mask(cfg, labels)
    eval_conds = pick_eval_conditions(cfg)
    eval_cores, cores_in_train = pick_eval_cores(cfg)
    eval_locs = eval_locations(labels, eval_cores, cfg.eval.locations_per_core, cfg.eval.seed)
    held_keys = {c.key for c in held}
    return dict(train_conds=train_conds, held=held, labels=labels, held_mask=held_mask,
                loc_pool=train_location_pool(held_mask), eval_conds=eval_conds, eval_cores=eval_cores,
                cores_in_train=cores_in_train, eval_locs=eval_locs,
                heldaxis=[c.key in held_keys for c in eval_conds])


def describe_split(cfg: SweepConfig, sp: dict | None = None) -> dict:
    sp = sp or resolve_split(cfg)
    lab, pool = sp["labels"], sp["loc_pool"]
    return {"run_name": run_name(cfg), "hash": config_hash(cfg), "backbone": cfg.backbone.name,
            "seed": cfg.run.seed, "train_conditions": len(sp["train_conds"]),
            "heldout_conditions": len(sp["held"]), "heldout_cores": [int(c) for c in cfg.batch.heldout_cores],
            "heldout_locations": int(sp["held_mask"].sum()),
            "train_locations": len(lab) if pool is None else int(len(pool)),
            "eval_conditions": len(sp["eval_conds"]), "eval_cores": sp["eval_cores"],
            "eval_locations": int(len(sp["eval_locs"])), "eval_cores_in_train": sp["cores_in_train"],
            "steps": cfg.optim.steps, "stop_step": cfg.optim.stop_step or cfg.optim.steps}


def build_evaluator(cfg: SweepConfig, model, device, sp: dict, run_dir: Path) -> Evaluator:
    et = eval_tile_set(cfg, sp["eval_conds"], sp["eval_locs"])
    gp = guard_eval_path(cfg)
    return Evaluator(cfg, device, et, sp["eval_conds"], sp["eval_locs"], sp["labels"], run_dir,
                     heldaxis=sp["heldaxis"], cores_in_train=sp["cores_in_train"],
                     guard=GuardEval(cfg, device, gp) if gp else None)


# ======================================================================================
# train
# ======================================================================================


def train(cfg: SweepConfig, run_dir: Path) -> int:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "ckpt").mkdir(exist_ok=True)
    log = JsonlLog(run_dir / "metrics.jsonl")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for s in (signal.SIGUSR1, signal.SIGTERM):
        signal.signal(s, _on_signal)

    torch.manual_seed(cfg.run.seed)
    np.random.seed(cfg.run.seed)
    random.seed(cfg.run.seed)

    sp = resolve_split(cfg)
    train_conds, held, held_mask = sp["train_conds"], sp["held"], sp["held_mask"]
    held_mask_t = torch.as_tensor(held_mask)

    model = SweepModel(cfg).to(device)
    model.train()
    summ = model.trainable_summary()
    print(f"[train] {run_dir.name}: {json.dumps(summ)}", flush=True)
    print(f"[train] {json.dumps(describe_split(cfg, sp))}", flush=True)

    groups = param_groups(model, cfg)
    opt = torch.optim.AdamW(groups, lr=cfg.optim.lr, betas=(cfg.optim.beta1, cfg.optim.beta2))
    objective = Objective(cfg, model, device)

    step = 0
    resume = run_dir / "resume.pt"
    if resume.exists():
        blob = torch.load(resume, map_location="cpu", weights_only=False)
        if blob["hash"] != config_hash(cfg):
            raise RuntimeError(f"resume.pt hash {blob['hash']} != config {config_hash(cfg)}")
        load_trainable(model, blob["trainable"])
        opt.load_state_dict(blob["optimizer"])
        restore_rng(blob["rng"])
        objective.adv_queue.load_state_dict(blob["adv_queue"], device)
        step = int(blob["step"])
        log.truncate_after(step)
        print(f"[train] RESUMED at step {step} from {resume} "
              f"(adv queue {objective.adv_queue.n}/{objective.adv_queue.size})", flush=True)

    END = cfg.optim.stop_step or cfg.optim.steps
    if step >= END:
        write_state(run_dir, status="done", step=step)
        (run_dir / "DONE").write_text(f"step={step}\n")
        return 0

    log.write({"kind": "info", "step": step, "event": "start", "params": summ,
               "train_conditions": [c.key for c in train_conds],
               "heldout_conditions": [c.key for c in held], "heldout_cores": list(cfg.batch.heldout_cores),
               "n_heldout_locations": int(held_mask.sum()), "eval_cores": sp["eval_cores"],
               "eval_cores_in_train": sp["cores_in_train"], "hash": config_hash(cfg),
               "job_id": os.environ.get("SLURM_JOB_ID"), "host": socket.gethostname()})
    write_state(run_dir, status="running", step=step, steps=cfg.optim.steps, error=None, signal=None)

    t0 = time.time()
    evaluator = build_evaluator(cfg, model, device, sp, run_dir)
    print(f"[train] eval set {evaluator.eval_tiles.shape} in {time.time() - t0:.0f}s", flush=True)
    if step == 0:
        evaluator.ensure_base(model)
        if cfg.eval.at_start:
            rec = evaluator(model, 0)
            log.write(rec)
            print(f"[eval] step 0 {_short(rec)}", flush=True)
    elif evaluator.ret_tiles is not None and evaluator.base_ret is None:
        print("[train] WARNING: no cached base retention embeddings; retention metrics off", flush=True)

    loader = build_train_loader(cfg, train_conds, sp["loc_pool"], start_step=step)
    lab_table = scanner_stain_labels(train_conds)
    allowed = set(range(len(train_conds)))
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    win_t, win_n = time.time(), 0
    t_data = time.time()
    for batch in loader:
        if step >= END:
            break
        data_s = time.time() - t_data
        t_step = time.time()
        assert_grid_batch(batch, allowed_conditions=allowed)
        if bool(held_mask_t[batch["tile_idx"].reshape(-1)].any()):
            raise AssertionError("a HELD-OUT core location reached a training batch")
        C, T = int(batch["n_cond"]), int(batch["n_tiles"])
        f = lr_factor(step, cfg)
        set_lrs(opt, cfg, f)
        x = model.prep(to_float01(batch["image"].to(device, non_blocking=True)))
        cond_flat = batch["cond_idx"].reshape(-1).numpy()
        meta = {"n_cond": C, "n_tiles": T, "step": step,
                "adv_targets": {t: torch.as_tensor(lab_table[t][cond_flat], device=device)
                                for t in cfg.adv.targets}}
        with objective.autocast():
            cls = model.cls(x)
        loss, mets = objective(cls, meta)
        loss.backward()
        gn, gn_adv = clip_groups(opt, cfg.optim.grad_clip) if cfg.optim.grad_clip else (None, None)
        mets["adv/grad_norm"] = gn_adv
        opt.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        dt = time.time() - t_step
        win_n += C * T

        if step % cfg.run.log_every == 0 or step == END:
            now = time.time()
            rec = {"kind": "train", "step": step, "lr": cfg.optim.lr * f, "grad_norm": gn,
                   "step_s": dt, "data_wait_s": data_s, "imgs_per_s": win_n / max(now - win_t, 1e-9),
                   **mets}
            if device.type == "cuda":
                rec["mem_alloc_gib"] = torch.cuda.max_memory_allocated(device) / 2**30
            log.write(rec)
            win_t, win_n = now, 0
            print(f"[train] step {step} loss {mets['loss']:.4f} top1 {mets['h0_cls/top1']:.3f} "
                  f"lr {cfg.optim.lr * f:.2e} {dt:.2f}s/step", flush=True)

        sched = cfg.ckpt.schedule
        final = step == END
        if ((step in sched) if sched else (step % cfg.ckpt.every == 0)) or final:
            save_ckpt(model, run_dir, step, cfg)
        es = cfg.eval.schedule
        if ((step in es) if es else (step % cfg.eval.every == 0)) or final:
            rec = evaluator(model, step)
            log.write(rec)
            print(f"[eval] step {step} {_short(rec)}", flush=True)
        if step % cfg.ckpt.resume_every == 0 or final or _STOP["flag"]:
            save_resume(model, opt, run_dir, step, cfg, objective.adv_queue)
            write_state(run_dir, status="running", step=step, steps=cfg.optim.steps)
        if _STOP["flag"] and not final:
            write_state(run_dir, status="preempted", step=step, signal=_STOP["sig"])
            print(f"[train] exiting on signal at step {step} (resume state saved)", flush=True)
            return 143
        t_data = time.time()

    write_state(run_dir, status="done", step=step)
    (run_dir / "DONE").write_text(f"step={step}\n")
    print(f"[train] done {run_dir}", flush=True)
    return 0


def _short(rec: dict) -> str:
    keys = [k for k in rec if k.endswith(("ret_top1_xscanner", "croma_median", "croma_ltm10"))]
    g = f" guard={rec['guard_lin_bacc']:.4f}" if "guard_lin_bacc" in rec else ""
    return (f"primary={rec['primary']:.4f}+-{rec['primary_se']:.4f}{g} "
            + " ".join(f"{k}={rec[k]:.4f}" for k in keys) + f" ({rec.get('eval_s', 0):.0f}s)")


def save_ckpt(model, run_dir: Path, step: int, cfg: SweepConfig) -> None:
    d = run_dir / "ckpt"
    save_safetensors(trainable_state(model), d / f"step_{step:07d}.safetensors", cfg.ckpt.dtype)
    if cfg.ckpt.keep_last > 0:
        keep = {f"step_{int(x):07d}.safetensors" for x in cfg.ckpt.schedule}  # scheduled steps are never pruned
        old = [p for p in sorted(d.glob("step_*.safetensors"))[:-cfg.ckpt.keep_last] if p.name not in keep]
        for p in old:
            p.unlink(missing_ok=True)


def save_resume(model, opt, run_dir: Path, step: int, cfg: SweepConfig, adv_queue: AdvQueue) -> None:
    blob = {"step": step, "hash": config_hash(cfg), "adv_queue": adv_queue.state_dict(),
            "trainable": {k: v.cpu() for k, v in trainable_state(model).items()},
            "optimizer": opt.state_dict(), "rng": capture_rng()}
    tmp = run_dir / "resume.pt.tmp"
    torch.save(blob, tmp)
    tmp.replace(run_dir / "resume.pt")


# ======================================================================================
# CLI
# ======================================================================================


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None, help="YAML/JSON config (may use base:)")
    ap.add_argument("--run-dir", type=Path, default=None,
                    help="existing run dir: train from its stored config.json (resumes)")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="run dir to create (default <runs root>/<sweep>/<run name>)")
    ap.add_argument("--sweep", default="dev")
    ap.add_argument("--print-name", action="store_true", help="print run name + hash and exit")
    ap.add_argument("--dry-run", action="store_true",
                    help="resolve the config and the data split (CPU, no model, no GPU), print, exit")
    ap.add_argument("overrides", nargs="*", help="key.sub=value overrides")
    a = ap.parse_args(argv)
    if a.run_dir is not None:
        if a.overrides:
            raise SystemExit("--run-dir trains the stored config; overrides are not allowed")
        cfg = load_dumped(a.run_dir / "config.json")
        run_dir = a.run_dir
    else:
        cfg = load_config(a.config, a.overrides)
        if a.print_name:
            print(run_name(cfg), config_hash(cfg))
            return 0
        run_dir = a.out_dir or (runs_root() / a.sweep / run_name(cfg))
    if a.dry_run:
        print(json.dumps({**describe_split(cfg), "run_dir": str(run_dir)}, indent=2))
        return 0
    if a.run_dir is None:
        run_dir.mkdir(parents=True, exist_ok=True)
        cp = run_dir / "config.json"
        if cp.exists() and config_hash(load_dumped(cp)) != config_hash(cfg):
            raise SystemExit(f"{cp} holds a different config; refusing to mix runs")
        dump(cfg, cp)
    print(f"[train] run {run_dir} hash {config_hash(cfg)}", flush=True)
    try:
        rc = train(cfg, run_dir)
    except torch.cuda.OutOfMemoryError:
        write_state(run_dir, status="failed", error="OOM")
        raise
    except Exception as exc:
        write_state(run_dir, status="failed", error=f"{type(exc).__name__}: {exc}"[:500])
        raise
    return rc


if __name__ == "__main__":
    sys.exit(main())
