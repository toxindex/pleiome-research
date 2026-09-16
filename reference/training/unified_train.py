"""
B-FULL trainer — co-train the UnifiedModel on PREDICT (D-MPNN -> property heads) + GENERATE (property slots
-> SELFIES decoder), sharing the property-slot representation. Graph mode required (needs both the graph for
the D-MPNN predict path and the SELFIES tokens for the generation target).

  PYTHONPATH=./ CM_PROPS=bigrun/props_full CM_MIN_SUPPORT=10 CM_GRAPH_DIR=cache/graphs_ik CM_D_MODEL=512 \
    CM_LAYERS=5 CM_STEPS=120000 UM_GEN_LAYERS=6 UM_LAMBDA=1.0 CM_LOGDIR=cache/unified \
    .venv/bin/python property_multitask/unified_train.py
"""
import argparse, functools, os, numpy as np, torch
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import roc_auc_score
import pyarrow.parquet as pq
import cvae.tokenizer
from property_multitask.train_typed import TypedRecords, _to_dev
from property_multitask.typed_model import TYPE_IDS
from property_multitask.unified_model import UnifiedModel, SOS, EOS, PADT
from property_multitask.dmpnn_encoder import MolGraph, collate_graphs

MAXLEN = 128


class UnifiedRecords(TypedRecords):
    """Returns (graph_struct, selfies_tokens, pid, typ, val) — graph for predict, tokens for generate."""
    def __getitem__(self, i):
        c = int(self.keep[i])
        fa, bonds = self.ik2graph.get(self.all_ikey[c], (torch.zeros(1, 43, dtype=torch.float16), []))
        struct = (fa.float(), bonds)
        toks = torch.from_numpy(self.tok_vals[self.tok_off[c]:self.tok_off[c + 1]][:MAXLEN - 1].astype(np.int64))
        ps, pe = self.pair_off[c], self.pair_off[c + 1]
        pid = torch.from_numpy(self.pair_pid[ps:pe][:self.max_props].astype(np.int64))
        typ = torch.from_numpy(self.pair_typ[ps:pe][:self.max_props].astype(np.int64))
        val = torch.from_numpy(self.pair_val[ps:pe][:self.max_props].copy())
        num = typ == TYPE_IDS["numeric"]
        if num.any():
            m = self.num_mean_arr[pid]; s = self.num_std_arr[pid]
            val = torch.where(num, ((val - m) / s).clamp(-10, 10), val)
        return struct, toks, pid, typ, val


def collate_unified(batch, known_frac=0.5, rand_frac=False):
    P = max(len(p) for _, _, p, *_ in batch); B = len(batch)
    pid = torch.zeros(B, P, dtype=torch.long); typ = torch.zeros(B, P, dtype=torch.long)
    val = torch.zeros(B, P); smask = torch.zeros(B, P, dtype=torch.bool)
    for b, (_, _, p, ty, v) in enumerate(batch):
        n = len(p); pid[b, :n] = p; typ[b, :n] = ty; val[b, :n] = v; smask[b, :n] = True
    structure = collate_graphs([MolGraph(t[0], t[1]) for t, *_ in batch])
    # SELFIES teacher-forcing tensors: inp = [SOS]+toks, tgt = toks+[EOS], padded to MAXLEN
    tin = torch.zeros(B, MAXLEN, dtype=torch.long); ttg = torch.zeros(B, MAXLEN, dtype=torch.long)
    for b, (_, toks, *_) in enumerate(batch):
        L = min(len(toks), MAXLEN - 1)
        tin[b, 0] = SOS
        if L > 0:
            tin[b, 1:1 + L] = toks[:L]
            ttg[b, :L] = toks[:L]
        ttg[b, L] = EOS
    if rand_frac:
        kf = torch.rand(B, 1)                                 # per-example mask ratio ~ U(0,1): train on all context amounts
        vmask = (torch.rand(B, P) < kf) & smask
    else:
        vmask = (torch.rand(B, P) < known_frac) & smask
    return structure, tin, ttg, pid, typ, val, vmask, smask


@torch.no_grad()
def evaluate(model, dl, dev, raw_pids=None):
    model.eval()
    by = {}
    tok_corr = tok_tot = 0
    for structure, tin, ttg, pid, typ, val, vmask, smask in dl:
        structure = _to_dev(structure, dev)
        tin, ttg, pid, typ, val, vmask, smask = (x.to(dev) for x in (tin, ttg, pid, typ, val, vmask, smask))
        predict = smask & ~vmask
        bm = predict & (typ == TYPE_IDS["binary"])
        if bm.any():
            p = torch.sigmoid(model(structure, pid, typ, val, vmask, smask)["binary"])[bm].cpu().numpy()
            y = val[bm].cpu().numpy(); pids = pid[bm].cpu().numpy()
            for pp, yy, dd in zip(pids, y, p):
                by.setdefault(int(pp), ([], []))[0].append(dd); by[int(pp)][1].append(yy)
        # generation teacher-forced token accuracy (a cheap proxy for generation health)
        slots = model._profile_slots(pid, typ, val, smask)
        logits = model.struct_dec(tin, slots, ~smask)
        pr = logits.argmax(-1); m = ttg != PADT
        tok_corr += ((pr == ttg) & m).sum().item(); tok_tot += m.sum().item()
    aucs = []; chem = []; brick = []
    for pp, (pr, yl) in by.items():
        yl = np.array(yl)
        if len(yl) >= 8 and yl.min() != yl.max():
            au = roc_auc_score(yl, np.array(pr)); aucs.append(au)
            if raw_pids is not None:
                (brick if raw_pids[pp] >= 2_000_000 else chem).append(au)
    macro = float(np.mean(aucs)) if aucs else 0.0
    frac80 = float(np.mean(np.array(aucs) >= 0.8)) if aucs else 0.0
    chem_m = float(np.mean(chem)) if chem else 0.0
    brick_m = float(np.mean(brick)) if brick else 0.0
    return macro, frac80, len(aucs), tok_corr / max(tok_tot, 1), chem_m, len(chem), brick_m, len(brick)


@torch.no_grad()
def per_property_auc(model, dl, dev):
    """Per-property binary AUC on val (known_frac from the loader) -> {compact_pid: auc}."""
    model.eval(); by = {}
    for structure, tin, ttg, pid, typ, val, vmask, smask in dl:
        structure = _to_dev(structure, dev)
        pid, typ, val, vmask, smask = (x.to(dev) for x in (pid, typ, val, vmask, smask))
        m = (smask & ~vmask) & (typ == TYPE_IDS["binary"])
        if not m.any():
            continue
        p = torch.sigmoid(model(structure, pid, typ, val, vmask, smask)["binary"])[m].cpu().numpy()
        y = val[m].cpu().numpy(); ps = pid[m].cpu().numpy()
        for a, b, c in zip(ps, p, y):
            by.setdefault(int(a), ([], []))[0].append(b); by[int(a)][1].append(c)
    out = {}
    for pp, (pl, yl) in by.items():
        yl = np.array(yl)
        if len(yl) >= 8 and yl.min() != yl.max():
            out[pp] = roc_auc_score(yl, np.array(pl))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--props", default=os.getenv("CM_PROPS", "bigrun/props_sample"))
    ap.add_argument("--steps", type=int, default=int(os.getenv("CM_STEPS", "120000")))
    ap.add_argument("--batch", type=int, default=int(os.getenv("CM_BATCH", "48")))
    a = ap.parse_args()
    dev = torch.device("cuda:0" if torch.cuda.is_available() and not a.smoke else "cpu")
    tok = cvae.tokenizer.SelfiesPropertyValTokenizer.load("brick/selfies_property_val_tokenizer")
    ds = UnifiedRecords(a.props, max_props=64 if a.smoke else 128)
    d = 64 if a.smoke else int(os.getenv("CM_D_MODEL", "512"))
    L = 2 if a.smoke else int(os.getenv("CM_LAYERS", "5"))
    gl = 2 if a.smoke else int(os.getenv("UM_GEN_LAYERS", "6"))
    lam = float(os.getenv("UM_LAMBDA", "1.0"))
    sc_on = bool(os.getenv("CM_SELFCOND"))            # self-conditioning: train on the model's own predictions
    sc_max = float(os.getenv("CM_SELFCOND_MAX", "0.5"))   # max prob a revealed-context slot is replaced by self-pred
    sc_warm = int(os.getenv("CM_SELFCOND_WARM", str(max(1, a.steps // 2))))  # ramp 0 -> sc_max over these steps
    struct_aux = float(os.getenv("CM_STRUCT_AUX", "0.0"))  # weight of a direct structure-only (kf=0) loss term
    model = UnifiedModel.from_tokenizer(tok, num_properties=ds.num_props, encoder_type="dmpnn",
                                        d_model=d, nhead=4 if a.smoke else 8, struct_layers=L, slot_layers=L,
                                        gen_layers=gl, gen_max_len=MAXLEN).to(dev)
    print(f"UNIFIED model | props={ds.num_props:,} d={d} predict_layers={L} gen_layers={gl} lambda={lam} "
          f"params={sum(p.numel() for p in model.parameters())/1e6:.1f}M", flush=True)
    if os.getenv("UM_INIT"):                                     # warm-start (fine-tune from a checkpoint)
        _sd=torch.load(os.getenv("UM_INIT"), map_location=dev, weights_only=False)["model"]
        _r=model.load_state_dict(_sd, strict=not os.getenv("UM_INIT_LOOSE"))
        print(f"warm-load missing={len(_r.missing_keys)} unexpected={len(_r.unexpected_keys)}", flush=True)
        print(f"warm-started from {os.getenv('UM_INIT')}", flush=True)
    if os.getenv("UM_ENCODER_INIT"):                             # load molpile-pretrained D-MPNN encoder
        _e = torch.load(os.getenv("UM_ENCODER_INIT"), map_location=dev, weights_only=False)
        _e = _e.get("encoder", _e)
        _m = model.encoder.load_state_dict(_e, strict=False)
        print(f"loaded pretrained encoder {os.getenv('UM_ENCODER_INIT')} (missing={len(_m.missing_keys)} unexpected={len(_m.unexpected_keys)})", flush=True)
    bh = model.handlers["binary"]                                # rare-ACTIVE imbalance levers
    bh.pos_weight = float(os.getenv("UM_POS_WEIGHT", "1.0")); bh.focal_gamma = float(os.getenv("UM_FOCAL_GAMMA", "0.0"))
    print(f"imbalance: pos_weight={bh.pos_weight} focal_gamma={bh.focal_gamma}", flush=True)
    la_tau = float(os.getenv("CM_LOGIT_ADJ", "0"))          # logit adjustment: rebalance rare-positive gradient (macro/frac80 lever)
    lsm = float(os.getenv("CM_LABEL_SMOOTH", "0"))
    if la_tau > 0 or lsm > 0:
        _bm = ds.pair_typ == TYPE_IDS["binary"]
        _bp = ds.pair_pid[_bm].astype(np.int64); _bv = ds.pair_val[_bm]
        npos = np.bincount(_bp, weights=(_bv > 0.5), minlength=ds.num_props)
        ntot = np.bincount(_bp, minlength=ds.num_props).astype(np.float64)
        pi = np.clip(npos / np.maximum(ntot, 1), 1e-3, 1 - 1e-3)
        la = np.log(pi / (1 - pi)).astype(np.float32)
        model.handlers["binary"].set_logit_adj(torch.from_numpy(la).to(dev), la_tau)
        model.handlers["binary"].label_smooth = lsm
        print(f"LOGIT ADJUSTMENT tau={la_tau}, label_smooth={lsm}", flush=True)

    stride = int(os.getenv("CM_HOLDOUT_STRIDE", "25"))
    n = len(ds); va_idx = list(range(n))[::stride]
    va_idx = va_idx[:400] if a.smoke else va_idx[:int(os.getenv("CM_VAL_CAP", "6000"))]
    tr_idx = [i for i in range(n) if i % stride != 0]
    if a.smoke:
        tr_idx = tr_idx[:400]
    nw = 0 if ds.graph_dir else 4
    dl = DataLoader(Subset(ds, tr_idx), batch_size=a.batch, shuffle=True, num_workers=nw, drop_last=True,
                    collate_fn=functools.partial(collate_unified, known_frac=0.5, rand_frac=bool(os.getenv("CM_RAND_FRAC"))))
    vdl = DataLoader(Subset(ds, va_idx), batch_size=a.batch, shuffle=False, num_workers=nw,
                     collate_fn=functools.partial(collate_unified, known_frac=0.5))
    vdl0 = DataLoader(Subset(ds, va_idx), batch_size=a.batch, shuffle=False, num_workers=nw,
                      collate_fn=functools.partial(collate_unified, known_frac=0.0))  # structure-only (kf=0) eval
    opt = torch.optim.AdamW(model.parameters(), lr=float(os.getenv("CM_LR", "5e-4")), weight_decay=0.01)
    warmup = int(os.getenv("CM_WARMUP", "3000")); base_lr = float(os.getenv("CM_LR", "5e-4"))
    logdir = os.getenv("CM_LOGDIR", "cache/unified"); os.makedirs(logdir, exist_ok=True)
    eval_every = 6 if a.smoke else int(os.getenv("CM_EVAL_EVERY", "10000"))
    maxs = 8 if a.smoke else a.steps
    if os.getenv("CM_TAIL_REWEIGHT"):                            # Option-1: gentle, clamped, AUC-targeted tail upweighting
        tau = float(os.getenv("CM_TAIL_TAU", "0.85")); beta = float(os.getenv("CM_TAIL_BETA", "2.0"))
        wmax = float(os.getenv("CM_TAIL_WMAX", "3.0")); wunk = float(os.getenv("CM_TAIL_UNKNOWN", "1.3"))
        aucd = per_property_auc(model, vdl, dev)
        w = torch.full((ds.num_props,), wunk)                    # unmeasured (rare) props -> mild boost
        for pp, au in aucd.items():
            deficit = max(0.0, tau - au); w[pp] = min(wmax, 1.0 + beta * deficit / tau)  # strong props -> ~1 (protected)
        model.handlers["binary"].register_buffer("prop_weight", w.to(dev), persistent=False)
        print(f"TAIL-REWEIGHT: measured={len(aucd)} mean_w={float(w.mean()):.3f} "
              f"n(w>=1.5)={int((w >= 1.5).sum())} wmax={wmax} tau={tau} beta={beta}", flush=True)
    best = -1.0; best_chem = -1.0; step = 0
    while step < maxs:
        for structure, tin, ttg, pid, typ, val, vmask, smask in dl:
            if warmup > 0:
                import math as _m
                if step < warmup:
                    _lr = base_lr * (step / max(1, warmup))
                elif os.getenv("CM_LR_DECAY"):
                    _tot = int(os.getenv("CM_STEPS", "300000")); _min = float(os.getenv("CM_MIN_LR", "1e-5"))
                    _prog = min(1.0, (step - warmup) / max(1, _tot - warmup))
                    _lr = _min + 0.5 * (base_lr - _min) * (1 + _m.cos(_m.pi * _prog))
                else:
                    _lr = base_lr
                for g in opt.param_groups:
                    g["lr"] = _lr
            structure = _to_dev(structure, dev)
            tin, ttg, pid, typ, val, vmask, smask = (x.to(dev) for x in (tin, ttg, pid, typ, val, vmask, smask))
            model.train()
            val_in = val
            if sc_on:
                sc_p = sc_max * min(1.0, step / max(1, sc_warm))
                if sc_p > 0:
                    with torch.no_grad():                              # structure-only pass -> self-predictions
                        pre = model(structure, pid, typ, val, torch.zeros_like(vmask), smask)
                    vs = val.clone()
                    bm_ = typ == TYPE_IDS["binary"]
                    if bm_.any(): vs = torch.where(bm_, (torch.sigmoid(pre["binary"]) > 0.5).float(), vs)
                    nm_ = typ == TYPE_IDS["numeric"]
                    if nm_.any(): vs = torch.where(nm_, pre["numeric"]["mu"], vs)
                    cm_ = typ == TYPE_IDS["categorical"]
                    if cm_.any(): vs = torch.where(cm_, pre["categorical"].argmax(-1).float(), vs)
                    replace = (torch.rand_like(val) < sc_p) & vmask    # swap some revealed context for self-pred
                    val_in = torch.where(replace, vs, val)
            preds = model(structure, pid, typ, val_in, vmask, smask)
            lp, parts = model.compute_loss(preds, pid, typ, val, vmask, smask)  # targets = TRUE val on unrevealed
            lg = model.generation_loss(tin, ttg, pid, typ, val, smask)
            loss = lp + lam * lg
            if struct_aux > 0:                                  # direct zero-context training (predict ALL from structure)
                empty = torch.zeros_like(vmask)
                preds0 = model(structure, pid, typ, val, empty, smask)
                lp0, _ = model.compute_loss(preds0, pid, typ, val, empty, smask)
                loss = loss + struct_aux * lp0
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            if step % (1 if a.smoke else 500) == 0:
                print(f"step={step} lr={opt.param_groups[0]['lr']:.2e} L_pred={float(lp):.4f} "
                      f"L_gen={float(lg):.4f}", flush=True)
            if step > 0 and step % eval_every == 0:
                macro, frac80, npr, tok_acc, chem_m, chem_n, brick_m, brick_n = evaluate(model, vdl, dev, np.asarray(ds.raw_pids))
                print(f"VAL step={step} bin_macro={macro:.4f} frac80={frac80:.3f} nprops={npr} "
                      f"gen_tok_acc={tok_acc:.3f} CHEM={chem_m:.4f}(n={chem_n}) BRICK={brick_m:.4f}(n={brick_n})", flush=True)
                so = evaluate(model, vdl0, dev, np.asarray(ds.raw_pids))
                print(f"VAL step={step} STRUCT_ONLY bin_macro={so[0]:.4f} frac80={so[1]:.3f} nprops={so[2]}", flush=True)
                ck = {"model": model.state_dict(), "num_props": ds.num_props, "step": step,
                      "num_mean_arr": ds.num_mean_arr, "num_std_arr": ds.num_std_arr}
                torch.save(ck, os.path.join(logdir, "last.pt"))
                if macro > best and step >= warmup:
                    best = macro; torch.save(ck, os.path.join(logdir, "best.pt"))
                if chem_m > best_chem and step >= warmup:
                    best_chem = chem_m; torch.save(ck, os.path.join(logdir, "best_chem.pt"))
            step += 1
            if step >= maxs:
                break
    print(f"DONE best_bin_macro={best:.4f}", flush=True)


if __name__ == "__main__":
    main()
