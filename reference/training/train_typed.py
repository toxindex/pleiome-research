"""
Trainer for the generic TypedPropertyModel (property_multitask/typed_model.py), wired to the
real STAGE-2 typed-record shards (bigrun/normalize_properties.py output):
  props_*.parquet rows = {tokens: list<int16>, pairs: list<(property_id:int32, type:int8, value:float32)>}

Masked-value prediction across types: each real slot is randomly KNOWN (value_mask=1, value fed in)
or PREDICT (value_mask=0, loss target). Reuses the per-type heads' objectives from typed_model.py.

  PYTHONPATH=./ python property_multitask/train_typed.py --smoke        # CPU smoke test on the sample
  PYTHONPATH=./ CM_PROPS=./bigrun/props_sample python property_multitask/train_typed.py
"""
import argparse, glob, os, numpy as np, torch
from torch.utils.data import Dataset, DataLoader
import pyarrow.parquet as pq
import cvae.tokenizer
from property_multitask.typed_model import TypedPropertyModel, TYPE_IDS

MASK_PAD = 0


class TypedRecords(Dataset):
    def __init__(self, shard_dir, max_props=128):
        self.max_props = max_props
        # COMPACT loader: store the whole pool as flat CSR-style numpy arrays (tokens + pairs), NOT Python
        # dicts. The old load-everything .to_pylist() blew up ~160x (a struct pair-dict ~240B x 250M pairs
        # ~= 60GB) and OOM'd at 124GB on the full 64-shard pool. Packed arrays are ~3GB -> the full 4M
        # compounds fit in RAM easily (props_full is only 767MB on disk). CM_MAX_SHARDS caps shards
        # (0 = all); a shard subset spans all properties for a quick test.
        _shards = sorted(glob.glob(os.path.join(shard_dir, "props_*.parquet")))
        _cap = int(os.getenv("CM_MAX_SHARDS", "0"))
        if _cap > 0:
            _shards = _shards[:_cap]
        self.graph_dir = os.getenv("CM_GRAPH_DIR")          # GRAPH mode: D-MPNN encoder over original SMILES
        tok_vals, tok_lens = [], []
        p_pid, p_typ, p_val, p_lens = [], [], [], []
        ikeys = []
        for f in _shards:
            _cols = ["tokens", "pairs"] + (["inchikey"] if self.graph_dir else [])
            t = pq.read_table(f, columns=_cols)
            if self.graph_dir:
                ikeys.extend(t.column("inchikey").to_pylist())
            tok = t.column("tokens").combine_chunks()
            pr = t.column("pairs").combine_chunks()
            tok_vals.append(tok.values.to_numpy(zero_copy_only=False).astype(np.int16))
            tok_lens.append(np.diff(tok.offsets.to_numpy()))
            sv = pr.values                                      # StructArray over all pairs in the shard
            p_pid.append(sv.field("property_id").to_numpy(zero_copy_only=False).astype(np.int32))
            p_typ.append(sv.field("type").to_numpy(zero_copy_only=False).astype(np.int8))
            p_val.append(sv.field("value").to_numpy(zero_copy_only=False).astype(np.float32))
            p_lens.append(np.diff(pr.offsets.to_numpy()))
        self.tok_vals = np.concatenate(tok_vals)
        tok_lens = np.concatenate(tok_lens).astype(np.int64)
        self.tok_off = np.zeros(len(tok_lens) + 1, np.int64); np.cumsum(tok_lens, out=self.tok_off[1:])
        pid = np.concatenate(p_pid); typ = np.concatenate(p_typ); val = np.concatenate(p_val)
        plens = np.concatenate(p_lens).astype(np.int64)
        n_comp = len(plens)
        # DROP missing-value sentinels: brick numeric columns encode "not measured" as ~1e31
        # (0.87%% of numeric pairs; one property is entirely sentinel). They poison per-property
        # mean/std and inject garbage targets. Remove the pair entirely (|numeric| > 1e12 == missing).
        _NUMT = TYPE_IDS["numeric"]
        _sent = (typ == _NUMT) & (np.abs(val) > 1e12)
        if _sent.any():
            _keep = ~_sent
            _cop = np.repeat(np.arange(n_comp), plens)
            plens = np.bincount(_cop[_keep], minlength=n_comp).astype(np.int64)
            pid = pid[_keep]; typ = typ[_keep]; val = val[_keep]
            print(f"dropped {int(_sent.sum()):,} numeric missing-value sentinels (>1e12)", flush=True)
        # MIN-SUPPORT filter (numpy): the full pool has ~1M properties, most with 1-few compounds. Keep
        # properties with >= CM_MIN_SUPPORT pairs, remap to a compact id range, drop dropped pairs +
        # any compound left with no pairs. Default 0 = no filter (small pools).
        min_support = int(os.getenv("CM_MIN_SUPPORT", "0"))
        if min_support > 0:
            cnt = np.bincount(pid)
            keep_props = np.where(cnt >= min_support)[0]
            self.raw_pids = keep_props                            # compact_id -> raw property_id (for factors)
            remap = np.full(cnt.shape[0], -1, np.int64); remap[keep_props] = np.arange(len(keep_props))
            new_pid = remap[pid]
            mask = new_pid >= 0
            comp_of_pair = np.repeat(np.arange(n_comp), plens)  # compound index for each flat pair
            plens = np.bincount(comp_of_pair[mask], minlength=n_comp).astype(np.int64)
            pid = new_pid[mask].astype(np.int32); typ = typ[mask]; val = val[mask]
            print(f"min-support(>={min_support}): {cnt.shape[0]:,}->{len(keep_props):,} props", flush=True)
        self.pair_off = np.zeros(n_comp + 1, np.int64); np.cumsum(plens, out=self.pair_off[1:])
        self.pair_pid, self.pair_typ, self.pair_val = pid, typ, val
        self.keep = np.where(plens > 0)[0]                      # compounds with >=1 kept pair
        self.num_props = int(pid.max()) + 1 if len(pid) else 1
        if not hasattr(self, "raw_pids"): self.raw_pids = np.arange(self.num_props)
        # PER-PROPERTY numeric normalization (global fallback for sparse properties). Global standardization
        # gave held-out numeric RMSE ~1.0 (= mean-prediction baseline); per-property z-scoring removes the
        # offset so the head only predicts within-property SHAPE (learnable). Computed via bincount moments.
        NUM = TYPE_IDS["numeric"]
        nmask = typ == NUM
        num_pid = pid[nmask].astype(np.int64); num_v = val[nmask].astype(np.float64)
        fin = np.isfinite(num_v)                               # drop any residual nan/inf
        num_pid = num_pid[fin]; num_v = num_v[fin]
        gmean = float(num_v.mean()) if len(num_v) else 0.0
        gstd = float(max(num_v.std(), 1e-3)) if len(num_v) else 1.0
        self.num_mean_arr = torch.full((self.num_props,), gmean)
        self.num_std_arr = torch.full((self.num_props,), gstd)
        if len(num_v):
            cnts = np.bincount(num_pid, minlength=self.num_props).astype(np.float64)
            sums = np.bincount(num_pid, weights=num_v, minlength=self.num_props)
            sqs = np.bincount(num_pid, weights=num_v * num_v, minlength=self.num_props)
            ok = cnts >= 5                                      # per-property stats only when enough samples
            means = sums / np.maximum(cnts, 1.0)
            stds = np.sqrt(np.maximum(sqs / np.maximum(cnts, 1.0) - means ** 2, 1e-6))
            self.num_mean_arr[ok] = torch.tensor(means[ok], dtype=torch.float)
            self.num_std_arr[ok] = torch.tensor(np.maximum(stds[ok], 1e-3), dtype=torch.float)
        self.num_mean, self.num_std = gmean, gstd               # globals kept for logging/reference
        print(f"loaded compact: {len(self.keep):,} compounds, {len(pid):,} pairs, {self.num_props:,} props "
              f"(numeric global {gmean:.3f}/{gstd:.3f})", flush=True)
        if self.graph_dir:                                      # load inchikey->graph (original-SMILES features)
            self.all_ikey = ikeys                               # per-compound inchikey (pre-keep order)
            self.ik2graph = {}
            for gf in sorted(glob.glob(os.path.join(self.graph_dir, "shard_*.pt"))):
                self.ik2graph.update(torch.load(gf, weights_only=False))
            print(f"GRAPH mode: {len(self.ik2graph):,} precomputed graphs loaded", flush=True)

    def __len__(self): return len(self.keep)

    def __getitem__(self, i):
        c = int(self.keep[i])
        if self.graph_dir:                                      # GRAPH: return (f_atoms, bonds) from real SMILES
            fa, bonds = self.ik2graph.get(self.all_ikey[c], (torch.zeros(1, 43, dtype=torch.float16), []))
            struct = (fa.float(), bonds)
        else:
            # Truncate SELFIES to the encoder's positional-embedding length (SelfiesEncoder max_len=120).
            struct = torch.from_numpy(self.tok_vals[self.tok_off[c]:self.tok_off[c + 1]][:120].astype(np.int64))
        ps, pe = self.pair_off[c], self.pair_off[c + 1]
        pid = torch.from_numpy(self.pair_pid[ps:pe][: self.max_props].astype(np.int64))
        typ = torch.from_numpy(self.pair_typ[ps:pe][: self.max_props].astype(np.int64))
        val = torch.from_numpy(self.pair_val[ps:pe][: self.max_props].copy())
        num = typ == TYPE_IDS["numeric"]                    # per-property z-score for numeric slots only
        if num.any():
            m = self.num_mean_arr[pid]; s = self.num_std_arr[pid]
            val = torch.where(num, ((val - m) / s).clamp(-10, 10), val)
        return struct, pid, typ, val


def _to_dev(x, dev):
    """Move a batch structure to device: tensor OR a graph-batch dict (GRAPH mode)."""
    if isinstance(x, dict):
        return {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in x.items()}
    return x.to(dev)


def collate(batch, known_frac=0.5):
    P = max(len(p) for _, p, *_ in batch); B = len(batch)
    is_graph = isinstance(batch[0][0], tuple)                 # struct is (f_atoms, bonds) in GRAPH mode
    pid = torch.zeros(B, P, dtype=torch.long); typ = torch.zeros(B, P, dtype=torch.long)
    val = torch.zeros(B, P); slot_mask = torch.zeros(B, P, dtype=torch.bool)
    for b, (_, p, ty, v) in enumerate(batch):
        n = len(p); pid[b, :n] = p; typ[b, :n] = ty; val[b, :n] = v; slot_mask[b, :n] = True
    if is_graph:
        from property_multitask.dmpnn_encoder import MolGraph, collate_graphs
        structure = collate_graphs([MolGraph(t[0], t[1]) for t, *_ in batch])   # graph-batch dict
    else:
        L = max(len(t) for t, *_ in batch)
        structure = torch.zeros(B, L, dtype=torch.long)
        for b, (t, *_) in enumerate(batch):
            structure[b, :len(t)] = t
    value_mask = (torch.rand(B, P) < known_frac) & slot_mask
    return structure, pid, typ, val, value_mask, slot_mask


def conflict_averse_backward(group_losses, shared_params, private_params, opt):
    """CONFLICT-AVERSE (PCGrad) gradient over property-support TIERS. Steepest descent on the summed loss
    lets common-property gradients trample rare ones (destructive interference -> tail degrades, long
    training collapses). Instead: get each tier's gradient on the SHARED trunk, and where two tiers CONFLICT
    (negative inner product) project one off the other so no tier is sacrificed. Private per-property heads
    train on their own (summed) gradient -- interference is a SHARED-representation problem. ~K+1 backward
    cost. Sets .grad in place for opt.step()."""
    opt.zero_grad()
    total = sum(group_losses)
    # private params: normal summed gradient
    if private_params:
        gp = torch.autograd.grad(total, private_params, retain_graph=True, allow_unused=True)
        for p, g in zip(private_params, gp):
            if g is not None:
                p.grad = g
    # shared params: per-tier flat grads -> PCGrad projection
    flats = []
    for i, L in enumerate(group_losses):
        g = torch.autograd.grad(L, shared_params, retain_graph=(i < len(group_losses) - 1), allow_unused=True)
        flats.append(torch.cat([(gi if gi is not None else torch.zeros_like(p)).reshape(-1)
                                for gi, p in zip(g, shared_params)]))
    proj = [f.clone() for f in flats]
    for i in range(len(flats)):
        for j in range(len(flats)):
            if i == j:
                continue
            dot = torch.dot(proj[i], flats[j])
            if dot < 0:
                proj[i] = proj[i] - dot / (torch.dot(flats[j], flats[j]) + 1e-12) * flats[j]
    combined = torch.stack(proj).sum(0)
    idx = 0
    for p in shared_params:
        n = p.numel(); p.grad = combined[idx:idx + n].view_as(p); idx += n


@torch.no_grad()
def evaluate(model, val_dl, dev):
    """Held-out per-type metrics on masked (predict) slots. Reports pooled binary AUC (bin_auc) AND
    per-property MACRO AUC (bin_macro) — the SELECTION metric. Pooled AUC anti-correlates with the goal
    (long training inflates pooled via a cross-property base-rate crutch while per-property discrimination
    DEGRADES); macro is base-rate-invariant, so best.pt tracks what we actually want."""
    from sklearn.metrics import roc_auc_score
    from collections import defaultdict
    import numpy as _np
    model.eval()
    bp, by, np_, ny = [], [], [], []
    bypid = defaultdict(lambda: ([], []))
    for selfies, pid, typ, val, vmask, smask in val_dl:
        selfies = _to_dev(selfies, dev)
        pid, typ, val, vmask, smask = (x.to(dev) for x in (pid, typ, val, vmask, smask))
        preds = model(selfies, pid, typ, val, vmask, smask)
        predict = smask & ~vmask
        bm = predict & (typ == TYPE_IDS["binary"]); nm = predict & (typ == TYPE_IDS["numeric"])
        if bm.any():
            pb = torch.sigmoid(preds["binary"][bm]).cpu(); yb = val[bm].cpu(); ib = pid[bm].cpu()
            bp.append(pb); by.append(yb)
            for pp, dd, yy in zip(ib.tolist(), pb.tolist(), yb.tolist()):
                bypid[pp][0].append(dd); bypid[pp][1].append(yy)
        if nm.any(): np_.append(preds["numeric"]["mu"][nm].cpu()); ny.append(val[nm].cpu())
    model.train()
    out = {}
    if bp:
        p = torch.cat(bp).numpy(); y = torch.cat(by).numpy()
        out["bin_auc"] = float(roc_auc_score(y, p)) if len(_np.unique(y)) > 1 else float("nan"); out["bin_n"] = len(y)
        macros = []
        for preds_l, ys_l in bypid.values():
            yl = _np.array(ys_l)
            if len(yl) >= 8 and yl.min() != yl.max():
                macros.append(roc_auc_score(yl, _np.array(preds_l)))
        out["bin_macro"] = float(_np.mean(macros)) if macros else float("nan"); out["macro_np"] = len(macros)
    if np_:
        p = torch.cat(np_).numpy(); y = torch.cat(ny).numpy()
        out["num_rmse"] = float(_np.sqrt(((p - y) ** 2).mean())); out["num_n"] = len(y)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--props", default=os.getenv("CM_PROPS", "./bigrun/props_sample"))
    ap.add_argument("--steps", type=int, default=int(os.getenv("CM_STEPS", "40000")))
    ap.add_argument("--batch", type=int, default=int(os.getenv("CM_BATCH", "64")))
    ap.add_argument("--d_model", type=int, default=int(os.getenv("CM_D_MODEL", "512")))
    ap.add_argument("--layers", type=int, default=int(os.getenv("CM_LAYERS", "6")))
    ap.add_argument("--pretrained", default=os.getenv("CM_PRETRAINED"))
    a = ap.parse_args()
    dev = torch.device("cuda:0" if torch.cuda.is_available() and not a.smoke else "cpu")
    tok = cvae.tokenizer.SelfiesPropertyValTokenizer.load("brick/selfies_property_val_tokenizer")

    ds = TypedRecords(a.props, max_props=64 if a.smoke else 128)
    print(f"loaded {len(ds)} compounds | num_properties {ds.num_props} | dev {dev}", flush=True)
    d = 64 if a.smoke else a.d_model
    sl = 2 if a.smoke else a.layers
    enc_type = "dmpnn" if os.getenv("CM_GRAPH_DIR") else "selfies"
    model = TypedPropertyModel.from_tokenizer(
        tok, num_properties=ds.num_props, extra_vocab=1 if a.pretrained else 0,
        d_model=d, nhead=4 if a.smoke else 8, struct_layers=2 if a.smoke else sl, slot_layers=sl,
        encoder_type=enc_type).to(dev)
    if enc_type == "dmpnn":
        print(f"ENCODER = D-MPNN GRAPH (Bet A) over ORIGINAL smiles", flush=True)
    if a.pretrained:
        miss = model.encoder.load_state_dict(torch.load(a.pretrained, map_location=dev), strict=False)
        print(f"loaded pretrained encoder (missing {len(miss.missing_keys)})", flush=True)
    factors_path = os.getenv("CM_FACTORS")                    # FACTOR-shared binary heads (build_factors.py)
    if factors_path:
        from collections import Counter
        ftab = pq.read_table(factors_path).to_pydict()
        cpid = np.asarray(ftab["compact_property_id"]); fid = np.asarray(ftab["factor_id"])
        fid = fid[np.argsort(cpid)]                            # align to compact property order
        assert len(fid) == ds.num_props, f"factor file has {len(fid)} props, dataset {ds.num_props} (min_support mismatch?)"
        cnt = Counter(fid.tolist()); multi = sorted(f for f, n in cnt.items() if n >= 2)
        relab = {f: i for i, f in enumerate(multi)}; K = len(multi)
        factor_row = np.array([relab.get(int(f), K) for f in fid], dtype=np.int64)
        fmode = os.getenv("CM_FACTOR_MODE", "potential")
        model.handlers["binary"].enable_factors(torch.from_numpy(factor_row).to(dev), K, mode=fmode)
        model.to(dev)                                         # move newly-created factor params to GPU
        print(f"FACTORS ON (mode={fmode}): {K} multi-member factors, {int((factor_row < K).sum()):,} properties shared", flush=True)
        gates_path = os.getenv("CM_FACTOR_PRIORS")            # LLM priors: per-property sign*coupling gate
        if gates_path:
            gt = pq.read_table(gates_path).to_pydict()
            gate = np.asarray(gt["gate"])[np.argsort(np.asarray(gt["compact_property_id"]))]
            model.handlers["binary"].set_factor_priors(torch.from_numpy(gate.astype(np.float32)).to(dev))
            model.to(dev)
            print(f"FACTOR PRIORS ON: {int((gate != 1.0).sum()):,} gated members "
                  f"({int((gate < 0).sum()):,} sign-flipped)", flush=True)
        if int(os.getenv("CM_FACTOR_GCN", "0")):                # message passing over factor edges (GCN)
            model.enable_factor_gcn(torch.from_numpy(factor_row).to(dev), K); model.to(dev)
            print("FACTOR-GCN ON: slot decoder message-passes over factor edges", flush=True)
    tau = float(os.getenv("CM_LOGIT_ADJ", "0"))                  # LOGIT ADJUSTMENT + LABEL SMOOTHING (imbalance)
    lsm = float(os.getenv("CM_LABEL_SMOOTH", "0"))
    if tau > 0 or lsm > 0:
        _bm = ds.pair_typ == TYPE_IDS["binary"]
        _bp = ds.pair_pid[_bm].astype(np.int64); _bv = ds.pair_val[_bm]
        npos = np.bincount(_bp, weights=(_bv > 0.5), minlength=ds.num_props)
        ntot = np.bincount(_bp, minlength=ds.num_props).astype(np.float64)
        pi = np.clip(npos / np.maximum(ntot, 1), 1e-3, 1 - 1e-3)
        la = np.log(pi / (1 - pi)).astype(np.float32)
        model.handlers["binary"].set_logit_adj(torch.from_numpy(la).to(dev), tau)
        model.handlers["binary"].label_smooth = lsm
        print(f"LOGIT ADJUSTMENT tau={tau}, label_smooth={lsm}", flush=True)
    print(f"params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M", flush=True)
    # LR schedule: bigger transformers (d768/d1024) DIVERGE at a flat 3e-4 (binary loss explodes, Kendall
    # weight collapses ~step 10k). Linear warmup + cosine decay fixes it. CM_WARMUP=0 keeps the old flat LR
    # (d512 was stable). CM_LR sets peak; use a lower peak for the larger models.
    import math as _math
    base_lr = float(os.getenv("CM_LR", "3e-4"))
    warmup = int(os.getenv("CM_WARMUP", "0"))
    opt = torch.optim.AdamW(model.parameters(), lr=base_lr, betas=(0.9, 0.98), weight_decay=1e-2)
    _total = 6 if a.smoke else a.steps
    def lr_at(s):
        if warmup <= 0:
            return base_lr
        if s < warmup:
            return base_lr * (s + 1) / warmup
        prog = (s - warmup) / max(1, _total - warmup)
        return base_lr * (0.1 + 0.9 * 0.5 * (1 + _math.cos(_math.pi * min(1.0, prog))))
    # held-out val split (last ~4% of compounds, disjoint) → honest per-type metrics + overfit check
    from torch.utils.data import Subset
    # STRIDED split (every 25th compound) — representative of both types; a tail split lands entirely in
    # the chembl-only (numeric) region since combine_multitype orders merged compounds first.
    # CM_NO_HOLDOUT=1 → PRODUCTION mode: train on ALL rows (no val split). The holdout only exists to
    # VALIDATE the recipe isn't overfit; the shippable model trains on 100% of the data. (Eval is skipped;
    # final.pt is saved at the end + periodic last.pt.)
    all_idx = list(range(len(ds)))
    no_holdout = int(os.getenv("CM_NO_HOLDOUT", "0"))
    if no_holdout:
        va_idx = []; tr_idx = all_idx
        print("PRODUCTION mode: NO holdout — training on ALL rows", flush=True)
    else:
        stride = int(os.getenv("CM_HOLDOUT_STRIDE", "25"))   # denser split (e.g. 10) -> more held-out per prop
        va_set = set(all_idx[::stride]); va_idx = list(va_set)
        tr_idx = [i for i in all_idx if i not in va_set]
        val_cap = int(os.getenv("CM_VAL_CAP", "0"))          # cap held-out eval size for fast monitoring
        if val_cap > 0 and len(va_idx) > val_cap:
            va_idx = va_idx[:: max(1, len(va_idx) // val_cap)][:val_cap]
    # GRAPH mode: workers>0 fork the huge ik2graph dict and COW-break it (refcounting) -> RAM OOM.
    # Default to 0 workers in graph mode. CM_NWORKERS overrides.
    _nw = int(os.getenv("CM_NWORKERS", "0" if (a.smoke or ds.graph_dir) else "6"))
    dl = DataLoader(Subset(ds, tr_idx), batch_size=a.batch, shuffle=True, collate_fn=collate,
                    num_workers=_nw, drop_last=True, persistent_workers=_nw > 0)
    val_dl = DataLoader(Subset(ds, va_idx), batch_size=a.batch, shuffle=False, collate_fn=collate,
                        num_workers=max(0, _nw // 2))
    logdir = os.getenv("CM_LOGDIR", "cache/typed_run"); os.makedirs(logdir, exist_ok=True)
    eval_every = 1 if a.smoke else int(os.getenv("CM_EVAL_EVERY", "1000"))
    best = -1.0
    # CONFLICT-AVERSE GRADIENT (PCGrad over support tiers) setup
    pcgrad = int(os.getenv("CM_PCGRAD", "0"))
    n_tiers = 3
    if pcgrad:
        _bm = ds.pair_typ == TYPE_IDS["binary"]
        _cnt = np.bincount(ds.pair_pid[_bm].astype(np.int64), minlength=ds.num_props)
        tier = np.zeros(ds.num_props, np.int64)
        tier[_cnt >= 250] = 1; tier[_cnt >= 2000] = 2      # rare<250 / mid / common>=2000
        support_tier = torch.from_numpy(tier).to(dev)
        _shared_names = ("encoder.", "slot_dec.", "type_emb.", "log_var")   # trunk shared by all properties
        shared_params = [p for n, p in model.named_parameters()
                         if p.requires_grad and any(n.startswith(s) or n == s for s in _shared_names)]
        private_params = [p for n, p in model.named_parameters()
                          if p.requires_grad and not any(n.startswith(s) or n == s for s in _shared_names)]
        print(f"CONFLICT-AVERSE (PCGrad) ON: {n_tiers} support tiers, shared={len(shared_params)} "
              f"private={len(private_params)} param-tensors; tier sizes "
              f"{[int((tier==t).sum()) for t in range(n_tiers)]}", flush=True)
    selfcond = int(os.getenv("CM_SELFCOND", "0"))     # SELF-CONDITIONING (two-pass; see loop)
    sc_max = float(os.getenv("CM_SC_MAX", "0.5"))     # max frac of masked binary slots to self-fill
    sc_ramp = int(os.getenv("CM_SC_RAMP", "6000"))    # steps to ramp the self-fill frac 0->sc_max
    if selfcond:
        print(f"self-conditioning ON (sc_max={sc_max}, sc_ramp={sc_ramp})", flush=True)

    step, maxs = 0, (6 if a.smoke else a.steps)
    while step < maxs:
        for selfies, pid, typ, val, vmask, smask in dl:
            if warmup > 0:
                for g in opt.param_groups:
                    g["lr"] = lr_at(step)
            selfies = _to_dev(selfies, dev)
            pid, typ, val, vmask, smask = (x.to(dev) for x in (pid, typ, val, vmask, smask))
            if selfcond:
                # SELF-CONDITIONING: pass-1 (no grad) predict masked BINARY slots, fill the most-CONFIDENT
                # (scheduled ramp) with the model's OWN predicted value as pseudo-known, pass-2 predicts +
                # scores the rest. Teaches the model to bootstrap conditioning from its own predictions
                # (generative/iterative benefit, attributable predictions) with no extra labels.
                predict = smask & ~vmask
                bm = predict & (typ == TYPE_IDS["binary"])
                if bm.any():
                    with torch.no_grad():
                        p0 = torch.sigmoid(model(selfies, pid, typ, val, vmask, smask)["binary"])
                    conf = torch.maximum(p0, 1 - p0).masked_fill(~bm, -1.0)
                    frac = sc_max * min(1.0, step / max(1, sc_ramp))
                    quota = torch.floor(frac * bm.sum(1).float()).long()
                    order = conf.argsort(1, descending=True)
                    rank = torch.empty_like(order)
                    rank.scatter_(1, order, torch.arange(order.shape[1], device=dev).expand_as(order))
                    sc = bm & (rank < quota.unsqueeze(1))
                    val = torch.where(sc, (p0 > 0.5).float(), val)   # fill predicted binary value
                    vmask = vmask | sc                               # now "known" -> excluded from the loss
            preds = model(selfies, pid, typ, val, vmask, smask)
            loss, parts = model.compute_loss(preds, pid, typ, val, vmask, smask)
            if pcgrad:
                # CONFLICT-AVERSE: per-support-tier binary losses -> PCGrad over the shared trunk.
                import torch.nn.functional as _F
                predict = smask & ~vmask
                bmask = predict & (typ == TYPE_IDS["binary"])
                bce = _F.binary_cross_entropy_with_logits(preds["binary"], val.float(), reduction="none")
                glosses = []
                for t in range(n_tiers):
                    gm = bmask & (support_tier[pid.clamp(0, support_tier.shape[0] - 1)] == t)
                    if gm.any():
                        glosses.append((bce * gm).sum() / gm.sum().clamp_min(1))
                if len(glosses) >= 2:
                    conflict_averse_backward(glosses, shared_params, private_params, opt)
                else:
                    opt.zero_grad(); loss.backward()
            else:
                opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            if step % (1 if a.smoke else 500) == 0:
                pstr = " ".join(f"{k}={float(v):.3f}" for k, v in parts.items() if float(v) != 0)
                # learned Kendall weight per active type (½·e^{-s_t}); shows the auto-balancing.
                w = 0.5 * torch.exp(-model.log_var.detach())
                wstr = " ".join(f"w[{n}]={float(w[TYPE_IDS[n]]):.2f}" for n in model.handlers
                                if float(parts[n]) != 0)
                print(f"step={step} lr={opt.param_groups[0]['lr']:.2e} loss={float(loss):.4f} {pstr} | {wstr}", flush=True)
            if step > 0 and step % eval_every == 0:
                ckpt = {"model": model.state_dict(), "num_props": ds.num_props,
                        "num_mean_arr": ds.num_mean_arr, "num_std_arr": ds.num_std_arr,
                        "num_mean": ds.num_mean, "num_std": ds.num_std, "step": step}
                if no_holdout:
                    torch.save(ckpt, os.path.join(logdir, "last.pt"))   # production: no val -> periodic snapshot
                    print(f"  step={step} (no-holdout, saved last.pt)", flush=True)
                else:
                    m = evaluate(model, val_dl, dev)
                    print(f"  VAL step={step} " + " ".join(
                        f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in m.items()), flush=True)
                    torch.save({**ckpt, "val": m}, os.path.join(logdir, "last.pt"))   # ALWAYS keep the latest
                    # (selection can latch onto a noisy early peak; last.pt guarantees the converged model survives)
                    # SELECT on per-property MACRO AUC (base-rate-invariant), NOT pooled bin_auc which
                    # anti-correlates with the goal under long training. CM_SELECT overrides the key.
                    sc = m.get(os.getenv("CM_SELECT", "bin_macro"), float("nan"))
                    if sc == sc and sc > best and step >= warmup:  # not-nan, improved, and PAST WARMUP
                        best = sc                                  # (undertrained pre-warmup flukes can't win)
                        torch.save({**ckpt, "val": m}, os.path.join(logdir, "best.pt"))
            step += 1
            if step >= maxs: break
    if no_holdout:
        torch.save({"model": model.state_dict(), "num_props": ds.num_props,
                    "num_mean_arr": ds.num_mean_arr, "num_std_arr": ds.num_std_arr,
                    "num_mean": ds.num_mean, "num_std": ds.num_std, "step": step},
                   os.path.join(logdir, "final.pt"))
        print(f"DONE (production, no-holdout) — final.pt saved, {step} steps on ALL {len(ds):,} compounds", flush=True)
    else:
        print(f"SMOKE OK best_bin_auc={best:.4f}" if a.smoke else f"DONE best_bin_auc={best:.4f}", flush=True)


if __name__ == "__main__":
    main()
