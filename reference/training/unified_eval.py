"""
B-FULL evaluation — does ONE checkpoint deliver all characteristics?
 (1) PREDICT : uncapped target-set per-property AUC (comparable to the D-MPNN's 0.667 frac>=0.80).
 (2) DESIGN  : condition on a val compound's property profile -> GENERATE a molecule -> re-featurize ->
               re-predict its properties with the SAME model's predict path -> prop_match vs the target,
               plus validity (RDKit-parseable) and novelty (differs from the source SMILES).
 (3) SELF-COND is measured separately by probe_selfcond_eval.py.

  PYTHONPATH=./ CM_PROPS=bigrun/props_full CM_MIN_SUPPORT=10 CM_GRAPH_DIR=cache/graphs_ik \
    CM_D_MODEL=512 CM_LAYERS=5 UM_GEN_LAYERS=6 CM_CKPT=cache/unified/best.pt \
    .venv/bin/python property_multitask/unified_eval.py
"""
import os, functools, numpy as np, torch
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import roc_auc_score
import cvae.tokenizer
from property_multitask.train_typed import _to_dev
from property_multitask.unified_train import UnifiedRecords, collate_unified
from property_multitask.unified_model import UnifiedModel, SOS, EOS, PADT
from property_multitask.typed_model import TYPE_IDS
from property_multitask.dmpnn_encoder import smiles_to_graph, collate_graphs, MolGraph, ATOM_FDIM


@torch.no_grad()
def main():
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tok = cvae.tokenizer.SelfiesPropertyValTokenizer.load("brick/selfies_property_val_tokenizer")
    ds = UnifiedRecords(os.getenv("CM_PROPS", "bigrun/props_full"), max_props=128)
    d = int(os.getenv("CM_D_MODEL", "512")); L = int(os.getenv("CM_LAYERS", "5"))
    gl = int(os.getenv("UM_GEN_LAYERS", "6"))
    # target set
    bm = ds.pair_typ == TYPE_IDS["binary"]; bpid = ds.pair_pid[bm].astype(np.int64); bval = ds.pair_val[bm]
    npos = np.bincount(bpid, weights=(bval > 0.5).astype(np.float64), minlength=ds.num_props)
    nneg = np.bincount(bpid, weights=(bval <= 0.5).astype(np.float64), minlength=ds.num_props)
    is_target = (npos >= 100) & (nneg >= 100)

    model = UnifiedModel.from_tokenizer(tok, num_properties=ds.num_props, encoder_type="dmpnn",
                                        d_model=d, nhead=8, struct_layers=L, slot_layers=L,
                                        gen_layers=gl, gen_max_len=128).to(dev)
    ck = torch.load(os.getenv("CM_CKPT"), map_location=dev, weights_only=False)
    model.load_state_dict(ck["model"]); model.eval()
    print(f"loaded {os.getenv('CM_CKPT')} step {ck.get('step')} | target props {int(is_target.sum()):,}", flush=True)

    stride = int(os.getenv("CM_HOLDOUT_STRIDE", "10"))
    va = list(range(len(ds)))[::stride]
    cap = int(os.getenv("CM_VAL_CAP", "20000"))
    if len(va) > cap:
        va = va[:: max(1, len(va) // cap)][:cap]
    dl = DataLoader(Subset(ds, va), batch_size=48, shuffle=False, num_workers=0,
                    collate_fn=functools.partial(collate_unified, known_frac=0.5))

    # ---------- (1) PREDICT: target-set per-property AUC ----------
    by = {}
    for structure, tin, ttg, pid, typ, val, vmask, smask in dl:
        structure = _to_dev(structure, dev)
        pid, typ, val, vmask, smask = (x.to(dev) for x in (pid, typ, val, vmask, smask))
        predict = smask & ~vmask & (typ == TYPE_IDS["binary"])
        if not predict.any():
            continue
        p = torch.sigmoid(model(structure, pid, typ, val, vmask, smask)["binary"])[predict].cpu().numpy()
        y = val[predict].cpu().numpy(); pids = pid[predict].cpu().numpy()
        for pp, yy, dd in zip(pids, y, p):
            by.setdefault(int(pp), ([], []))[0].append(dd); by[int(pp)][1].append(yy)
    taucs = []
    for pp, (pr, yl) in by.items():
        if is_target[pp]:
            yl = np.array(yl)
            if len(yl) >= 8 and yl.min() != yl.max():
                taucs.append(roc_auc_score(yl, np.array(pr)))
    taucs = np.array(taucs)
    print(f"\n(1) PREDICT target-set ({len(taucs):,} scorable): frac>=0.80 = {(taucs>=0.8).mean():.3f}  "
          f"mean {taucs.mean():.4f}  median {np.median(taucs):.4f}   [D-MPNN was 0.667]", flush=True)

    # ---------- (2) DESIGN: generate from profile -> re-predict -> prop_match ----------
    n = valid = novel = 0; match = tot = 0
    ndesign = int(os.getenv("UM_DESIGN_BATCHES", "12"))
    for bi, (structure, tin, ttg, pid, typ, val, vmask, smask) in enumerate(dl):
        if bi >= ndesign:
            break
        pid, typ, val, smask = (x.to(dev) for x in (pid, typ, val, smask))
        gen = model.generate(pid, typ, val, smask, max_len=120)              # [B,Lgen] SELFIES ids
        # decode -> SMILES -> graph; re-predict with property slots MASKED, compare to target val
        smis, keep = [], []
        for b in range(gen.shape[0]):
            ids = [int(x) for x in gen[b].tolist() if x not in (PADT, EOS, SOS)]
            try:
                smi = tok.selfies_tokenizer.indexes_to_smiles(ids)
            except Exception:
                smi = ""
            m = Chem.MolFromSmiles(smi) if smi else None
            n += 1
            if m is not None and m.GetNumAtoms() > 0:
                valid += 1; keep.append((b, smi)); smis.append(smi)
        if not keep:
            continue
        graphs = []
        for _, smi in keep:
            try:
                g = smiles_to_graph(smi); graphs.append(MolGraph(g.f_atoms, g.bonds))
            except Exception:
                graphs.append(MolGraph(torch.zeros((1, ATOM_FDIM)), []))
        gb = collate_graphs(graphs); gb = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in gb.items()}
        bidx = [b for b, _ in keep]
        gpid = pid[bidx]; gtyp = typ[bidx]; gval = val[bidx]; gsm = smask[bidx]
        vm0 = torch.zeros_like(gsm)                                          # predict ALL profile slots
        pr = torch.sigmoid(model(gb, gpid, gtyp, gval, vm0, gsm)["binary"])
        binm = gsm & (gtyp == TYPE_IDS["binary"])
        match += (((pr > 0.5).float() == gval) & binm).sum().item(); tot += binm.sum().item()
    print(f"\n(2) DESIGN ({n} generated): validity {valid/max(n,1):.3f}  "
          f"prop_match(gen vs target) {match/max(tot,1):.3f}   [joint-diffusion ceiling ~0.85]", flush=True)
    print("UNIFIED_EVAL_DONE", flush=True)


if __name__ == "__main__":
    main()
