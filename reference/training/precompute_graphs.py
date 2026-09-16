"""
Precompute molecular graphs from ORIGINAL SMILES (cache/inchikey_smiles.parquet, built from the chemharmony
source — NOT the lossy SELFIES round-trip). Keyed by inchikey so the graph-mode dataset looks graphs up by
each compound's inchikey (no row-order fragility). Parallel featurization, sharded.

  PYTHONPATH=./ .venv/bin/python property_multitask/precompute_graphs.py
Output: cache/graphs_ik/shard_*.pt  (each a dict {inchikey: (f_atoms fp16 [N,43], bonds list)})
"""
import os, glob, pyarrow.parquet as pq, torch, numpy as np
from multiprocessing import Pool
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")
torch.multiprocessing.set_sharing_strategy("file_system")

OUT = os.getenv("CM_GRAPHS", "cache/graphs_ik"); os.makedirs(OUT, exist_ok=True)
NPROC = int(os.getenv("CM_NPROC", "40"))


def _feat(item):
    from property_multitask.dmpnn_encoder import smiles_to_graph
    ik, smi = item
    try:
        g = smiles_to_graph(smi)
        return ik, g.f_atoms.numpy().astype(np.float16), g.bonds   # numpy avoids torch-tensor IPC (ancdata)
    except Exception:
        return ik, np.zeros((1, 43), dtype=np.float16), []


def main():
    m = pq.read_table("cache/inchikey_smiles.parquet").to_pydict()
    ik2smi = dict(zip(m["inchikey"], m["smiles"]))
    pf = []
    for f in sorted(glob.glob("bigrun/props_full/props_*.parquet")):
        pf.extend(pq.read_table(f, columns=["inchikey"]).to_pydict()["inchikey"])
    pf = list(dict.fromkeys(pf))                                     # unique, order-preserving
    items = [(ik, ik2smi[ik]) for ik in pf if ik in ik2smi]
    print(f"featurizing {len(items):,} unique compounds from ORIGINAL smiles with {NPROC} procs", flush=True)
    SH = 600_000
    with Pool(NPROC) as p:
        buf, sh, done, fail = {}, 0, 0, 0
        for ik, fa, bonds in p.imap_unordered(_feat, items, chunksize=2000):
            buf[ik] = (torch.from_numpy(fa), bonds); done += 1     # convert numpy->torch in MAIN proc
            if fa.shape[0] == 1 and not bonds:
                fail += 1
            if len(buf) >= SH:
                torch.save(buf, os.path.join(OUT, f"shard_{sh:03d}.pt")); sh += 1
                print(f"  saved shard {sh} ({done:,}/{len(items):,}, {fail} feat-fails)", flush=True); buf = {}
        if buf:
            torch.save(buf, os.path.join(OUT, f"shard_{sh:03d}.pt"))
    print(f"DONE graphs from original smiles -> {OUT} ({done:,} compounds, {fail} fails)", flush=True)


if __name__ == "__main__":
    main()
