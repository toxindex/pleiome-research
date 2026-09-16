"""
Big-run preprocessing STAGE 2: property normalization ETL.

Reads chemharmony `activities` (aid,sid,pid,source,inchi,smiles,value,binary_value)
and emits the unified *typed property records* described in
docs/research/bigrun-preprocessing-plan.md:

  one row per COMPOUND:
    { tokens : list<int16>   # SELFIES token ids from smiles (maxlen-truncated)
      inchikey : str         # dedup key
      compound_id : int32
      pairs : list<struct<property_id:int32, type:int8, value:float32>> }

plus two global registries:
    property_registry.parquet : (property_id int32, pid str, source str,
                                  type int8, n_records int64)
    compound registry         : the compounds/ shards (compound_id, inchikey, tokens)

TYPE detection (general, source-agnostic) per activity row:
    value parses as a finite float            -> NUMERIC (type=1), value = float(value)
    else binary_value in {0,1}                 -> BINARY  (type=0), value = float(binary_value)
    else                                       -> row skipped
  (chemharmony's activities brick is fully binarized: value in {positive,negative},
   binary_value in {0,1}; so every chemharmony record resolves to type=BINARY. The
   numeric branch is kept so the same schema/loader generalizes to ChEMBL/PubChemQC/... )

ALGORITHM (RAM-safe, two phase, hash-shard by compound):
  MAP  : stream activities in record batches; worker procs compute (inchikey, tokens)
         per row; the main proc assigns global compound_id / property_id and writes
         partitioned shards  pairs/shard_KK/*  and  compounds/shard_KK/*  where
         KK = compound_id % n_shards (so all activities of a compound share a shard).
         Only the inchikey->id and pid->id int maps live in RAM (bounded, small);
         tokens are streamed to disk, never accumulated.
  REDUCE: for each shard, group its pairs by compound_id and join the compound's
          tokens from the matching compounds shard -> props/props_KK.parquet.

USAGE (validated sample -- a few hundred k activities):
  PYTHONPATH=./ .venv/bin/python bigrun/normalize_properties.py \
      --src bigrun/_stage/activities_part00000.parquet \
      --out bigrun/props_sample --stage bigrun/_stage/pd_work \
      --workers 24 --n-shards 8 --limit 300000

FULL 255M run (see scale-up notes at bottom of file):
  copy the whole activities.parquet brick to the training machine first (gentle, sequential),
  then point --src at the directory, drop --limit, bump --n-shards to 256, run in tmux.
"""
import argparse
import glob
import json
import os
import math
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

import selfies as sf
from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

# Type ids MUST match property_multitask/typed_model.TYPE_IDS
# (binary=0, categorical=1, numeric=2, association=3, profile=4) so the ETL's `type` column
# routes to the correct per-type head. (Numeric was 1, which collided with categorical.)
TYPE_BINARY = 0
TYPE_NUMERIC = 2

# pyarrow type for the emitted `pairs` column
PAIR_STRUCT = pa.struct([
    ("property_id", pa.int32()),
    ("type", pa.int8()),
    ("value", pa.float32()),
])
PAIRS_TYPE = pa.list_(PAIR_STRUCT)
TOKENS_TYPE = pa.list_(pa.int16())

# ---------------------------------------------------------------------------
# worker: parse one batch of rows into per-row typed tuples (encoding-parallel)
# ---------------------------------------------------------------------------
_S2I = None
_MAXLEN = None


def _init_worker(tok_path, maxlen):
    global _S2I, _MAXLEN
    with open(tok_path) as fh:
        _S2I = json.load(fh)["symbol_to_index"]
    _MAXLEN = maxlen


def _smiles_to_tokens(smi):
    """SELFIES token ids (int16-range) from a SMILES string, or None."""
    try:
        enc = sf.encoder(smi, strict=False)
        if not enc:
            return None
        ids = [_S2I.get(sym, 0) for sym in sf.split_selfies(enc)][:_MAXLEN]
        return ids or None
    except Exception:
        return None


def _typed_value(value_str, binary_value):
    """Return (type:int8, value:float32) or None to skip."""
    if value_str is not None:
        try:
            f = float(value_str)
            if math.isfinite(f):
                return TYPE_NUMERIC, np.float32(f)
        except (ValueError, TypeError):
            pass
    if binary_value in (0, 1):
        return TYPE_BINARY, np.float32(binary_value)
    return None


def _parse_batch(rows):
    """rows: list of (inchi, smiles, pid, source, value, binary_value).
    returns list of (inchikey, tokens, pid, source, type, value)."""
    out = []
    for inchi, smi, pid, source, value_str, bv in rows:
        tv = _typed_value(value_str, bv)
        if tv is None:
            continue
        typ, fval = tv
        # dedup key: InChIKey (prefer the provided InChI; fall back to SMILES)
        ikey = None
        if inchi:
            try:
                ikey = Chem.inchi.InchiToInchiKey(inchi) or None
            except Exception:
                ikey = None
        if ikey is None and smi:
            try:
                m = Chem.MolFromSmiles(smi)
                if m is not None:
                    ikey = Chem.MolToInchiKey(m) or None
            except Exception:
                ikey = None
        if ikey is None or not smi:
            continue
        toks = _smiles_to_tokens(smi)
        if toks is None:
            continue
        out.append((ikey, toks, pid, source, int(typ), float(fval)))
    return out


# ---------------------------------------------------------------------------
# shard writers
# ---------------------------------------------------------------------------
class ShardWriter:
    """Buffers rows per shard and flushes fixed-size parquet chunks."""

    def __init__(self, root, n_shards, schema, flush_rows):
        self.root = root
        self.n = n_shards
        self.schema = schema
        self.flush_rows = flush_rows
        self.bufs = [[] for _ in range(n_shards)]
        self.chunk = [0] * n_shards
        for k in range(n_shards):
            os.makedirs(os.path.join(root, f"shard_{k:03d}"), exist_ok=True)

    def add(self, k, row):
        self.bufs[k].append(row)
        if len(self.bufs[k]) >= self.flush_rows:
            self._flush(k)

    def _flush(self, k):
        rows = self.bufs[k]
        if not rows:
            return
        cols = list(zip(*rows))
        arrays = [pa.array(list(c), type=self.schema.field(i).type)
                  for i, c in enumerate(cols)]
        tbl = pa.table(arrays, schema=self.schema)
        path = os.path.join(self.root, f"shard_{k:03d}", f"chunk_{self.chunk[k]:05d}.parquet")
        pq.write_table(tbl, path)
        self.chunk[k] += 1
        self.bufs[k] = []

    def close(self):
        for k in range(self.n):
            self._flush(k)


# ---------------------------------------------------------------------------
# MAP phase
# ---------------------------------------------------------------------------
def run_map(args):
    pair_schema = pa.schema([
        ("compound_id", pa.int32()),
        ("property_id", pa.int32()),
        ("type", pa.int8()),
        ("value", pa.float32()),
    ])
    comp_schema = pa.schema([
        ("compound_id", pa.int32()),
        ("inchikey", pa.string()),
        ("tokens", TOKENS_TYPE),
    ])
    pairs_w = ShardWriter(os.path.join(args.stage, "pairs"), args.n_shards, pair_schema, args.flush)
    comp_w = ShardWriter(os.path.join(args.stage, "compounds"), args.n_shards, comp_schema, args.flush)

    ikey2cid = {}                 # inchikey -> compound_id
    pid2pid = {}                  # source pid (str) -> property_id
    prop_meta = []                # property_id -> [pid, source, type, n_records]

    d = ds.dataset(args.src)
    cols = ["inchi", "smiles", "pid", "source", "value", "binary_value"]
    scanner = d.scanner(columns=cols, batch_size=args.batch)

    seen = kept = 0
    with ProcessPoolExecutor(max_workers=args.workers,
                             initializer=_init_worker,
                             initargs=(args.tokenizer, args.maxlen)) as ex:
        for rb in scanner.to_batches():
            n = rb.num_rows
            if args.limit and seen >= args.limit:
                break
            if args.limit and seen + n > args.limit:
                rb = rb.slice(0, args.limit - seen)
                n = rb.num_rows
            seen += n
            rows = list(zip(rb.column("inchi").to_pylist(),
                            rb.column("smiles").to_pylist(),
                            rb.column("pid").to_pylist(),
                            rb.column("source").to_pylist(),
                            rb.column("value").to_pylist(),
                            rb.column("binary_value").to_pylist()))
            # split batch across workers
            chunk = max(1, len(rows) // (args.workers * 4) + 1)
            parts = [rows[i:i + chunk] for i in range(0, len(rows), chunk)]
            for res in ex.map(_parse_batch, parts):
                for ikey, toks, pid, source, typ, fval in res:
                    cid = ikey2cid.get(ikey)
                    if cid is None:
                        cid = len(ikey2cid)
                        ikey2cid[ikey] = cid
                        comp_w.add(cid % args.n_shards, (cid, ikey, toks))
                    prop = pid2pid.get(pid)
                    if prop is None:
                        prop = len(pid2pid)
                        pid2pid[pid] = prop
                        prop_meta.append([pid, source, typ, 0])
                    prop_meta[prop][3] += 1
                    pairs_w.add(cid % args.n_shards, (cid, prop, typ, fval))
                    kept += 1
            if (seen // args.batch) % 20 == 0:
                print(f"[map] scanned {seen/1e6:.2f}M rows | kept {kept/1e6:.2f}M | "
                      f"compounds {len(ikey2cid)/1e6:.3f}M | properties {len(pid2pid)}",
                      flush=True)

    pairs_w.close()
    comp_w.close()

    # property registry
    reg = pa.table({
        "property_id": pa.array(range(len(prop_meta)), pa.int32()),
        "pid": pa.array([m[0] for m in prop_meta], pa.string()),
        "source": pa.array([m[1] for m in prop_meta], pa.string()),
        "type": pa.array([m[2] for m in prop_meta], pa.int8()),
        "n_records": pa.array([m[3] for m in prop_meta], pa.int64()),
    })
    os.makedirs(args.out, exist_ok=True)
    pq.write_table(reg, os.path.join(args.out, "property_registry.parquet"))
    print(f"[map] DONE scanned {seen} rows, kept {kept} pairs, "
          f"{len(ikey2cid)} compounds, {len(pid2pid)} properties", flush=True)
    return dict(seen=seen, kept=kept, compounds=len(ikey2cid), properties=len(pid2pid))


# ---------------------------------------------------------------------------
# REDUCE phase
# ---------------------------------------------------------------------------
def _reduce_shard(k, stage, out, maxlen):
    pair_glob = os.path.join(stage, "pairs", f"shard_{k:03d}", "*.parquet")
    comp_glob = os.path.join(stage, "compounds", f"shard_{k:03d}", "*.parquet")
    pair_files = sorted(glob.glob(pair_glob))
    comp_files = sorted(glob.glob(comp_glob))
    if not pair_files:
        return 0
    # compound_id -> tokens for this shard (bounded to shard size)
    tok = {}
    ikey = {}
    for cf in comp_files:
        ct = pq.read_table(cf)
        for cid, ik, tk in zip(ct.column("compound_id").to_pylist(),
                               ct.column("inchikey").to_pylist(),
                               ct.column("tokens").to_pylist()):
            tok[cid] = tk
            ikey[cid] = ik
    # group pairs by compound_id
    groups = {}
    for pf in pair_files:
        pt = pq.read_table(pf)
        for cid, prop, typ, val in zip(pt.column("compound_id").to_pylist(),
                                       pt.column("property_id").to_pylist(),
                                       pt.column("type").to_pylist(),
                                       pt.column("value").to_pylist()):
            groups.setdefault(cid, []).append(
                {"property_id": prop, "type": typ, "value": val})
    cids = sorted(groups)
    out_tbl = pa.table({
        "compound_id": pa.array(cids, pa.int32()),
        "inchikey": pa.array([ikey.get(c) for c in cids], pa.string()),
        "tokens": pa.array([tok.get(c, []) for c in cids], TOKENS_TYPE),
        "pairs": pa.array([groups[c] for c in cids], PAIRS_TYPE),
    })
    os.makedirs(out, exist_ok=True)
    pq.write_table(out_tbl, os.path.join(out, f"props_{k:03d}.parquet"))
    return len(cids)


def run_reduce(args):
    total = 0
    for k in range(args.n_shards):
        n = _reduce_shard(k, args.stage, args.out, args.maxlen)
        total += n
        print(f"[reduce] shard {k:03d}: {n} compound rows", flush=True)
    print(f"[reduce] DONE {total} compound rows -> {args.out}/props_*.parquet", flush=True)
    return total


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="activities parquet file or dataset dir")
    ap.add_argument("--out", required=True, help="final output dir (props_*.parquet + registry)")
    ap.add_argument("--stage", required=True, help="scratch dir for intermediate shards")
    ap.add_argument("--tokenizer",
                    default="brick/selfies_property_val_tokenizer/selfies_tokenizer.json")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--n-shards", type=int, default=8)
    ap.add_argument("--maxlen", type=int, default=128)
    ap.add_argument("--batch", type=int, default=20000)
    ap.add_argument("--flush", type=int, default=200000, help="rows buffered per shard before flush")
    ap.add_argument("--limit", type=int, default=0, help="cap activities scanned (0 = all)")
    ap.add_argument("--phase", choices=["map", "reduce", "all"], default="all")
    a = ap.parse_args()

    if a.phase in ("map", "all"):
        run_map(a)
    if a.phase in ("reduce", "all"):
        run_reduce(a)


if __name__ == "__main__":
    main()
