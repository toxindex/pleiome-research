"""
Combine binary (chemharmony) + numeric (ChEMBL pchembl) typed-record dirs into one MULTI-TYPE
dataset for the generic TypedPropertyModel. Merges by InChIKey so a compound measured in BOTH
sources carries binary AND numeric pairs in one row (tests cross-type conditioning). ChEMBL
property_ids are offset by the binary registry size so the two id-spaces don't collide.

  PYTHONPATH=./ python bigrun/combine_multitype.py \
    --binary bigrun/props_v1 --numeric bigrun/props_chembl --out bigrun/props_multitype
"""
import argparse, glob, os
import pyarrow as pa, pyarrow.parquet as pq


def _load(shard_dir):
    rows = []
    for f in sorted(glob.glob(os.path.join(shard_dir, "props_*.parquet"))):
        rows.extend(pq.read_table(f).to_pylist())
    return rows


def _num_props(rows):
    return 1 + max((p["property_id"] for r in rows for p in r["pairs"]), default=-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", required=True)
    ap.add_argument("--numeric", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-shards", type=int, default=16)
    a = ap.parse_args()

    bin_rows = _load(a.binary)
    off = _num_props(bin_rows)                       # offset ChEMBL pids past the binary id-space
    num_rows = _load(a.numeric)
    print(f"binary compounds={len(bin_rows):,} (offset={off}) | numeric compounds={len(num_rows):,}", flush=True)

    merged = {}                                      # inchikey -> {tokens, pairs}
    for r in bin_rows:
        merged[r["inchikey"]] = {"tokens": r["tokens"], "pairs": list(r["pairs"])}
    both = 0
    # normalize_properties emits numeric as type=1, but typed_model.TYPE_IDS has 1=categorical,
    # 2=numeric — remap the (all-numeric) ChEMBL source to typed_model's numeric id so pchembl
    # floats hit the Gaussian-NLL head, not the categorical head.
    NUMERIC = 2
    for r in num_rows:
        pairs = [{"property_id": p["property_id"] + off, "type": NUMERIC, "value": p["value"]}
                 for p in r["pairs"]]
        if r["inchikey"] in merged:
            merged[r["inchikey"]]["pairs"].extend(pairs); both += 1
        else:
            merged[r["inchikey"]] = {"tokens": r["tokens"], "pairs": pairs}

    os.makedirs(a.out, exist_ok=True)
    items = list(merged.items())
    per = (len(items) + a.n_shards - 1) // a.n_shards
    ntot = 0
    for s in range(a.n_shards):
        chunk = items[s * per:(s + 1) * per]
        if not chunk:
            continue
        tbl = pa.table({
            "inchikey": [k for k, _ in chunk],
            "tokens": [v["tokens"] for _, v in chunk],
            "pairs": [v["pairs"] for _, v in chunk],
        })
        pq.write_table(tbl, os.path.join(a.out, f"props_{s:03d}.parquet"))
        ntot += len(chunk)
    nprops = 1 + max((p["property_id"] for _, v in items for p in v["pairs"]), default=-1)
    print(f"DONE {ntot:,} compounds ({both:,} have BOTH binary+numeric) | num_properties={nprops:,} -> {a.out}",
          flush=True)


if __name__ == "__main__":
    main()
