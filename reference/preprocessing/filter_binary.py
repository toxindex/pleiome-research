"""props_bricks_g -> props_bricks_gb: keep only BINARY pairs (type==0), drop records left with none.
Tests whether the numeric type (not the extra properties) was degrading the core binary prediction."""
import glob, os, pyarrow as pa, pyarrow.parquet as pq
os.makedirs("bigrun/props_bricks_gb", exist_ok=True)
pair_type = pa.list_(pa.struct([("property_id", pa.int32()), ("type", pa.int8()), ("value", pa.float32())]))
schema = pa.schema([("inchikey", pa.string()), ("smiles", pa.string()), ("tokens", pa.list_(pa.int16())), ("pairs", pair_type)])
kept = 0; nbin = 0; ncat = 0; nnum = 0
for f in sorted(glob.glob("bigrun/props_bricks_g/props_*.parquet")):
    t = pq.read_table(f).to_pydict()
    iks, sms, tks, prs = [], [], [], []
    for ik, sm, tk, pl in zip(t["inchikey"], t["smiles"], t["tokens"], t["pairs"]):
        bp = [p for p in pl if p["type"] == 0]
        nbin += len(bp); ncat += sum(1 for p in pl if p["type"] == 1); nnum += sum(1 for p in pl if p["type"] == 2)
        if bp:
            iks.append(ik); sms.append(sm); tks.append(tk); prs.append(bp); kept += 1
    if iks:
        pq.write_table(pa.table({"inchikey": iks, "smiles": sms, "tokens": tks, "pairs": prs}, schema=schema),
                       os.path.join("bigrun/props_bricks_gb", os.path.basename(f)))
print(f"FILTER_BINARY DONE: kept {kept:,} records | binary pairs {nbin:,} (kept) | dropped categorical {ncat:,} numeric {nnum:,}", flush=True)
