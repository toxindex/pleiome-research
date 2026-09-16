"""
Merge each MOLECULE's full property profile into ONE row: chemharmony binary + brick tox/numeric/ADMET pairs
together (keyed by inchikey). This enables cross-type SELF-CONDITIONING (binary prediction can condition on
the molecule's brick properties) — the fix for the multi-type degradation (previously brick pairs were in
SEPARATE rows so they only competed, never helped). All props_bricks_g compounds are already props_full
compounds (filtered to graphs_ik), so we just APPEND brick pairs to the 4M props_full rows.
"""
import glob, os, pyarrow as pa, pyarrow.parquet as pq
from collections import defaultdict

brick = defaultdict(list)                                          # inchikey -> [brick pair-dicts]
for f in sorted(glob.glob("bigrun/props_bricks_g/props_*.parquet")):
    t = pq.read_table(f, columns=["inchikey", "pairs"]).to_pydict()
    for ik, pl in zip(t["inchikey"], t["pairs"]):
        brick[ik].extend(pl)
print(f"brick compounds with pairs: {len(brick):,}", flush=True)

os.makedirs("bigrun/props_merged", exist_ok=True)
pair_type = pa.list_(pa.struct([("property_id", pa.int32()), ("type", pa.int8()), ("value", pa.float32())]))
schema = pa.schema([("inchikey", pa.string()), ("tokens", pa.list_(pa.int16())), ("pairs", pair_type)])
merged_comp = appended = 0
for f in sorted(glob.glob("bigrun/props_full/props_*.parquet")):
    t = pq.read_table(f, columns=["inchikey", "tokens", "pairs"]).to_pydict()
    iks, tks, prs = [], [], []
    for ik, tk, pl in zip(t["inchikey"], t["tokens"], t["pairs"]):
        m = list(pl)
        if ik in brick:
            m = m + brick[ik]; appended += 1
        iks.append(ik); tks.append(tk); prs.append(m); merged_comp += 1
    pq.write_table(pa.table({"inchikey": iks, "tokens": tks, "pairs": prs}, schema=schema),
                   os.path.join("bigrun/props_merged", os.path.basename(f)))
print(f"MERGE DONE: {merged_comp:,} compounds, {appended:,} got brick pairs appended -> bigrun/props_merged", flush=True)
