"""union the new expansion sources into props_merged -> props_merged_v2. Appends each compound's new
pairs to its existing row (by inchikey) and adds NEW-only compounds as new rows (tokens from the new records).
"""
import glob, os, pyarrow as pa, pyarrow.parquet as pq
from collections import defaultdict

NEW = ["bigrun/props_bindingdb", "bigrun/props_chembl_td", "bigrun/props_papyrus",
       "bigrun/props_meta/admet_tdc", "bigrun/props_meta/herg_lib", "bigrun/props_meta/epa_ecotox",
       "bigrun/props_meta/pka_dissociation", "bigrun/props_meta/lit_pcba"]
newpairs = defaultdict(list); newtok = {}
for d in NEW:
    for f in glob.glob(os.path.join(d, "props_*.parquet")):
        t = pq.read_table(f).to_pydict()
        for ik, tk, pl in zip(t["inchikey"], t["tokens"], t["pairs"]):
            if not ik:
                continue
            newpairs[ik].extend(pl)
            if ik not in newtok:
                newtok[ik] = tk
print(f"new sources: {len(newpairs):,} compounds carrying new pairs", flush=True)

os.makedirs("bigrun/props_merged_v2", exist_ok=True)
pt = pa.list_(pa.struct([("property_id", pa.int32()), ("type", pa.int8()), ("value", pa.float32())]))
schema = pa.schema([("inchikey", pa.string()), ("tokens", pa.list_(pa.int16())), ("pairs", pt)])
seen = set(); comp = appended = 0
for f in sorted(glob.glob("bigrun/props_merged/props_*.parquet")):
    t = pq.read_table(f).to_pydict(); iks, tks, prs = [], [], []
    for ik, tk, pl in zip(t["inchikey"], t["tokens"], t["pairs"]):
        m = list(pl)
        if ik in newpairs:
            m = m + newpairs[ik]; appended += 1
        seen.add(ik); iks.append(ik); tks.append(tk); prs.append(m); comp += 1
    pq.write_table(pa.table({"inchikey": iks, "tokens": tks, "pairs": prs}, schema=schema),
                   os.path.join("bigrun/props_merged_v2", os.path.basename(f)))
# NEW-only compounds (not in props_merged) -> new rows
iks, tks, prs = [], [], []; shard = 0; newonly = 0
for ik, pl in newpairs.items():
    if ik in seen or ik not in newtok:
        continue
    iks.append(ik); tks.append(newtok[ik]); prs.append(list(pl)); comp += 1; newonly += 1
    if len(iks) >= 200000:
        pq.write_table(pa.table({"inchikey": iks, "tokens": tks, "pairs": prs}, schema=schema),
                       f"bigrun/props_merged_v2/props_newonly_{shard:04d}.parquet")
        iks, tks, prs = [], [], []; shard += 1
if iks:
    pq.write_table(pa.table({"inchikey": iks, "tokens": tks, "pairs": prs}, schema=schema),
                   f"bigrun/props_merged_v2/props_newonly_{shard:04d}.parquet")
print(f"MERGE_V2_DONE total_compounds={comp:,} appended_to_existing={appended:,} new_only_compounds={newonly:,}", flush=True)
