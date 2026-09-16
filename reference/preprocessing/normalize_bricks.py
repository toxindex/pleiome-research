"""
Normalize a curated set of biobricks datasets into the SAME typed-record schema as bigrun/props_full
(rows = {inchikey, tokens:list<int16> SELFIES ids, pairs:list<struct<property_id:int32,type:int8,value:float32>>}).
Auto-detects the structure column + value columns; handles WIDE assay-matrices (each numeric col = a property),
SINGLE-endpoint files, and optional GROUPBY long-format (property per distinct group value). Property-ids are
globally offset (>= PID_OFFSET) so they never collide with chemharmony's space; the training loader compacts
them after min-support. Robust: per-dataset + per-row try/except, skip-and-log, streaming (RAM-safe).

  PYTHONPATH=./ .venv/bin/python bigrun/normalize_bricks.py --out bigrun/props_bricks --workers 24
"""
import argparse, glob, json, os, math, traceback
from concurrent.futures import ProcessPoolExecutor
import numpy as np, pyarrow as pa, pyarrow.parquet as pq
import selfies as sf
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")

TYPE_BINARY, TYPE_CATEGORICAL, TYPE_NUMERIC = 0, 1, 2
PID_OFFSET = 2_000_000
MAXLEN = 120
BB = os.environ.get("BIOBRICKS_ROOT", "biobricks")

STRUCT_COLS = ["canonical_smiles", "smiles", "SMILES", "isomeric_smiles", "Canonical SMILES", "Smiles",
               "smiles_solute", "standardized_smiles", "Original_SMILES", "QSAR_READY_SMILES", "ligand",
               "structure", "mol", "inchi", "standard_inchi", "InChI", "inchikey", "InChIKey"]
SKIP_COLS = set(c.lower() for c in STRUCT_COLS) | {
    "cas", "casrn", "cas_number", "dtxsid", "id", "compound_id", "name", "title", "target", "target_id",
    "assay_id", "assay", "split", "unit", "units", "source", "index", "idx", "smiles_canonical", "mol_id",
    "pid", "sid", "aid", "cid", "chembl_id", "molregno", "inchi_key", "inchikey_standard", "std_inchi",
    "protein", "target_sequence", "uniprot", "organism", "reference", "year", "doc_id", "cmpd_id"}

# ---- curated datasets: (brick, file_glob, source_tag, groupby_col_or_None) ----
# wide-matrix + single-endpoint benchmarks across ALL modalities (deduped; drug-like -> high graph overlap).
DATASETS = [
    # ---- TOX panels ----
    ("moleculenet-datasets", "toxcast_data.parquet", "toxcast", None),
    ("moleculenet-datasets", "tox21.parquet", "tox21", None),
    ("moleculenet-datasets", "muv.parquet", "muv", None),
    ("moleculenet-datasets", "pcba.parquet", "pcba", None),
    ("moleculenet-datasets", "sider.parquet", "sider", None),
    ("toxric-30", "acute_toxicity_115_endpoints.parquet", "toxric_ld50_115", None),
    ("toxric-30", "acute_toxicity_59_endpoints.parquet", "toxric_ld50_59", None),
    ("toxric-30", "toxric_30_assays.parquet", "toxric_30", "assay_name"),
    ("gh-ncats-ld50multitask", "dataset.parquet", "ld50_multitask", None),
    ("cpdb-carcinogenicity", "data.parquet", "cpdb_carcino", None),
    ("bayer-dili", "bayer_dili_data.parquet", "dili_bayer", None),
    ("zenodo-zou-group", "unitox.parquet", "unitox", None),
    ("zou-group-unitox-website", "unitox.parquet", "unitox2", None),
    ("nura", "nura.parquet", "nura_nr", None),
    ("dataverse-herg-central", "herg_central.parquet", "herg_central", None),
    ("nih-herg-toxicity-prediction-using-trad", "training_all.parquet", "herg_tox", None),
    ("ec-tox-toxicology-datasets-europe", "toxric_endocrine_disruption_nr-ar.parquet", "ectox_nrar", None),
    ("adore-ecotox", "ecotox_mortality.parquet", "ecotox_mort", "endpoint"),
    ("qsar-aquatic-tox", "aquatic_toxicity.parquet", "aquatic_tox", None),
    ("cambridgemed-mutagenicity", "mutagenicity.parquet", "ames_cam", None),
    ("kinetic-solubility", "kinetic_solubility.parquet", "kinetic_sol", None),
    # ---- ADMET (TDC full suite; one clean copy) ----
    ("hf-skfp-cyp1a2-veith", "data.parquet", "cyp1a2", None), ("hf-skfp-cyp2c19-veith", "data.parquet", "cyp2c19", None),
    ("hf-skfp-cyp2c9-veith", "data.parquet", "cyp2c9", None), ("hf-skfp-cyp2d6-veith", "data.parquet", "cyp2d6", None),
    ("hf-skfp-cyp3a4-veith", "data.parquet", "cyp3a4", None), ("hf-skfp-bbbp", "data.parquet", "bbbp", None),
    ("hf-skfp-caco2-wang", "data.parquet", "caco2", None), ("hf-skfp-pampa-ncats", "data.parquet", "pampa", None),
    ("hf-skfp-clearance-hepatocyte-az", "clearance_hepatocyte_az.parquet", "cl_hep", None),
    ("hf-skfp-solubility-aqsoldb", "data.parquet", "sol_aqsol", None), ("hf-skfp-ppbr-az", "ppbr_az.parquet", "ppbr", None),
    ("hf-skfp-hia-hou", "data.parquet", "hia", None), ("hf-skfp-bioavailability-ma", "data.parquet", "bioavail", None),
    ("hf-skfp-pgp-broccatelli", "data.parquet", "pgp", None), ("hf-skfp-vdss-lombardo", "data.parquet", "vdss", None),
    ("hf-skfp-lipophilicity", "data.parquet", "lipo", None), ("hf-skfp-herg-karim", "data.parquet", "herg_karim", None),
    ("hf-skfp-dili", "dili.parquet", "dili", None), ("hf-skfp-ld50-zhu", "data.parquet", "ld50_zhu", None),
    ("hf-skfp-ames", "data.parquet", "ames", None), ("hf-skfp-clintox", "data.parquet", "clintox", None),
    ("hf-skfp-esol", "data.parquet", "esol", None), ("hf-skfp-freesolv", "data.parquet", "freesolv", None),
    ("bigsoldb", "solubility.parquet", "bigsoldb", None),
    ("acs-development-of-models-to-predi", "melting_point.parquet", "melting_pt", None),
    ("flashpoint", "flashpoint.parquet", "flashpoint", None),
    # ---- physchem / QM ----
    ("moleculenet-datasets", "qm9.parquet", "qm9", None), ("moleculenet-datasets", "qm8.parquet", "qm8", None),
    ("gh-glambard-moleculesdatasetcoll", "gdb9.parquet", "gdb9_qm", None),
    ("rdkit-properties", "rdkit_properties.parquet", "rdkit_phys", None),
    # ---- potency (single-endpoint, drug-like) ----
    ("davis-drug-target-affinity-benchmark", "data.parquet", "davis_kd", None),
    ("zenodo-5807719", "herg.parquet", "herg_pic50", None), ("zenodo-5807731", "nav15.parquet", "nav15_pic50", None),
    # ---- NOVEL modalities ----
    ("odor-smiles", "odor_smiles.parquet", "odor", None),
    ("chemtastesdb", "chemtastesdb.parquet", "taste", None),
    ("npclassifier-dataset", "natural_products.parquet", "np_class", None),
    ("bicerano-polymers", "bicerano_polymers.parquet", "polymer", None),
    ("pubchemghs", "PubChemGHSAll.parquet", "ghs_hazard", None),
    ("comptox-zebrafish", "zebrafish_toxicity.parquet", "zebrafish", None),
    ("metalcytotoxdb", "MetalCytoToxDB.parquet", "metal_cytotox", None),
    ("t3db", "toxins.parquet", "t3db_class", None),
]

_S2I, _TOK = None, None
def _init(tokdir, maxlen):
    global _S2I, _TOK
    import cvae.tokenizer
    _TOK = cvae.tokenizer.SelfiesPropertyValTokenizer.load(tokdir)
    _S2I = _TOK.selfies_tokenizer.symbol_to_index

def _struct_to_key_tokens(s):
    """structure string (smiles or inchi) -> (inchikey, tokens) or (None,None)."""
    try:
        s = str(s)
        if s.startswith("InChI="):
            m = Chem.MolFromInchi(s); ikey = Chem.inchi.InchiToInchiKey(s) if m else None
            smi = Chem.MolToSmiles(m) if m else None
        else:
            m = Chem.MolFromSmiles(s); ikey = Chem.MolToInchiKey(m) if m else None
            smi = Chem.MolToSmiles(m) if m else None
        if not ikey or not smi:
            return None, None
        enc = sf.encoder(smi, strict=False)
        toks = [_S2I.get(sym, 0) for sym in sf.split_selfies(enc)][:MAXLEN]
        return ikey, toks
    except Exception:
        return None, None

def _worker(task):
    """task=(struct_str, [(pid,typ,val)...]) -> (inchikey, tokens, pairs) or None."""
    s, pairs = task
    ikey, toks = _struct_to_key_tokens(s)
    if ikey is None or not toks:
        return None
    return (ikey, toks, pairs)


def _detect_struct_col(cols):
    lower = {c.lower(): c for c in cols}
    for pref in STRUCT_COLS:
        if pref.lower() in lower:
            return lower[pref.lower()]
    return None

def _col_type_and_values(arr):
    """Infer (type, float_array_or_None). binary if ⊆{0,1}; numeric if finite floats; else categorical->None."""
    import pandas as pd
    s = pd.Series(arr)
    num = pd.to_numeric(s, errors="coerce")
    frac_num = num.notna().mean()
    if frac_num > 0.6:                                              # numeric-ish column
        vals = num.to_numpy(dtype=np.float64)
        uniq = np.unique(vals[np.isfinite(vals)])
        if len(uniq) <= 2 and set(uniq.tolist()) <= {0.0, 1.0}:
            return TYPE_BINARY, vals
        return TYPE_NUMERIC, vals
    # categorical: map distinct strings to ints if few-valued
    cats = s.astype(str)
    uniq = cats.dropna().unique()
    if 2 <= len(uniq) <= 64:
        m = {u: i for i, u in enumerate(sorted(uniq))}
        return TYPE_CATEGORICAL, cats.map(m).to_numpy(dtype=np.float64)
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="bigrun/props_bricks")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--n-shards", type=int, default=64)
    ap.add_argument("--tokenizer", default="brick/selfies_property_val_tokenizer")
    ap.add_argument("--maxlen", type=int, default=MAXLEN)
    ap.add_argument("--max-rows", type=int, default=3_000_000)      # per-dataset cap (papyrus etc.)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    _init(a.tokenizer, a.maxlen)

    pair_type = pa.list_(pa.struct([("property_id", pa.int32()), ("type", pa.int8()), ("value", pa.float32())]))
    schema = pa.schema([("inchikey", pa.string()), ("tokens", pa.list_(pa.int16())), ("pairs", pair_type)])
    registry = []                                                  # (property_id, source, col, type)
    next_pid = PID_OFFSET
    shard_rows = [[] for _ in range(a.n_shards)]
    shard_idx = 0
    n_ok = n_props = 0

    def flush(final=False):
        for k in range(a.n_shards):
            if shard_rows[k] and (final or len(shard_rows[k]) >= 20000):
                ik = [r[0] for r in shard_rows[k]]; tk = [r[1] for r in shard_rows[k]]; pr = [r[2] for r in shard_rows[k]]
                t = pa.table({"inchikey": ik, "tokens": tk, "pairs": pr}, schema=schema)
                pq.write_table(t, os.path.join(a.out, f"props_{k:03d}_{shard_idx}.parquet"))
                shard_rows[k] = []

    with ProcessPoolExecutor(a.workers, initializer=_init, initargs=(a.tokenizer, a.maxlen)) as ex:
        for brick, fname, source, groupby in DATASETS:
            path = os.path.join(BB, brick, "brick", fname)
            if not os.path.exists(path):
                print(f"[skip] missing {source}: {path}", flush=True); continue
            try:
                tb = pq.read_table(path)
                cols = tb.column_names
                sc = _detect_struct_col(cols)
                if sc is None:
                    print(f"[skip] no struct col {source} ({cols[:6]})", flush=True); continue
                n = min(tb.num_rows, a.max_rows)
                d = tb.slice(0, n).to_pydict()
                structs = [str(x) for x in d[sc]]
                # value columns
                vcols = [c for c in cols if c != sc and c.lower() not in SKIP_COLS
                         and not any(c.lower() == s.lower() for s in STRUCT_COLS)]
                if groupby and groupby in cols:
                    # long-format: property = (source, group value); a single value col (first numeric non-group)
                    valcol = next((c for c in vcols if c != groupby), None)
                    if valcol is None:
                        print(f"[skip] no value col for grouped {source}", flush=True); continue
                    typ, vals = _col_type_and_values(d[valcol])
                    if vals is None:
                        continue
                    groups = [str(x) for x in d[groupby]]
                    gpid = {}
                    tasks = []
                    for i in range(n):
                        g = groups[i]; v = vals[i]
                        if not np.isfinite(v):
                            continue
                        if g not in gpid:
                            gpid[g] = next_pid; registry.append((next_pid, source, g, typ)); next_pid += 1
                        tasks.append((structs[i], [(int(gpid[g]), int(typ), float(v))]))
                    n_props += len(gpid)
                else:
                    # wide/single: each value col -> a property
                    colpid = {}
                    coltyp = {}
                    keep = []
                    for c in vcols:
                        typ, vals = _col_type_and_values(d[c])
                        if vals is None:
                            continue
                        colpid[c] = next_pid; coltyp[c] = (typ, vals); registry.append((next_pid, source, c, typ)); next_pid += 1
                        keep.append(c)
                    if not keep:
                        print(f"[skip] no value cols {source}", flush=True); continue
                    n_props += len(keep)
                    tasks = []
                    for i in range(n):
                        pairs = []
                        for c in keep:
                            typ, vals = coltyp[c]; v = vals[i]
                            if np.isfinite(v):
                                pairs.append((int(colpid[c]), int(typ), float(v)))
                        if pairs:
                            tasks.append((structs[i], pairs))
                # tokenize in parallel
                got = 0
                for res in ex.map(_worker, tasks, chunksize=256):
                    if res is None:
                        continue
                    ikey, toks, pairs = res
                    shard_rows[hash(ikey) % a.n_shards].append((ikey, toks, [
                        {"property_id": p, "type": t, "value": v} for (p, t, v) in pairs]))
                    got += 1
                n_ok += got
                flush()
                print(f"[ok] {source:22} rows={n:>8,} props={n_props:>6,} kept={got:>8,} total_props={next_pid-PID_OFFSET:,}", flush=True)
            except Exception as e:
                print(f"[FAIL] {source}: {str(e)[:120]}", flush=True)
                traceback.print_exc()
    flush(final=True)
    reg = pa.table({"property_id": [r[0] for r in registry], "source": [r[1] for r in registry],
                    "col": [str(r[2]) for r in registry], "type": [r[3] for r in registry]})
    pq.write_table(reg, os.path.join(a.out, "brick_registry.parquet"))
    print(f"\nDONE: {n_ok:,} compound-records, {next_pid-PID_OFFSET:,} new properties, registry -> brick_registry.parquet", flush=True)
    print("NORMALIZE_BRICKS_DONE", flush=True)


if __name__ == "__main__":
    main()
