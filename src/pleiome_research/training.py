"""Small, deterministic training reference with explicit compound partitions."""

import csv
import importlib.metadata
import json
import math
from pathlib import Path
import random

import numpy as np
from rdkit import Chem
import selfies as sf
from sklearn.metrics import roc_auc_score
import torch

from pleiome.dmpnn_encoder import collate_graphs, smiles_to_graph
from pleiome.proptypes import TYPE_IDS
from pleiome.unified_model import UnifiedModel
from .artifacts import sha256


def read_data(path):
    groups, membership, property_types = {}, {}, {}
    with Path(path).open(newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"smiles", "property_id", "type", "value", "split"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"CSV requires columns {sorted(required)}")
        for row in reader:
            mol = Chem.MolFromSmiles(row["smiles"])
            if mol is None or not mol.GetNumAtoms():
                raise ValueError("Invalid or empty molecule")
            smiles = Chem.MolToSmiles(mol, canonical=True)
            split, prop, kind = row["split"], row["property_id"], row["type"]
            if split not in ("train", "validation", "test") or not prop:
                raise ValueError("Invalid partition or empty property identifier")
            if kind not in ("binary", "numeric"):
                raise ValueError("The reference trainer supports binary and numeric values")
            if prop in property_types and property_types[prop] != kind:
                raise ValueError("A property cannot have different types")
            property_types[prop] = kind
            value = float(row["value"])
            if not math.isfinite(value) or (kind == "binary" and value not in (0, 1)):
                raise ValueError("Values must be finite; binary labels must be 0 or 1")
            if smiles in membership and membership[smiles] != split:
                raise ValueError("A canonical molecule occurs in multiple partitions")
            membership[smiles] = split
            entry = groups.setdefault(smiles, {"smiles": smiles, "split": split, "values": {}})
            if prop in entry["values"]:
                raise ValueError("Duplicate molecule/property observation")
            entry["values"][prop] = (kind, value)
    if not groups:
        raise ValueError("No observations")
    return list(groups.values())


def fit_metadata(rows):
    train = [row for row in rows if row["split"] == "train"]
    if not train:
        raise ValueError("Training partition is empty")
    properties = sorted({p for row in train for p in row["values"]})
    catalog = []
    for prop in properties:
        observations = [row["values"][prop] for row in train if prop in row["values"]]
        kind = observations[0][0]
        values = np.array([value for _, value in observations], dtype=np.float64)
        catalog.append(
            {
                "property_id": prop,
                "type": kind,
                "mean": float(values.mean()) if kind == "numeric" else 0.0,
                "std": max(float(values.std()), 1e-3) if kind == "numeric" else 1.0,
            }
        )
    symbols = sorted(
        {symbol for row in train for symbol in sf.split_selfies(sf.encoder(row["smiles"]))}
    )
    return {"properties": catalog, "vocabulary": {s: i + 3 for i, s in enumerate(symbols)}}


def encode_rows(rows, metadata, max_len):
    vocabulary = metadata["vocabulary"]
    prop_index = {row["property_id"]: i for i, row in enumerate(metadata["properties"])}
    encoded = []
    for row in rows:
        symbols = list(sf.split_selfies(sf.encoder(row["smiles"])))
        if any(symbol not in vocabulary for symbol in symbols):
            raise ValueError("SELFIES symbol absent from the training vocabulary")
        if len(symbols) + 1 > max_len:
            raise ValueError("Molecule exceeds the generation token limit")
        pairs = []
        for prop, (kind, value) in sorted(row["values"].items()):
            if prop not in prop_index:
                raise ValueError("Property absent from the training partition")
            index = prop_index[prop]
            info = metadata["properties"][index]
            if info["type"] != kind:
                raise ValueError("Property type differs from the training metadata")
            pairs.append((index, TYPE_IDS[kind], (value - info["mean"]) / info["std"]))
        encoded.append(
            {
                **row,
                "graph": smiles_to_graph(row["smiles"]),
                "pairs": pairs,
                "tokens": [vocabulary[s] for s in symbols],
            }
        )
    return encoded


def collate(rows, known_fraction=0.0):
    b, p = len(rows), max(len(row["pairs"]) for row in rows)
    pid = torch.zeros(b, p, dtype=torch.long)
    typ = torch.zeros_like(pid)
    values = torch.zeros(b, p)
    slots = torch.zeros(b, p, dtype=torch.bool)
    length = max(len(row["tokens"]) + 1 for row in rows)
    token_in = torch.zeros(b, length, dtype=torch.long)
    token_target = torch.zeros_like(token_in)
    for i, row in enumerate(rows):
        n = len(row["pairs"])
        for j, (prop, kind, value) in enumerate(row["pairs"]):
            pid[i, j], typ[i, j], values[i, j] = prop, kind, value
        slots[i, :n] = True
        tokens = row["tokens"]
        token_in[i, : len(tokens) + 1] = torch.tensor([1] + tokens)
        token_target[i, : len(tokens) + 1] = torch.tensor(tokens + [2])
    known = (
        (torch.rand(b, p) < known_fraction) & slots if known_fraction else torch.zeros_like(slots)
    )
    for i, row in enumerate(rows):
        if known[i].sum() == len(row["pairs"]):
            known[i, random.randrange(len(row["pairs"]))] = False
    structure = collate_graphs([row["graph"] for row in rows])
    return structure, pid, typ, values, known, slots, token_in, token_target


@torch.no_grad()
def evaluate_model(model, rows, metadata):
    if not rows:
        raise ValueError("Evaluation partition is empty")
    model.eval()
    results = {info["property_id"]: [] for info in metadata["properties"]}
    losses = []
    for row in rows:
        structure, pid, typ, values, known, slots, _, _ = collate([row])
        output = model(structure, pid, typ, values, known, slots)
        loss, _ = model.compute_loss(
            output, pid, typ, values, known, slots, uncertainty_weighting=False
        )
        losses.append(float(loss))
        for j, (index, _, _) in enumerate(row["pairs"]):
            info = metadata["properties"][index]
            predicted = (
                float(output["binary"][0, j].sigmoid())
                if info["type"] == "binary"
                else float(output["numeric"]["mu"][0, j]) * info["std"] + info["mean"]
            )
            results[info["property_id"]].append((row["values"][info["property_id"]][1], predicted))
    metrics = {}
    for info in metadata["properties"]:
        pairs = results[info["property_id"]]
        if not pairs:
            continue
        y, pred = np.array(pairs).T
        metric = {"n": len(y), "type": info["type"]}
        if info["type"] == "binary":
            clipped = np.clip(pred, 1e-7, 1 - 1e-7)
            metric.update(
                log_loss=float(-(y * np.log(clipped) + (1 - y) * np.log(1 - clipped)).mean()),
                roc_auc=float(roc_auc_score(y, pred)) if len(set(y)) == 2 else None,
            )
        else:
            metric.update(
                rmse=float(np.sqrt(np.mean((y - pred) ** 2))), mae=float(np.mean(abs(y - pred)))
            )
        metrics[info["property_id"]] = metric
    return {"structure_only_loss": float(np.mean(losses)), "per_property": metrics}


def train(data, config_path, output):
    config = json.loads(Path(config_path).read_text())
    if (
        config["epochs"] < 1
        or config["batch_size"] < 1
        or not math.isfinite(config["learning_rate"])
        or config["learning_rate"] <= 0
        or not 0 <= config["known_fraction"] <= 1
        or not math.isfinite(config["generation_weight"])
        or config["generation_weight"] < 0
    ):
        raise ValueError(
            "Invalid epochs, batch size, learning rate, context fraction, or generation weight"
        )
    torch.set_num_threads(1)
    seed = config["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    rows = read_data(data)
    metadata = fit_metadata(rows)
    architecture = {
        **config["architecture"],
        "selfies_vocab": len(metadata["vocabulary"]) + 3,
        "pad_idx": 0,
        "num_properties": len(metadata["properties"]),
        "encoder_type": "dmpnn",
    }
    encoded = encode_rows(rows, metadata, architecture["gen_max_len"])
    training = [row for row in encoded if row["split"] == "train"]
    validation = [row for row in encoded if row["split"] == "validation"]
    if not validation:
        raise ValueError("Validation partition is empty")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    model = UnifiedModel(**architecture)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=0.01)
    history, best = [], float("inf")
    for epoch in range(config["epochs"]):
        model.train()
        order = torch.randperm(len(training)).tolist()
        losses = []
        for offset in range(0, len(order), config["batch_size"]):
            batch = [training[i] for i in order[offset : offset + config["batch_size"]]]
            structure, pid, typ, values, known, slots, tin, target = collate(
                batch, config["known_fraction"]
            )
            pred = model(structure, pid, typ, values, known, slots)
            prediction_loss, _ = model.compute_loss(pred, pid, typ, values, known, slots)
            generation_loss = model.generation_loss(tin, target, pid, typ, values, slots)
            loss = prediction_loss + config["generation_weight"] * generation_loss
            if not torch.isfinite(loss):
                raise ValueError("Non-finite training loss")
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            losses.append(float(loss.detach()))
        metric = evaluate_model(model, validation, metadata)
        history.append({"epoch": epoch + 1, "training_loss": float(np.mean(losses)), **metric})
        if metric["structure_only_loss"] < best:
            best = metric["structure_only_loss"]
            torch.save(
                {"model": model.state_dict(), "architecture": architecture, "metadata": metadata},
                output / "best.pt",
            )
    versions = {
        name: importlib.metadata.version(name) for name in ["torch", "numpy", "rdkit", "selfies"]
    }
    (output / "run.json").write_text(
        json.dumps(
            {
                "config": config,
                "architecture": architecture,
                "data_sha256": sha256(data),
                "versions": versions,
            },
            indent=2,
        )
        + "\n"
    )
    (output / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    return history[-1]


def evaluate(run, data, split="test"):
    torch.set_num_threads(1)
    checkpoint = torch.load(Path(run) / "best.pt", map_location="cpu", weights_only=True)
    model = UnifiedModel(**checkpoint["architecture"])
    model.load_state_dict(checkpoint["model"], strict=True)
    rows = [r for r in read_data(data) if r["split"] == split]
    encoded = encode_rows(rows, checkpoint["metadata"], checkpoint["architecture"]["gen_max_len"])
    return evaluate_model(model, encoded, checkpoint["metadata"])
