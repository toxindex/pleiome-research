"""Local property prediction with explicit catalog types and decoding settings."""

import gzip
import json
import math
from pathlib import Path

import numpy as np
import torch

from pleiome.dmpnn_encoder import collate_graphs, smiles_to_graph
from pleiome.proptypes import TYPE_IDS
from pleiome.unified_model import UnifiedModel
from .artifacts import prepare


def load_catalog(directory="models/pleiome"):
    with gzip.open(Path(directory) / "properties.jsonl.gz", "rt") as stream:
        rows = [json.loads(line) for line in stream]
    if [row["index"] for row in rows] != list(range(len(rows))):
        raise ValueError("Catalog indices must be contiguous and in order")
    if any(row["type"] not in TYPE_IDS for row in rows):
        raise ValueError("Unknown catalog property type")
    return rows


class Predictor:
    def __init__(
        self, directory="models/pleiome", manifest="artifacts/model-manifest.json", device="cpu"
    ):
        self.directory = Path(directory)
        path = prepare(self.directory, manifest)
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA is unavailable; the locked environment uses CPU PyTorch")
        checkpoint = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        config = json.loads((self.directory / "architecture.json").read_text())
        self.catalog = load_catalog(self.directory)
        raw = np.load(self.directory / "raw_pids.npy", allow_pickle=False)
        if len(self.catalog) != checkpoint["num_props"] or config["num_properties"] != len(raw):
            raise ValueError("Checkpoint, architecture, and property mapping sizes disagree")
        if not np.array_equal(raw, [row["raw_property_id"] for row in self.catalog]):
            raise ValueError("Catalog and raw property IDs disagree")
        # Allocate parameters directly from the checkpoint instead of initializing a second 2 GB copy.
        with torch.device("meta"):
            self.model = UnifiedModel(**config)
        self.model.load_state_dict(checkpoint["model"], strict=True, assign=True)
        self.model.to(self.device).eval()
        self.mean = checkpoint["num_mean_arr"].to(self.device)
        self.std = checkpoint["num_std_arr"].to(self.device)
        if self.mean.shape != (len(raw),) or self.std.shape != (len(raw),):
            raise ValueError("Numeric normalization arrays have the wrong shape")
        if (
            not torch.isfinite(self.mean).all()
            or not torch.isfinite(self.std).all()
            or (self.std <= 0).any()
        ):
            raise ValueError(
                "Numeric normalization arrays must be finite with positive standard deviations"
            )

    @torch.no_grad()
    def predict(self, smiles, property_ids, rounds=1):
        return self.predict_batch([smiles], property_ids, rounds)[0]

    @torch.no_grad()
    def predict_batch(self, smiles_list, property_ids, rounds=1):
        ids = list(property_ids)
        if not ids or len(ids) > 4000:
            raise ValueError("Request between 1 and 4000 properties")
        if any(type(index) is not int or not 0 <= index < len(self.catalog) for index in ids):
            raise ValueError("Property index is outside the checkpoint vocabulary")
        if len(set(ids)) != len(ids):
            raise ValueError("Duplicate property indices are not allowed")
        if type(rounds) is not int or not 1 <= rounds <= 20:
            raise ValueError("rounds must be an integer between 1 and 20")
        if not smiles_list or any(not isinstance(s, str) or not s.strip() for s in smiles_list):
            raise ValueError("Supply nonempty SMILES strings")
        records = [self.catalog[index] for index in ids]
        if any(row["type"] not in ("binary", "numeric") for row in records):
            raise ValueError(
                "The prediction interface supports binary and numeric endpoints; categorical class mappings are not released"
            )
        batch = collate_graphs([smiles_to_graph(smiles) for smiles in smiles_list])
        b, p = len(smiles_list), len(ids)
        pid = torch.tensor([ids], device=self.device).expand(b, p).contiguous()
        typ = torch.tensor([[TYPE_IDS[row["type"]] for row in records]], device=self.device).expand(
            b, p
        )
        slots = torch.ones(b, p, dtype=torch.bool, device=self.device)
        known = torch.zeros_like(slots)
        values = torch.zeros(b, p, device=self.device)
        binary = typ == TYPE_IDS["binary"]
        recorded = torch.full_like(values, float("nan"))
        for step in range(rounds):
            output = self.model(batch, pid, typ, values, known, slots)
            probabilities = output["binary"].sigmoid()
            remaining = binary & ~known
            recorded = torch.where(remaining, probabilities, recorded)
            if step == rounds - 1 or not remaining.any():
                break
            confidence = torch.where(remaining, (probabilities - 0.5).abs(), 1e9)
            for molecule in range(b):
                count = int(remaining[molecule].sum())
                reveal = math.ceil(count / (rounds - step))
                if reveal:
                    indices = (-confidence[molecule]).topk(reveal).indices
                    known[molecule, indices] = True
                    values[molecule, indices] = (probabilities[molecule, indices] > 0.5).float()
        numeric = output["numeric"]["mu"] * self.std[pid] + self.mean[pid]
        results = []
        for molecule in range(b):
            entries = []
            for column, row in enumerate(records):
                key = "probability" if row["type"] == "binary" else "value"
                value = float(
                    recorded[molecule, column]
                    if key == "probability"
                    else numeric[molecule, column]
                )
                if not math.isfinite(value):
                    raise ValueError("Model produced a non-finite prediction")
                entries.append({**row, key: value})
            results.append(
                {"smiles": smiles_list[molecule], "rounds": rounds, "predictions": entries}
            )
        return results
