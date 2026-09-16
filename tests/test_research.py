import csv
import json
from pathlib import Path

import pytest
import torch

from pleiome_research.training import encode_rows, evaluate, fit_metadata, read_data, train

ROOT = Path(__file__).resolve().parents[1]


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["smiles", "property_id", "type", "value", "split"])
        writer.writerows(rows)


@pytest.mark.parametrize(
    "rows,message",
    [
        (
            [("CCO", "a", "binary", 0, "train"), ("OCC", "a", "binary", 1, "test")],
            "multiple partitions",
        ),
        ([("C", "a", "binary", 0, "train"), ("C", "a", "binary", 1, "train")], "Duplicate"),
        ([("C", "a", "binary", 2, "train")], "binary labels"),
        ([("C", "a", "numeric", "nan", "train")], "finite"),
        ([("", "a", "binary", 0, "train")], "empty molecule"),
        ([("C", "a", "binary", 0, "train"), ("CC", "a", "numeric", 1, "test")], "different types"),
    ],
)
def test_rejects_invalid_training_data(tmp_path, rows, message):
    path = tmp_path / "data.csv"
    write_csv(path, rows)
    with pytest.raises(ValueError, match=message):
        read_data(path)


def test_normalization_uses_training_rows_only(tmp_path):
    path = tmp_path / "data.csv"
    write_csv(
        path,
        [
            ("C", "a", "numeric", 0, "train"),
            ("CC", "a", "numeric", 2, "train"),
            ("CCC", "a", "numeric", 9000, "test"),
        ],
    )
    metadata = fit_metadata(read_data(path))
    assert metadata["properties"][0]["mean"] == 1
    assert metadata["properties"][0]["std"] == 1


def test_unknown_properties_and_symbols_are_rejected(tmp_path):
    path = tmp_path / "data.csv"
    write_csv(path, [("C", "a", "binary", 0, "train"), ("CC", "b", "binary", 1, "test")])
    rows = read_data(path)
    metadata = fit_metadata(rows)
    with pytest.raises(ValueError, match="Property absent"):
        encode_rows(rows, metadata, 64)
    write_csv(path, [("C", "a", "binary", 0, "train"), ("N", "a", "binary", 1, "test")])
    rows = read_data(path)
    with pytest.raises(ValueError, match="SELFIES symbol"):
        encode_rows(rows, fit_metadata(rows), 64)


def test_training_is_repeatable_and_evaluation_loads_checkpoint(tmp_path):
    data, config = ROOT / "examples/toy.csv", ROOT / "configs/smoke.json"
    a, b = tmp_path / "a", tmp_path / "b"
    train(data, config, a)
    train(data, config, b)
    assert json.loads((a / "history.json").read_text()) == json.loads(
        (b / "history.json").read_text()
    )
    first = torch.load(a / "best.pt", weights_only=True)
    second = torch.load(b / "best.pt", weights_only=True)
    for name, tensor in first["model"].items():
        torch.testing.assert_close(tensor, second["model"][name], rtol=0, atol=0)
    result = evaluate(a, data)
    assert result["per_property"]["toy_atoms"]["n"] == 4
    assert result == evaluate(b, data)
    with pytest.raises(FileExistsError):
        train(data, config, a)
