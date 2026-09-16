import json
from pathlib import Path

import pytest
import torch

from pleiome_research.inference import Predictor

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def predictor():
    torch.set_num_threads(1)
    return Predictor(ROOT / "models/pleiome", ROOT / "artifacts/model-manifest.json")


def test_pretrained_matches_upstream_numerical_examples(predictor):
    reference = json.loads((ROOT / "artifacts/reference-predictions.json").read_text())
    assert sum(p.numel() for p in predictor.model.parameters()) == 557_464_475
    for case in reference["cases"]:
        results = predictor.predict_batch(case["smiles"], case["property_indices"], case["rounds"])
        for result, expected in zip(results, case["values"], strict=True):
            actual = [p.get("probability", p.get("value")) for p in result["predictions"]]
            assert actual == pytest.approx(expected, abs=reference["atol"], rel=reference["rtol"])


def test_batch_and_single_predictions_agree(predictor):
    panel = [174208, 174209, 165254]
    together = predictor.predict_batch(["CCO", "c1ccccc1"], panel, rounds=3)
    for result in together:
        alone = predictor.predict(result["smiles"], panel, rounds=3)
        actual = [p.get("probability", p.get("value")) for p in result["predictions"]]
        expected = [p.get("probability", p.get("value")) for p in alone["predictions"]]
        assert actual == pytest.approx(expected, abs=1e-5, rel=1e-5)


@pytest.mark.parametrize(
    "smiles,indices,rounds,message",
    [
        ("CCO", [-1], 1, "outside"),
        ("CCO", [181007], 1, "outside"),
        ("CCO", [1, 1], 1, "Duplicate"),
        ("CCO", [163201], 1, "categorical"),
        ("CCO", [0], 0, "rounds"),
        ("", [0], 1, "nonempty"),
        ("not-a-molecule", [0], 1, "parse"),
    ],
)
def test_invalid_prediction_requests_fail(predictor, smiles, indices, rounds, message):
    with pytest.raises(ValueError, match=message):
        predictor.predict(smiles, indices, rounds)
