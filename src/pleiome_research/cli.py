"""Command-line setup, inference, numerical reproduction, and reference training."""

import argparse
import json
import math
from pathlib import Path

import torch

from .artifacts import prepare
from .inference import Predictor, load_catalog


def reproduce(
    directory="models/pleiome",
    manifest="artifacts/model-manifest.json",
    reference="artifacts/reference-predictions.json",
):
    torch.set_num_threads(1)
    model = Predictor(directory, manifest)
    expected = json.loads(Path(reference).read_text())
    results = []
    for case in expected["cases"]:
        actual = model.predict_batch(case["smiles"], case["property_indices"], case["rounds"])
        errors = []
        for molecule, targets in zip(actual, case["values"], strict=True):
            for prediction, value in zip(molecule["predictions"], targets, strict=True):
                got = (
                    prediction["probability"]
                    if prediction["type"] == "binary"
                    else prediction["value"]
                )
                error = abs(got - value)
                if not math.isfinite(error) or error > expected["atol"] + expected["rtol"] * abs(
                    value
                ):
                    raise ValueError(f"Prediction mismatch in {case['name']}")
                errors.append(error)
        results.append({"name": case["name"], "max_absolute_error": max(errors)})
    return {
        "checksums": "passed",
        "strict_load": "passed",
        "properties": len(model.catalog),
        "parameters": sum(p.numel() for p in model.model.parameters()),
        "predictions": results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=1, help="CPU threads (reference: 1)")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ["prepare", "reproduce", "predict"]:
        command = commands.add_parser(name)
        command.add_argument("--model", default="models/pleiome")
        command.add_argument("--manifest", default="artifacts/model-manifest.json")
        if name == "predict":
            command.add_argument("--smiles", required=True, nargs="+")
            command.add_argument("--properties", type=int, nargs="+", required=True)
            command.add_argument("--rounds", type=int, default=1)
            command.add_argument("--device", default="cpu")
    catalog = commands.add_parser("catalog")
    catalog.add_argument("--model", default="models/pleiome")
    catalog.add_argument("--search", default="")
    catalog.add_argument("--limit", type=int, default=20)
    train = commands.add_parser("train")
    train.add_argument("--data", default="examples/toy.csv")
    train.add_argument("--config", default="configs/smoke.json")
    train.add_argument("--output", required=True)
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--run", required=True)
    evaluate.add_argument("--data", default="examples/toy.csv")
    evaluate.add_argument("--split", choices=["train", "validation", "test"], default="test")
    args = parser.parse_args()
    try:
        if args.threads < 1:
            raise ValueError("threads must be positive")
        torch.set_num_threads(args.threads)
        if args.command == "prepare":
            result = {"checkpoint": str(prepare(args.model, args.manifest)), "checksums": "passed"}
        elif args.command == "reproduce":
            result = reproduce(args.model, args.manifest)
        elif args.command == "predict":
            predictor = Predictor(args.model, args.manifest, args.device)
            result = predictor.predict_batch(args.smiles, args.properties, args.rounds)
        elif args.command == "catalog":
            if args.limit < 1:
                raise ValueError("limit must be positive")
            rows = [
                r for r in load_catalog(args.model) if args.search.lower() in json.dumps(r).lower()
            ]
            result = {"matches": len(rows), "shown": rows[: args.limit]}
        elif args.command == "train":
            from .training import train

            result = train(args.data, args.config, args.output)
        else:
            from .training import evaluate

            result = evaluate(args.run, args.data, args.split)
        print(json.dumps(result, indent=2, allow_nan=False))
    except (ValueError, FileNotFoundError, FileExistsError, RuntimeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
