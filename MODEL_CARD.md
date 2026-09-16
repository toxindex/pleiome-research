# Pleiome research model card

## Intended use

Study molecular graph representation, typed property prediction, self-conditioning, and property-conditioned SELFIES generation. The release supports local binary/numeric inference from the stored Pleiome 0.4 checkpoint and a small reproducible training experiment.

## Artifact

The model has 557,464,475 parameters, hidden width 768, eight directed graph updates, eight property decoder layers, and six structure decoder layers. The checkpoint records training step 120,000. The catalog covers 181,007 properties: 164,967 binary, 14,323 numeric, and 1,717 categorical. All weights and tokenizer indices are retained.

Weights use Git LFS in three parts. The manifest identifies both those parts and the original assembled checkpoint by SHA-256. The checkpoint includes per-property numeric normalization arrays but no compound/activity records. Catalog entries preserve source attribution. Training inputs inspected for the catalog audit were not copied into this release.

## Evidence

Verified behavior includes strict weight loading, numerical agreement with upstream inference for fixed binary/numeric/self-conditioned examples, agreement between batched and individual predictions, graph batching, exclusion of masked labels from inputs, generation causality, finite training gradients, save/load equality, and repeatable small CPU training runs.

The original repository describes large-scale public-source training and performance claims. This release does not independently reproduce those scientific benchmark results. The inspected historical input supports the catalog/index audit; an immutable, licensed training release and complete run lineage are still missing.

## Limits that affect interpretation

- A query's property panel and self-conditioning rounds affect its predictions. Record them with results.
- Numeric output uses the original training scale. Units and valid ranges require source endpoint definitions; a numeric value is not universally a hazard score.
- Catalog titles are inherited descriptive metadata. Some numeric columns may describe identifiers or ancillary fields rather than a scientifically useful endpoint. Inclusion in the vocabulary does not establish endpoint validity.
- Categorical class-index mappings are missing, so the public prediction interface rejects those endpoints. Association/profile handlers in the architecture are not evidence of trained performance for those tasks.
- Atom chirality is not explicitly represented by the graph featurizer. Stereoisomer sensitivity is limited.
- Calibration, applicability domain, uncertainty quality, external validation, and prospective use have not been established by this extraction.
- The generation decoder is included for research; this release does not validate generated molecules or achievement of requested properties.
- Historical normalization and support filtering precede the historical holdout split. Read the training notes before interpreting historical validation results.

## License and remaining work

Code and released weights use the MIT license. Original dataset terms still apply to source data. Exact historical retraining requires immutable data/graph snapshots, initialization checkpoints, training configuration and source lineage, optimizer/RNG history, and the compute environment. Scientific reproduction additionally requires benchmark definitions and outputs tied to those artifacts.
