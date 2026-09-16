# Pleiome research

Pleiome predicts molecular properties from a directed molecular graph and a requested set of property slots. The same property embeddings condition an autoregressive SELFIES generator. This repository contains the architecture, pretrained weights, property catalog, a local inference interface, a small training workflow, and historical training/preprocessing source.

This research derivative of [toxindex/pleiome](https://github.com/toxindex/pleiome) has independent Git history. The release focuses on model research and local reproduction.

New readers and integrators: [start here](docs/getting-started.md) for a reading path, prediction meanings, public links, and the distinction between local research and hosted access.

## Released model

| Component | Verified contents |
| --- | --- |
| Checkpoint | Pleiome 0.4 / v4, saved training step 120,000; unchanged checkpoint bytes |
| Parameters | 557,464,475 |
| Encoder | Directed message-passing neural network, width 768, eight message updates |
| Property decoder | Eight transformer decoder layers, eight attention heads |
| Structure decoder | Six transformer decoder layers, 2,467 SELFIES tokens |
| Property catalog | 181,007 entries: 164,967 binary, 14,323 numeric, 1,717 categorical |
| Local prediction interface | Binary and numeric endpoints; categorical endpoints require missing class mappings |

The original checkpoint, raw-property mapping, complete property types, and numerical examples are included. Exact historical retraining and scientific benchmark results are not reproduced by this release. See the [model card](MODEL_CARD.md) and [historical reproduction notes](docs/training.md).

## Reproduce predictions locally

Install [Git LFS](https://git-lfs.com/) and [uv](https://docs.astral.sh/uv/getting-started/installation/). The reference environment is Linux, Python 3.11, and CPU PyTorch. Allow at least 10 GB of disk space for Git objects, working files, assembled weights, and dependencies. An 8 GB RAM machine is a practical starting point for the small inference examples; larger batches and panels require more memory.

```bash
git lfs install
git clone https://github.com/toxindex/pleiome-research.git
cd pleiome-research
git lfs pull
uv sync --locked --extra dev --python 3.11

# Verify all artifacts, assemble the checkpoint, and reproduce recorded predictions.
uv run --locked pleiome reproduce

# Predict three binary endpoints: AMES, BBB_Martins, and Bioavailability_Ma.
uv run --locked pleiome predict --smiles CCO --properties 174208 174209 174210

# Find endpoints in the complete catalog.
uv run --locked pleiome catalog --search molecular_weight
```

The weights total 2,231,395,316 bytes and are stored as three Git LFS parts. `pleiome prepare`, `pleiome reproduce`, and `pleiome predict` verify the parts and assemble `models/pleiome/best.pt` automatically. The assembled file is ignored by Git. No cloud account or service credentials are needed.

In the recorded environment, ethanol's AMES probability is approximately `0.602006` for the three-property panel above with one decoding round. This is a numerical regression example, not an accuracy claim. Predictions can change when the requested panel or decoding rounds change. Numeric values retain their training scale; they are not automatically converted into hazard scores.

See [local reproduction](docs/local-reproduction.md) for Python use, batching, self-conditioning, GPU setup, and troubleshooting.

## Train and evaluate a small model

```bash
uv run --locked pytest -q
uv run --locked pleiome train --output runs/demo
uv run --locked pleiome evaluate --run runs/demo --split test
```

The included CSV contains 24 molecules with artificial oxygen/nitrogen labels and atom counts. Training jointly optimizes masked property prediction and SELFIES reconstruction. It writes the selected checkpoint, environment/configuration record, input checksum, and epoch history. Tests verify identical learned tensors and metrics across repeated runs in the same CPU environment.

For your own data, provide `smiles,property_id,type,value,split` columns and pass `--data`. The reference supports binary and numeric observations, explicit compound partitions, and training-only normalization. Read the [data contract](docs/data.md) before running an experiment. This small trainer is not the historical full-scale training recipe.

## Repository contents

| Path | Purpose |
| --- | --- |
| `src/pleiome/` | Graph features, directed message passing, typed property heads, SELFIES decoder, original tokenizer |
| `src/pleiome_research/` | Verified artifact loading, catalog-aware inference, local training and evaluation |
| `models/pleiome/` | LFS weight parts, architecture, tokenizers, raw IDs, compressed property catalog |
| `artifacts/` | Checksums and predictions recorded using upstream inference code |
| `configs/` | Small training configuration and recovered historical settings |
| `reference/` | Historical training and preprocessing source for study |
| `tests/` | Graph batching, label masking, generation causality, gradients, repeatability, and pretrained inference |

[Architecture](docs/architecture.md) explains the model. [Provenance](docs/provenance.md) describes the extraction and changes.

## License

The research code and released model weights use the [MIT license](LICENSE). The catalog preserves source attribution and descriptive metadata. Source datasets retain their own terms. No measured activity records, customer data, deployment configuration, or credentials are distributed.
