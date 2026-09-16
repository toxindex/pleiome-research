# Local reproduction

## Install and verify

Use the Git LFS clone and `uv sync --locked --extra dev --python 3.11` commands in the README. Run all commands from the repository root. The CPU lockfile was validated with uv 0.7.21 and Python 3.11.13.

`uv run --locked pleiome reproduce` checks every artifact, assembles the checkpoint if necessary, loads all state-dictionary tensors strictly, and repeats the upstream binary, numeric, and self-conditioned examples. It reports `checksums: passed`, `strict_load: passed`, the property/parameter counts, and each example's maximum numerical difference. Float32 comparison uses absolute and relative tolerances of `1e-5`.

The assembled checkpoint's SHA-256 is `486bf984b7a94b609880c1b7a002c3c6bd8bed73f8f8fc6ecb44cfa73ed1f28f`. The three LFS parts preserve its original bytes. Assembly uses a temporary file, verifies the full checksum, and then renames the result. A corrupted existing checkpoint causes an error; remove that assembled file and rerun `prepare` after restoring the parts.

`git lfs fsck` checks the local LFS objects separately. A small file beginning with `version https://git-lfs.github.com/spec/v1` is a pointer, not model weights. Run `git lfs install` and `git lfs pull` to replace pointers. ZIP archives are not the tested distribution path. If GitHub reports an LFS quota restriction, contact the repository maintainers.

## Query explicit properties

```bash
uv run --locked pleiome catalog --search AMES --limit 20
uv run --locked pleiome predict --smiles CCO 'c1ccccc1' \
  --properties 174208 165254 165255 --rounds 1
```

This mixed panel requests AMES, predicted logP, and predicted molecular weight. Numeric outputs are model estimates on the stored training scale; they are not exact RDKit descriptor calculations. The interface returns property index, raw/source identifiers, source name, type, descriptive name, and `probability` or `value`. Missing source identifiers and metadata remain null. There are 17,806 entries without a source property identifier; the compact index and raw identifier remain available.

The supplied panel is part of the inference protocol. Property slots attend to each other even when all values are masked. Requesting an endpoint alone can produce a different value from requesting it in a larger panel. Preserve the complete panel and its order when comparing experiments. Splitting a large panel into smaller calls is a different inference protocol.

One round predicts from structure and masked property slots. More rounds use upstream self-conditioning: the least-confident remaining binary predictions are thresholded at 0.5 and revealed to the remaining slots. These are model-generated labels. The interface does not fetch observed activities. Numeric slots are predicted on the final round and are never revealed as conditioning values by this decoder. The allowed range is 1–20 rounds and 1–4,000 unique property indices per call.

## Use Python

```python
import torch
from pleiome_research.inference import Predictor

torch.set_num_threads(1)
predictor = Predictor()  # defaults to CPU; verifies and assembles artifacts
result = predictor.predict("CCO", [174208, 174209, 174210], rounds=1)
batch = predictor.predict_batch(["CCO", "c1ccccc1"], [174208, 165254], rounds=1)
```

Keep one `Predictor` instance when scoring many molecules to avoid repeated hash verification and model loading. Batching combines independent molecular graphs; tests compare batched and individual results. Invalid/empty SMILES, duplicate or out-of-range indices, and unsupported categorical queries fail explicitly.

The original SELFIES tokenizer is included for studying the structure decoder. Its old `num_assays` field describes an earlier tokenizer format and is not the Pleiome property count. Pleiome's separate property embeddings and catalog contain 181,007 entries.

## GPU experiments

The tested lockfile selects CPU PyTorch. Use a separate environment for a GPU build, install a PyTorch 2.8.0 build compatible with your CUDA installation, then install this package there. For CUDA 12.8:

```bash
uv venv --python 3.11 .venv-gpu
uv pip install --python .venv-gpu/bin/python torch==2.8.0 \
  --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv-gpu/bin/python -e .
.venv-gpu/bin/pleiome predict --device cuda --smiles CCO --properties 174208 174209 174210
```

These GPU commands have not been executed in this release's CPU validation. Record the complete GPU environment and compare predictions with an appropriate numerical tolerance. The CPU lockfile does not describe that separate environment. The small reference trainer intentionally runs on CPU; historical GPU training source is under `reference/`.

## Reproduction boundary

The release reproduces inference from this checkpoint and a small, deterministic training experiment. It does not establish exact historical retraining, held-out scientific accuracy, categorical class semantics, or the quality of generated molecules. See the model card and training notes for the missing artifacts.
