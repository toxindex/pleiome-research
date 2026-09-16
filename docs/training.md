# Training and historical reproduction

## Tested local workflow

`pleiome train` uses `configs/smoke.json` by default. The small model has width 32, three graph updates, one property decoder layer, and one structure decoder layer. AdamW minimizes the architecture's masked property loss plus a weighted SELFIES reconstruction loss. It uses explicit seeds, one CPU thread, deterministic PyTorch operations, gradient clipping, and a fixed learning rate.

The command writes `best.pt`, `run.json`, and `history.json` into a new output directory and refuses to overwrite an existing run. `run.json` records the effective architecture, training configuration, input SHA-256, and scientific package versions. Repeating the same run in the locked CPU environment produces identical selected tensors and epoch metrics. Equality across hardware, platforms, library versions, or GPU implementations is not guaranteed.

The toy labels are generated from 24 simple molecules: oxygen presence, nitrogen presence, and atom count. No measured assay observations are included. Adjusting the small configuration creates a new experiment; it does not recreate Pleiome v4.

## Recovered historical evidence

The distributed weights are byte-identical to the saved v4 best checkpoint, which records step 120,000. A recovered launcher uses width 768, eight prediction layers, six generation layers, batch size 40, a nominal 150,000-step run, AdamW, warmup and cosine scheduling, mixed known-value fractions, self-conditioning, a structure-only auxiliary loss, and property weighting. `configs/historical-v4.json` records those settings without machine paths or deployment instructions.

The launcher warm-starts from a prior checkpoint or resumes saved weights. The recovered trainer does not save all optimizer/RNG state in the distributed checkpoint. The checkpoint itself does not contain a full training configuration or source revision. Therefore the launcher supports architectural and procedural provenance, but cannot establish exact retraining by itself.

`reference/training/` contains the recovered unified trainer, typed-record loader, evaluator, and graph preprocessing source. `reference/preprocessing/` contains property normalization and merge stages. These are archival files, excluded from the installed package. They reference input snapshots and modules that are not fully distributed here; their commands are not advertised as runnable in the research environment.

## Interpretation limits in historical evaluation

The historical loader computes numeric normalization statistics and support-based property filtering before its stride-based training/validation split. That differs from the new reference trainer's training-only fit. It also selects only the first configured number of property observations per compound and silently substitutes a dummy graph when a precomputed graph is missing. These historical behaviors must be accounted for in any reproduction or evaluation claim.

The historical normalizer's early numeric type identifier is `1`; the final architecture's numeric identifier is `2`. Do not feed early-stage records directly into the final model without the intervening type conversion. The released catalog was checked against all 72 shards of the v4 training input after the loader's numeric-sentinel and minimum-support rules: all compact-to-raw indices and property types matched.

No historical AUC, hazard-ranking, or molecular-generation score is asserted as reproduced here. Numerical agreement with upstream inference verifies the implementation for fixed inputs; it does not validate endpoint semantics, calibration, or generalization.

## Requirements for exact historical retraining

Exact retraining still needs immutable source/processed data and graph snapshots, dataset terms and transformations, the complete initialization chain, source revisions, effective resume history, optimizer and RNG state, and the original compute environment. Reproducing scientific performance also requires held-out split definitions, endpoint/class mappings, prediction panels, and benchmark predictions tied to that checkpoint. These artifacts are separate from the released inference bundle.
