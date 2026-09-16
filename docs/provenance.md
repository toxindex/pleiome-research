# Release provenance and review

`provenance.json` records the source repository revision and hashes of extracted architecture files. This repository begins with independent Git history. Deployment services, job workers, cloud configuration, application code, experiment logs, and unrelated models are excluded.

The released checkpoint preserves the original bytes. It contains model tensors, numeric normalization tensors, a property count, and a saved step. It contains no optimizer state, compound records, or free-text metadata. SHA-256 values cover each LFS part, the assembled checkpoint, architecture configuration, and metadata files.

The property catalog contains only index, raw/source property identifiers, source, type, descriptive name, accession, and metric. Support/positive/negative counts and measured activity records are omitted. All 181,007 compact-to-raw indices and property types were independently checked against the v4 input shards after reproducing the loader's filtering rules. Generic inherited names and missing identifiers are preserved; endpoint definitions and units have not been independently established for every property.

The deployment package supplied types for only 2,562 benchmark properties. That table disagreed with the complete training catalog for some properties. The research interface uses the audited training catalog and does not default every unlisted endpoint to binary. It also avoids the deployment wrapper's generic rule that negates every numeric prediction as a hazard score.

Architecture tensor names and numerical prediction behavior are retained. Deliberate changes include removing an unavailable alternate SELFIES encoder path, removing its obsolete smoke test, and using a boolean causal generation mask consistent with the boolean padding mask. The tokenizer's stored filename was made portable without changing vocabulary indices. Historical machine-specific copy instructions were removed from archival preprocessing comments.

New code provides checksum-verified assembly, strict safe checkpoint loading, explicit catalog/type validation, a CPU CLI, and a deterministic small training workflow. Recorded examples were produced by the upstream Pleiome inference implementation using explicit training-catalog types. Tests compare the new interface against those values and check individual/batched agreement.

The original repository's Git history and the research release are scanned for credentials. The model payload and catalog fields are reviewed separately because a text secret scanner does not establish binary artifact safety. This review does not establish that trained weights cannot memorize information or provide a scientific audit of all original datasets.
