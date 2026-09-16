# Historical source

These files preserve selected Pleiome training and preprocessing implementation for research. Their original hashes and recovery provenance are recorded in `provenance.json`; machine-specific paths and operational copy instructions have been removed where needed.

`training/` includes typed-record loading, unified prediction/generation training, evaluation, and graph construction. `preprocessing/` includes source normalization, type conversion, and dataset merging. These files are not installed with the package and are not covered by the runnable workflow's tests. They depend on historical datasets and surrounding modules that are not fully included.

Use `pleiome train` for the tested small workflow. See `docs/training.md` for historical split, normalization, filtering, and initialization limitations before adapting archival code.
