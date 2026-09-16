# Data contract for the reference trainer

The CSV schema is `smiles,property_id,type,value,split`. A row is one observed value of a property for a molecule. `type` is `binary` or `numeric`; values must be finite, and binary values must be 0 or 1. `split` is `train`, `validation`, or `test`.

```csv
smiles,property_id,type,value,split
C,example_atoms,numeric,1,train
CC,example_atoms,numeric,2,train
CCC,example_atoms,numeric,3,validation
CCCC,example_atoms,numeric,4,test
```

The reader canonicalizes SMILES with RDKit and rejects molecules that appear in multiple partitions. Duplicate molecule/property rows are rejected, including repeated identical observations. Different types for the same property are rejected. Missing properties are absent observations; they are never filled with negative labels. Canonical-SMILES grouping does not establish scaffold separation or resolve all salt/tautomer equivalences. Define those policies before constructing your partitions.

The SELFIES vocabulary, sorted property index, and per-property numeric mean and population standard deviation are fitted on training rows only. Standard deviations have a floor of `1e-3`. Validation/test properties and symbols absent from training cause an error. Overlong SELFIES sequences cause an error; the reference does not silently truncate them. The supplied pretrained tokenizer/catalog are separate from this new experiment.

Training groups observations by molecule. A configured fraction of property values is revealed as context; at least one observed slot per molecule remains masked. Loss scores only masked, observed targets. Generation receives the molecule's observed property profile and reconstructs its SELFIES sequence. Validation and test property prediction hide all values. Each molecule's requested panel consists of its observed properties, so panel membership remains part of this evaluation protocol.

Validation selects the saved checkpoint using mean per-molecule structure-only prediction loss with the architecture's type-specific losses and uncertainty weighting disabled. Test records do not select a checkpoint. Binary outputs include per-property log loss and AUC when both classes occur; numeric outputs include RMSE and MAE on the original input scale. Toy metrics are software checks, not toxicology performance estimates.

The implementation loads the CSV in memory and validates encodability across all partitions before starting training. It is intended for inspectable local experiments, not the full historical corpus. Data snapshot distribution, source licenses, endpoint definitions, unit harmonization, and label thresholds remain the experiment author's responsibility.
