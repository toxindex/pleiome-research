# Model architecture

## Structure and property prediction

RDKit converts a SMILES string into atom and bond features. Each undirected bond becomes two directed edges with an explicit reverse-edge mapping. The encoder initializes edge states from source-atom and bond features. At each message update, an edge receives messages entering its source atom with the reverse edge subtracted. After eight updates, incoming edge states are summed at each atom and combined with that atom's features. Batched molecules form disconnected graphs; the encoder returns padded atom memories and a padding mask.

The released model uses 43 atom features and 12 bond features. Atom features cover element, degree, charge, hybridization, hydrogen count, aromaticity, ring membership, and scaled atomic mass. Bond features include type, conjugation, ring membership, and stereo. Atom chirality is not explicitly encoded; the model must not be assumed to distinguish all stereoisomers.

A property slot combines a property embedding, type embedding, and either an encoded known value or a learned mask embedding. Eight transformer decoder layers apply self-attention across property slots and cross-attention to atom memories. There is no causal order among property slots. Masked target values cannot enter their input embeddings; the tests check that changing those target values leaves predictions unchanged.

The architecture defines five type handlers. Binary heads use a per-property weight and bias to produce logits. Numeric heads predict a mean and log variance in standardized space. Categorical heads emit class logits. Association and profile handlers remain in the checkpoint architecture, but the audited training catalog contains only binary, numeric, and categorical entries. Presence of a handler does not establish that its task was trained or validated.

## Generation

The structure decoder conditions on structure-free property slots whose values are marked known. Six causal transformer decoder layers predict SELFIES tokens autoregressively. The property/type/value embeddings are shared with prediction. Teacher-forced generation loss is cross-entropy against the next SELFIES token. The small reference trainer adds this loss to masked property prediction; generation causality is tested independently.

The released generator has a 2,467-token SELFIES vocabulary and 128 learned token-position embeddings. Generation code is available through `UnifiedModel`; molecular validity, novelty, and achievement of requested properties are not validated by this release. A generated structure needs independent chemical and experimental assessment.

## Configuration and checkpoint

`models/pleiome/architecture.json` specifies width 768, eight attention heads, eight graph updates, eight property decoder layers, and six structure decoder layers. Attention head count and graph update count are configuration values; tensor shapes alone cannot recover them. The recovered v4 launcher and upstream inference settings support these values.

The checkpoint has 557,464,475 trainable parameters and one 32-element Fourier-frequency buffer, for 557,464,507 state-dictionary tensor elements. It also stores the property count, saved step, and numeric normalization arrays. The research loader uses `weights_only=True`, memory-mapped loading, and strict state-dictionary matching. It never silently fills missing weights.

The public prediction interface supports the 164,967 binary and 14,323 numeric endpoints identified by the audited catalog. The 1,717 categorical endpoints are retained in the catalog but rejected by the interface because source class-index mappings are not included.
