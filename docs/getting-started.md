# Start here: study and integrate Pleiome

## Before a first research meeting

Start with the [README](../README.md), [architecture](architecture.md), and [model card](../MODEL_CARD.md). Then follow [local reproduction](local-reproduction.md) to download the Git LFS weight parts, install the locked environment, and run `pleiome reproduce`. Reading the entire repository is unnecessary preparation.

For a first code walkthrough, read [the inference wrapper](../src/pleiome_research/inference.py), [graph encoder](../src/pleiome/dmpnn_encoder.py), and [unified model](../src/pleiome/unified_model.py). Read [the data contract](data.md) and [training notes](training.md) before designing a new experiment. Historical source in `reference/` is supplementary material.

Bring a small endpoint panel, the source definitions and units for those endpoints, and a proposed compound-level evaluation split. The bundled checks reproduce checkpoint inference and a small artificial training experiment. Exact historical retraining and scientific benchmark results remain unverified.

## Interpret a prediction

The checkpoint catalog contains 181,007 entries. Select endpoints using the complete bundled catalog; inclusion in that catalog does not establish endpoint validity or predictive accuracy.

| Catalog type | Count | Public inference output |
| --- | --- | --- |
| `binary` | 164,967 | `probability`: model probability of class 1, between 0 and 1 |
| `numeric` | 14,323 | `value`: model estimate after reversing stored mean/std normalization |
| `categorical` | 1,717 | Rejected because source class-index mappings are missing |

Class 1 follows the source endpoint's label definition. Numeric output retains the original training scale, including any earlier source transformation. Units, valid ranges, and hazard direction require source definitions. Neither output type is automatically a calibrated hazard score. Some catalog entries represent ancillary fields rather than useful scientific endpoints.

The Python `Predictor.predict` result contains `smiles`, `rounds`, and a `predictions` array. Each entry includes `index`, `raw_property_id`, `property_id`, `source`, `type`, `name`, `accession`, and `metric`, plus either `probability` or `value`. Missing source metadata remains null. `Predictor.predict_batch` returns one such result per molecule. The CLI emits an array of these per-molecule results, including when given one SMILES.

Use `pleiome catalog --search ...` to find endpoints. CLI `--properties` values are compact catalog indices, not source database IDs. Preserve the source identifiers with exported results. Match source definitions before comparing an endpoint with ToxTransformer or another tool; numeric indices cannot be transferred between models.

Record the Git revision, manifest checksum, structure preprocessing, complete property panel and its order, decoding rounds, and device/environment. Panel composition affects predictions even with one round. Additional rounds condition on thresholded model predictions, with the least-confident binary predictions revealed first. These generated labels are not observed assay values. The architecture includes a SELFIES generator, but this release does not validate its generation quality.

## Public projects and hosted services

Both research repositories are public on their `main` branches. No GitHub invitation or ToxIndex account is required to read the code or download the released weights. Earlier links to `toxindex/toxtransformer` and `toxindex/pleiome` refer to private development repositories and may return 404. Use the research URLs for public documentation and local experiments.

The released checkpoint is identified by its artifact manifest. A research checkout does not establish which revision, preprocessing, property context, or output adapter a hosted service is currently using. Compare those settings explicitly before expecting local and hosted predictions to agree.

| Resource | Purpose |
| --- | --- |
| [ToxTransformer research](https://github.com/toxindex/toxtransformer-research) | SELFIES-based model with 6,647 binary property endpoints |
| [Pleiome research](https://github.com/toxindex/pleiome-research) | Graph-based model with binary/numeric prediction and a SELFIES generation architecture |
| [ToxIndex](https://toxindex.com/) | Product overview and a public link for model directories |
| [Insilica](https://insilica.co/) | Company background |
| [ToxIndex platform](https://platform.toxindex.com/) | Hosted application |
| [Gateway documentation](https://gateway.toxindex.com/docs) and [OpenAPI](https://gateway.toxindex.com/openapi.json) | Hosted prediction interfaces; separate from these local Python interfaces |

Suggested directory description: “ToxIndex provides access to chemical and toxicological data, prediction tools, and scientific literature through AI-assisted search and workflows.” Link this description to https://toxindex.com/; use the platform URL for an application sign-in link.

## License and hosted access

The research code and released weights use the repository's MIT license and can be downloaded for local use without a hosted subscription. Original source datasets retain their own terms. Local execution requires your own compute resources.

Hosted access is governed separately. The [ToxIndex pricing page](https://toxindex.com/pricing) describes a free platform preview and paid plans quoted for the deployment and scope of work. It does not promise free ToxTransformer/Pleiome API calls or establish partner-specific terms. Consult the current page and the account's agreement for hosted access.

For an existing hosted ToxTransformer integration, create an API key at [account settings](https://platform.toxindex.com/settings/keys). Send `Authorization: Bearer $TOXINDEX_API_KEY` on both `POST https://gateway.toxindex.com/v1/runs/toxtransformer` with JSON `{"smiles":"CCO"}` and subsequent `GET /v1/runs/{run_id}` polls. The run statuses are `queued`, `running`, `completed`, and `failed`; read `result` on completion and `error` on failure. Keep the key outside the repository. Use the live gateway documentation for the hosted response schema and current access rules. This release does not specify every other model available through ToxIndex.
