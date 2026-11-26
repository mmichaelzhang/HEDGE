# HEDGE-DTI

Heterogeneous Edge-tokenizeD Graph Encoder for drug–target interaction prediction.

[Paper](paper/HEDGE-DTI.pdf) · [Standalone appendix](paper/appendix.pdf) ·
[Complete supplemental details](paper/complete_supplemental_details.pdf)

## What is available

This repository contains two explicitly separated implementations:

- **`reference/`**: source-bound components of the later audited support/query
  implementation, including the concentrated sampler, ordered relation tuples,
  graph-distance tokens, Transformer, explicit-label objective, query-view
  reductions and stable probability ensemble. Core files retain their original
  bytes; extracted training components have source-segment hashes. A small CPU
  example and scientific contract tests run without benchmark data.
- **`hedge/` and `train_minimal.py`**: the earlier public demonstration, retained
  for compatibility. It uses a simpler graph and training loop. It is not the
  source of the later controlled experiment tables.

See [the reference implementation guide](reference/README.md) for source identity,
the exact objective and the requirements for a full table replication. Dataset
inputs, prepared banks, trained checkpoints and the full benchmark orchestration
are not included in this release.

## Quick start

Use Python 3.10+ in a virtual environment; install a PyTorch build appropriate for
your platform. The following commands require only the CPU:

```bash
python -m pip install -r requirements.txt
python reference/verify_sources.py
python -m unittest discover -s reference -p "test_*.py" -v
python -m unittest discover -s tests -v
python reference/synthetic_example.py
```

The example performs one synthetic optimization step and verifies a toy score
matrix. It neither downloads a dataset nor reports benchmark metrics.

## Reference workflow

1. Construct the support graph from optimization-training interactions and the
   permitted auxiliary relations. For query-hidden training, remove the current
   block's positive query edges in both directions, then rebuild walks and
   graph distances.
2. Retain a drug head and target occurrences from short walks. Compress each
   intervening path to its **ordered relation-type tuple**; auxiliary node
   identities are not retained by this compression.
3. Interleave node and connection tokens. Shared self-attention contextualizes
   the sequence; the shared pair decoder uses the head and target states.
4. Train on explicit positive and negative pairs. The exported full objective is
   BCE + 0.5 normalized contrastive loss + 0.3 normalized observation loss.
5. Average repeated target states within a view and pair probabilities across
   views. The runtime provides defined fallbacks for pairs with no shared view.

Global attention provides a different receptive field from local message
passing; the code alone does not establish a causal oversmoothing claim.
Short sequences bound per-view encoder cost, not the cost of building or storing
the graph, preparing all views, or scoring a large candidate universe.

## Evaluation and data

AUPR in the controlled experiments uses average precision. Retrieval evaluation
needs the full legal target candidate set, with label-blind target-ID tie
breaking: validation excludes optimization-training positives, and testing
excludes training and validation positives. P@K, recall@K, candidate ceilings,
and sampled-pair AUPR have different denominators. This release does not add a
new benchmark evaluator or silently substitute sampled negatives for the full
retrieval pool.

For the **legacy demonstration only**, one numeric `fold_k.npz` contains:

| Field | Shape and meaning |
|---|---|
| `nD`, `nT` | Positive integer scalar counts |
| `train_pos_mask` | Binary `(nD, nT)` matrix, exactly the training-positive pairs |
| `train_pos`, `test_pos`, `test_neg` | Integer `(N, 2)` local drug/target IDs |
| Optional `train_neg`, `val_pos`, `val_neg` | The same pair format, disjoint from the other supplied splits |

`hedge.data.load_fold` closes the NPZ archive, disables pickle, validates IDs and
split disjointness, and rejects any graph-mask edge outside training positives.
Upstream feature provenance still requires a separate dataset audit.

```bash
python train_minimal.py --root /path/to/folds --dataset luo --folds 10
```

This legacy command samples unlabeled negatives and has no validation checkpoint
selection or test-evaluation output. Its numbers must not be presented as a
reproduction of the later reference experiments.

## Source and maintenance

- [Reference source manifest](reference/SOURCE_MANIFEST.json)
- [Reference implementation and test details](reference/README.md)
- [Paper bundle](paper/README.md)

The maintenance update fixes small-graph top-k indexing and strengthens the
legacy numeric data contract. The original legacy model and loss computations
are otherwise retained. No credentials, private review correspondence or
benchmark participant data are part of the code release.

## License

MIT; see [LICENSE](LICENSE). Dataset and third-party software access remain
subject to their respective licenses.
