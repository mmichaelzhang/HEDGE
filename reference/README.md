# Source-bound reference core

This directory makes a portable subset of the later audited support/query
implementation available independently of the original cluster and datasets.
It is separate from the July demonstration in `hedge/`.

`SOURCE_MANIFEST.json` records SHA-256 and byte sizes. Six computational modules
are byte-identical to sources bound by the aligned experiment's frozen source
manifest. Two original synthetic test modules are also copied unchanged.
`main_components.py` contains unchanged source segments for `MainCore`,
`PermutationBatches`, `BlockPairSchedule`, `query_block_labels`, `objective`, and
the five-arm constants. Only their import environment is adapted: a small
namespace supplies `PaperCore` and `paper_fast` instead of importing the original
cluster runner. The manifest records each source module and segment hash.

## Run locally

Use Python 3.10 or later and the repository requirements. From the repository root:

```bash
python reference/verify_sources.py
python -m unittest discover -s reference -p "test_*.py" -v
python reference/synthetic_example.py
```

The example executes one **synthetic CPU update**, exercises all three query
branches, and scores a 2-by-4 toy matrix. It downloads no data, uses no GPU,
starts no benchmark, and reports no experimental performance. It is not a
replacement for the experiment's bank preparation or validation/test protocol.

## Modules and semantics

| Module | What it implements |
|---|---|
| `paper_core.py` | Directed CSR graph, concentrated walks, ordered relation-tuple vocabulary, exact graph distances, interleaved Transformer tokens, shared Hadamard decoder, attention-weighted observation loss |
| `paper_fast.py` | Vectorized losses and repeated-tail grouping, with synthetic value/gradient checks against the core |
| `stable_ensemble_score.py` | Finite-bank probability ensemble evaluated in log space, with global-target and absent-target fallbacks |
| `train_label_index.py` | Auxiliary contrastive candidates restricted to explicitly supplied optimization-training labels |
| `support_query_contract.py` | Deterministic five-block pair partition, bidirectional support-graph filtering, keyed query-view plans |
| `support_query_runtime.py` | Read-only packed contexts, CSR identity indices, differentiable query reduction and streaming scoring |
| `main_components.py` | Original five-block schedule and BCE/contrastive/observation objective; optional constant-tuple control |

The normalized full objective here is `BCE + 0.5 C + 0.3 O`. BCE supervises
explicit queried pairs even when they do not co-occur in a view. Auxiliary
contrastive labels come from the current query block; an unlabelled contextual
tail is not automatically a negative. Observation uses the unique encoded
views. The schedule completes a permutation of each block, including a short
last batch, before reshuffling.

For benchmark preparation, query-block positive edges must be removed in both
directions **before** recomputing walks and graph distances. Validation/test
positive edges are excluded from training graphs. Repeated target states are
averaged within each view; pair probabilities are averaged across co-occurring
views. With no co-occurrence, the runtime uses target states across the bank;
if a target is absent from every view, it uses the target input representation.
The stored score is log-odds of this mean probability, not an average of logits.

`constant_tuple=True` zeros the learned tuple output while retaining the shared
marker, parameters, graph, walk, distances, and initialization sequence. It is
not deletion of heterogeneous graph edges. The flag must be saved separately
from a model state dictionary when using that control.

## Boundaries for reproduction

The source subset does not supply licensed dataset files, molecule/protein
feature construction, fold definitions, complete prepared banks, benchmark
launchers, validation selection, or the baseline suite. A table replication
also needs the exact split IDs, feature construction, source graph, bank/index
hashes, objective, seed, training budget, checkpoint-selection rule, candidate
masks and metric implementation. These cannot be inferred from a model name.

The packed runtime preserves the original `int16` node-ID format and Luo-specific
1652-feature helper; it is not a generic 258K-node data loader. Dataset adapters
must define and validate their own formats and feature blocks. Fixed-length
attention bounds encoder cost per view, not graph preparation, memory for
all inputs, or the number of candidates to score.

The historical paper results and later controlled reconstructions have distinct
lineages. These tests establish implementation contracts and source identity;
they do not establish that every historical table was produced by this code.
