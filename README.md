# HEDGE-DTI

**H**eterogeneous **E**dge-tokenize**D** **G**raph **E**ncoder for drug–target
interaction prediction over heterogeneous biomedical networks.

> **Partial release.** This repository contains the core modules of the method —
> the concentrated sampler, the connection-aware transformer encoder, and the
> dual-view objective — together with the data interface needed to run them on
> your own folds. The full pipeline (dataset construction, feature extraction,
> tuned per-dataset configurations, the baseline suite, and the evaluation
> harness) will be released in full upon acceptance.

## Idea

A heterogeneous biomedical network is too large to encode as a whole, and the
evidence that explains an unobserved interaction is often several hops away.
HEDGE-DTI therefore:

1. **Samples instead of aggregating.** A sampler `Q : (G, v_h) -> S` maps the
   network and a head node to a multiset of short ordered sub-structures. Runs of
   intervening auxiliary nodes are collapsed into a single **connection** that
   keeps the ordered tuple of edge types, so no relational evidence is lost and
   no meta-path template is required. Cost is bounded independently of `|G|`.
2. **Tokenizes connections as first-class citizens.** Nodes and connections are
   embedded at the same width and interleaved into one sequence, each node
   augmented with a head-relative shortest-path position. A typed relation is a
   token, not an edge weight.
3. **Attends globally over the short sequence,** which reaches distant tails
   without deep stacking and structurally avoids over-smoothing.
4. **Ensembles the sampled views** in which a pair co-occurs.

## Layout

| file | paper section | contents |
|---|---|---|
| `hedge/sampling.py` | 4.1 / Alg. 1 | typed graph construction, concentrated walk sampler |
| `hedge/model.py`    | 4.2 / 4.3    | connection-aware transformer, multi-view prediction |
| `hedge/losses.py`   | 4.2.2        | contrastive + observation dual objective, listwise ranking |
| `hedge/data.py`     | 5.1          | fold interface and the leakage assertion |
| `train_minimal.py`  | —            | reference loop showing how the pieces compose |

## Data interface

Write one `fold_{k}.npz` per fold with `nD, nT, train_pos_mask, train_pos,
test_pos, test_neg` (see `hedge/data.py`). `train_pos_mask` must contain
**training positives only** — it is the array the sampler walks over, so any
held-out edge left in it is leakage. `hedge.data.check_no_leakage` asserts this.

Scores are ranked over the **full target space** excluding training positives.
Precision@K computed over a sampled candidate shortlist is not comparable to
Precision@K over the full space, and we do not report the former.

## Usage

```bash
pip install -r requirements.txt
python train_minimal.py --root /path/to/folds --dataset luo --folds 10
```

## License

MIT — see `LICENSE`.
