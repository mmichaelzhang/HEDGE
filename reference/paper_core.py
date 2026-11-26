"""Small, auditable HEDGE core; see paper_core_README.md for explicit completions.

No data loading, feature extraction, fold selection, metrics, or training runner.
The caller must supply a TRAIN-ONLY graph and train-only positive labels.
Defaults define the stabilized reconstruction, not a literal reproduction claim.
"""
from collections import deque
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class Sample:
    nodes: Tuple[int, ...]                  # exactly one drug, at index zero
    connections: Tuple[Tuple[int, ...], ...]  # ordered relation tuples
    distances: Tuple[int, ...]              # full training graph shortest paths


class TrainGraph:
    """Directed CSR view. For an undirected HBN supply both directions explicitly.

    Use uncapped CSR for distances; independently capped outgoing neighborhoods
    change graph distances and can destroy symmetry. This class never reads labels.
    """
    def __init__(self, offset, neighbor, relation, n_drugs, n_targets):
        self.offset = np.asarray(offset, dtype=np.int64)
        self.neighbor = np.asarray(neighbor, dtype=np.int64)
        self.relation = np.asarray(relation, dtype=np.int64)
        self.n_drugs, self.n_targets = int(n_drugs), int(n_targets)
        self.n_nodes = len(self.offset) - 1
        e = int(self.offset[-1])
        if (self.offset[0] != 0 or np.any(np.diff(self.offset) < 0)
                or len(self.neighbor) < e or len(self.relation) < e):
            raise ValueError("Invalid CSR offsets/edge arrays")
        if e and (np.any(self.neighbor[:e] < 0)
                  or np.any(self.neighbor[:e] >= self.n_nodes)
                  or np.any(self.relation[:e] < 0)):
            raise ValueError("Invalid node or relation ID")
        if not 0 < self.n_drugs <= self.n_drugs + self.n_targets <= self.n_nodes:
            raise ValueError("Expected [drugs][targets][auxiliary] node layout")
        self._distance_cache = {}

    def shortest_paths(self, head):
        """Exact unweighted BFS in the supplied training graph, -1 if unreachable."""
        head = int(head)
        if head not in self._distance_cache:
            dist = np.full(self.n_nodes, -1, dtype=np.int64)
            dist[head] = 0
            queue = deque([head])
            while queue:
                u = queue.popleft()
                for v in self.neighbor[self.offset[u]:self.offset[u + 1]]:
                    v = int(v)
                    if dist[v] < 0:
                        dist[v] = dist[u] + 1
                        queue.append(v)
            self._distance_cache[head] = dist
        return self._distance_cache[head]

    def sample(self, heads, m, max_nodes, max_intervening, rng):
        """Uniform outgoing-edge walks, retaining a drug head and target hits only.

        A connection crosses at most max_intervening + 1 edges. All later drug
        visits are auxiliary, including a revisit to the head. An unsuccessful
        segment terminates the sample; it never creates an auxiliary tail or a
        synthetic self edge. max_nodes counts retained nodes including the head.
        """
        if m < 1 or max_nodes < 1 or max_intervening < 0:
            raise ValueError("Invalid sampling budget")
        samples = []
        for head in map(int, heads):
            if not 0 <= head < self.n_drugs:
                raise ValueError("Sampling heads must be drugs")
            distances = self.shortest_paths(head)
            for _ in range(m):
                nodes, conns = [head], []
                current = head
                for _ in range(max_nodes - 1):
                    path, hit = [], None
                    for _ in range(max_intervening + 1):
                        lo, hi = self.offset[current:current + 2]
                        if lo == hi:
                            break
                        edge = int(rng.integers(lo, hi))
                        path.append(int(self.relation[edge]))
                        current = int(self.neighbor[edge])
                        if self.n_drugs <= current < self.n_drugs + self.n_targets:
                            hit = current
                            break
                    if hit is None:
                        break
                    nodes.append(hit)
                    conns.append(tuple(path))
                samples.append(Sample(tuple(nodes), tuple(conns),
                                      tuple(int(distances[v]) for v in nodes)))
        return samples


class TupleVocabulary:
    """Collision-free learned lookup IDs for ordered tuples, PAD=0.

    Freeze before creating optimizer. Fit on planned samples from the train-only
    graph (no labels needed). Unknown tuples fail loudly: no hash/mean collapse.
    Save tuples alongside checkpoints so table row identity is reproducible.
    """
    def __init__(self, tuples: Iterable[Tuple[int, ...]]):
        values = sorted(set(tuple(map(int, t)) for t in tuples))
        if any(not t or min(t) < 0 for t in values):
            raise ValueError("Real connections require nonempty nonnegative tuples")
        self.tuples = [()] + values
        self.ids = {t: i for i, t in enumerate(self.tuples)}

    @classmethod
    def from_samples(cls, samples):
        return cls(c for sample in samples for c in sample.connections)

    def __len__(self):
        return len(self.tuples)


@dataclass
class Batch:
    nodes: torch.Tensor
    connections: torch.Tensor
    distances: torch.Tensor
    valid: torch.Tensor

    @property
    def connection_valid(self):
        return self.valid[:, :-1] & self.valid[:, 1:]


def collate(samples: Sequence[Sample], vocabulary: TupleVocabulary, device="cpu"):
    if not samples:
        raise ValueError("At least one sample is required")
    b, length = len(samples), max(len(s.nodes) for s in samples)
    nodes = torch.zeros((b, length), dtype=torch.long, device=device)
    distances = torch.zeros_like(nodes)
    connections = torch.zeros((b, length - 1), dtype=torch.long, device=device)
    valid = torch.zeros_like(nodes, dtype=torch.bool)
    for i, sample in enumerate(samples):
        n = len(sample.nodes)
        if n < 1 or len(sample.connections) != n - 1 or len(sample.distances) != n:
            raise ValueError("Malformed concentrated sample")
        if sample.distances[0] != 0 or min(sample.distances) < 0:
            raise ValueError("All retained nodes must be reachable from the head")
        nodes[i, :n] = torch.tensor(sample.nodes, device=device)
        distances[i, :n] = torch.tensor(sample.distances, device=device)
        valid[i, :n] = True
        connections[i, :n - 1] = torch.tensor(
            [vocabulary.ids[c] for c in sample.connections], device=device)
    return Batch(nodes, connections, distances, valid)


@dataclass
class Encoded:
    batch: Batch
    nodes: torch.Tensor
    connections: torch.Tensor


class PaperCore(nn.Module):
    def __init__(self, n_nodes, n_connections, max_distance, d=128, nheads=4,
                 nlayers=2, dropout=0.1, features=None, trainable_node_ids=True):
        super().__init__()
        self.d = d
        self.node_embedding = nn.Embedding(n_nodes, d) if trainable_node_ids else None
        if self.node_embedding is not None:
            nn.init.normal_(self.node_embedding.weight, std=0.02)
        if features is None and self.node_embedding is None:
            raise ValueError("Supply features or trainable node IDs")
        self.register_buffer("features", None if features is None
                             else torch.as_tensor(features, dtype=torch.float32))
        self.feature_projection = None if features is None else nn.Linear(
            self.features.shape[1], d, bias=False)
        self.connection_embedding = nn.Embedding(n_connections, d, padding_idx=0)
        self.distance_embedding = nn.Embedding(max_distance + 1, d)
        self.connection_marker = nn.Parameter(torch.zeros(d))
        layer = nn.TransformerEncoderLayer(d, nheads, 2 * d, dropout,
                                          batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, nlayers,
                                             enable_nested_tensor=False)
        self.attention_vector = nn.Parameter(torch.empty(3 * d))
        nn.init.normal_(self.attention_vector, std=(3 * d) ** -0.5)
        self.predictor = nn.Sequential(nn.Linear(d, d, bias=False),
                                       nn.LeakyReLU(0.01), nn.Linear(d, 1, bias=False))

    def node_input(self, ids):
        result = None if self.node_embedding is None else self.node_embedding(ids)
        if self.features is not None:
            projected = self.feature_projection(self.features[ids])
            result = projected if result is None else result + projected
        return result

    def encode(self, batch: Batch):
        if batch.distances[batch.valid].max() >= self.distance_embedding.num_embeddings:
            raise ValueError("Distance table too small; distances must not be silently clipped")
        node_tokens = self.node_input(batch.nodes) + self.distance_embedding(batch.distances)
        conn_tokens = self.connection_embedding(batch.connections) + self.connection_marker
        b, length = batch.nodes.shape
        tokens = node_tokens.new_zeros(b, 2 * length - 1, self.d)
        tokens[:, ::2], tokens[:, 1::2] = node_tokens, conn_tokens
        token_valid = torch.zeros(tokens.shape[:2], dtype=torch.bool, device=tokens.device)
        token_valid[:, ::2], token_valid[:, 1::2] = batch.valid, batch.connection_valid
        states = self.encoder(tokens, src_key_padding_mask=~token_valid)
        # Output padding is ignored; valid queries never attend to padding keys.
        return Encoded(batch, states[:, ::2], states[:, 1::2])

    def pair_logit(self, head, target):
        """Eq. 10 before sigmoid, with a shared Hadamard MLP."""
        return self.predictor(head * target).squeeze(-1)

    def observation_loss(self, encoded, normalize=True, reduction="mean"):
        """Eq. 7 attention; Eq. 8 literal if normalize=False, stabilized otherwise.

        Normalization applies only to agreement, not attention inputs. This is
        explicitly a stabilized variant because submitted Eq. 8 uses raw dots.
        """
        h, g = encoded.nodes, encoded.connections
        mask = encoded.batch.connection_valid
        zero = h.sum() * 0.0
        if h.shape[1] == 1:
            return zero, h.new_zeros(mask.shape)
        triple = torch.cat((h[:, :-1], g, h[:, 1:]), -1)
        energy = F.leaky_relu(triple, negative_slope=0.01) @ self.attention_vector
        safe = energy.masked_fill(~mask, -torch.inf)
        has = mask.any(1)
        safe = torch.where(has[:, None], safe, torch.zeros_like(safe))
        alpha = torch.softmax(safe, dim=1) * mask
        endpoint = F.normalize(h, dim=-1) if normalize else h
        agree = (endpoint[:, :-1] * endpoint[:, 1:]).sum(-1)
        per_sample = -(alpha * agree).sum(1)
        if reduction == "sum":
            return per_sample.sum(), alpha
        if reduction != "mean":
            raise ValueError("reduction must be sum or mean")
        return per_sample.mean(), alpha


def unique_tails(encoded, row, allowed_targets=None):
    """Repeated occurrences: mean representations once per unique tail per view.

    This disambiguates h_t for repeated visits and prevents visit frequency from
    silently changing the InfoNCE candidate set or the ensemble view count.
    """
    valid = encoded.batch.valid[row]
    ids = encoded.batch.nodes[row, 1:][valid[1:]]
    reps = encoded.nodes[row, 1:][valid[1:]]
    if allowed_targets is not None:
        keep = torch.tensor([int(t) in allowed_targets for t in ids],
                            dtype=torch.bool, device=ids.device)
        ids, reps = ids[keep], reps[keep]
    unique = torch.unique(ids, sorted=True)
    if len(unique) == 0:
        return unique, reps
    return unique, torch.stack([reps[ids == target].mean(0) for target in unique])


def contrastive_loss(encoded, train_positives: Mapping[int, Iterable[int]],
                     tau=0.2, normalize=False, reduction="mean", allowed_targets=None):
    """Eq. 6 over targets in each sample; positives use GLOBAL target IDs.

    Samples without any observed positive contribute zero. Mean divides by all
    samples, matching a batch mean of the submitted sample-sum objective.
    No claim of unbiasedness relative to full-target InfoNCE is made.
    """
    if tau <= 0:
        raise ValueError("tau must be positive")
    total = encoded.nodes.sum() * 0.0
    positives = {int(u): set(map(int, values)) for u, values in train_positives.items()}
    for row in range(len(encoded.batch.nodes)):
        head = int(encoded.batch.nodes[row, 0])
        ids, tails = unique_tails(encoded, row, allowed_targets)
        positive = torch.tensor([int(t) in positives.get(head, set()) for t in ids],
                                dtype=torch.bool, device=tails.device)
        if not positive.any():
            continue
        anchor = encoded.nodes[row, 0]
        if normalize:
            anchor, tails = F.normalize(anchor, dim=-1), F.normalize(tails, dim=-1)
        logits = tails @ anchor / tau
        total = total - F.log_softmax(logits, dim=0)[positive].mean()
    if reduction == "sum":
        return total
    if reduction != "mean":
        raise ValueError("reduction must be sum or mean")
    return total / len(encoded.batch.nodes)


def decoder_bce(model, encoded, train_positives, pos_weight=None, allowed_targets=None):
    """Explicit additional supervised decoder training; NOT part of submitted Eq.9.

    Only sampled unique head-tail candidates are used. Nonpositive means
    unlabeled-as-negative, not experimentally confirmed inactive. Supply an
    externally defined candidate mask/loss if only measured negatives are wanted.
    """
    logits, labels = [], []
    positives = {int(u): set(map(int, v)) for u, v in train_positives.items()}
    for row in range(len(encoded.batch.nodes)):
        head = int(encoded.batch.nodes[row, 0])
        ids, tails = unique_tails(encoded, row, allowed_targets)
        if len(ids):
            logits.append(model.pair_logit(encoded.nodes[row, 0], tails))
            labels.extend(float(int(t) in positives.get(head, set())) for t in ids)
    if not logits:
        return encoded.nodes.sum() * 0.0
    values = torch.cat(logits)
    target = values.new_tensor(labels)
    weight = None if pos_weight is None else values.new_tensor(pos_weight)
    return F.binary_cross_entropy_with_logits(values, target, pos_weight=weight)


def ensemble_probabilities(model, batches: Sequence[Encoded], heads, targets,
                           fallback="error", fallback_target_states=None):
    """Eq. 11: mean PROBABILITY across views with the pair, not mean logit.

    Fallback 'global_target_mean' uses mean head context across its own views and
    mean target context across every sampled drug view in these batches. If a
    target is absent globally, caller must explicitly supply its fallback state.
    This completes an undefined manuscript case; it is not Eq.11 itself.
    Returns probability matrix, co-occurring-view counts, fallback-kind matrix
    (0=observed, 1=global mean, 2=caller-supplied target state).
    All target IDs here are global (n_drugs + target_index).
    """
    if fallback not in {"error", "global_target_mean"}:
        raise ValueError("Unknown fallback policy")
    pair_values, head_values, target_values = {}, {}, {}
    for encoded in batches:
        for row in range(len(encoded.batch.nodes)):
            head = int(encoded.batch.nodes[row, 0])
            anchor = encoded.nodes[row, 0]
            head_values.setdefault(head, []).append(anchor)
            ids, tails = unique_tails(encoded, row)
            probabilities = torch.sigmoid(model.pair_logit(anchor, tails))
            for target, state, prob in zip(ids.tolist(), tails, probabilities):
                pair_values.setdefault((head, target), []).append(prob)
                target_values.setdefault(target, []).append(state)
    head_means = {u: torch.stack(v).mean(0) for u, v in head_values.items()}
    target_means = {t: torch.stack(v).mean(0) for t, v in target_values.items()}
    rows, count_rows, kind_rows = [], [], []
    for head in map(int, heads):
        row, counts, kinds = [], [], []
        for target in map(int, targets):
            values = pair_values.get((head, target), [])
            counts.append(len(values))
            if values:
                row.append(torch.stack(values).mean())
                kinds.append(0)
                continue
            if fallback == "error":
                raise ValueError(f"Undefined non-cooccurring pair {(head, target)}")
            if head not in head_means:
                raise ValueError(f"No sampled head representation for {head}")
            state = target_means.get(target)
            kind = 1
            if state is None:
                if fallback_target_states is None or target not in fallback_target_states:
                    raise ValueError(f"Target {target} absent globally: supply explicit fallback state")
                state, kind = fallback_target_states[target], 2
            row.append(torch.sigmoid(model.pair_logit(head_means[head], state)))
            kinds.append(kind)
        rows.append(torch.stack(row))
        count_rows.append(counts)
        kind_rows.append(kinds)
    result = torch.stack(rows)
    return result, torch.tensor(count_rows, device=result.device), torch.tensor(kind_rows, device=result.device)


def assert_train_only_dti(graph, train_pairs, heldout_pairs, dti_relation=0):
    """Audit all represented DTI edges in both directions against fold labels.

    Pair target IDs are LOCAL here, matching standard fold NPZ files. This checks
    graph edges only; it cannot certify feature or external-network provenance.
    """
    allowed = {tuple(map(int, p)) for p in train_pairs}
    forbidden = {tuple(map(int, p)) for p in heldout_pairs}
    if allowed & forbidden:
        raise AssertionError("Train/heldout positive overlap")
    for u in range(graph.n_nodes):
        for edge in range(int(graph.offset[u]), int(graph.offset[u + 1])):
            if graph.relation[edge] != dti_relation:
                continue
            v = int(graph.neighbor[edge])
            pair = ((u, v - graph.n_drugs) if u < graph.n_drugs
                    else (v, u - graph.n_drugs))
            if pair not in allowed or pair in forbidden:
                raise AssertionError(f"Nontraining DTI edge in graph: {pair}")
