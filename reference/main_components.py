"""Extracted unchanged scientific components; see SOURCE_MANIFEST.json.

Imports replace the original runner-only dependencies. No experiment launcher,
dataset paths, or approval machinery is included in this portable module.
"""
from types import SimpleNamespace
import numpy as np
import torch
from torch.nn import functional as F
import paper_fast
import support_query_contract
import train_label_index
from paper_core import PaperCore, Encoded
from support_query_runtime import reduce_encoded_plans

pilot = SimpleNamespace(PaperCore=PaperCore, paper_fast=paper_fast)
BATCH_SIZE = 256


class PermutationBatches:
    """Complete each permutation, including its short final batch, then reshuffle."""
    def __init__(self, size, bs=BATCH_SIZE, seed=0):
        if size < 1 or bs < 1:
            raise ValueError('Empty sampling universe')
        self.size, self.bs = size, bs
        self.rng = np.random.default_rng(seed)
        self.order, self.offset, self.cycles = None, 0, 0

    def next(self):
        if self.order is None or self.offset >= self.size:
            self.order = self.rng.permutation(self.size)
            self.offset = 0
            self.cycles += 1
        out = self.order[self.offset:self.offset + self.bs]
        self.offset += len(out)
        return out

    def state(self):
        return {'size': self.size, 'bs': self.bs, 'order': self.order,
                'offset': self.offset, 'cycles_started': self.cycles,
                'numpy_rng': self.rng.bit_generator.state}


class BlockPairSchedule:
    def __init__(self, sizes, fold=0, batch_size=256):
        if len(sizes) != 5 or any(size <= 0 for size in sizes):
            raise ValueError('Exactly five nonempty query blocks are required')
        self.samplers = [PermutationBatches(size, bs=batch_size,
            seed=int(np.random.SeedSequence([support_query_contract.MASTER_SEED, 41, fold, b]).generate_state(1)[0]))
            for b, size in enumerate(sizes)]
        self.update = 0

    def next(self):
        block = self.update % 5
        self.update += 1
        return block, self.samplers[block].next()

    def state(self):
        return {'update': self.update, 'blocks': [s.state() for s in self.samplers]}


ARMS = ('exposed_BCE', 'hidden_BCE', 'hidden_con', 'hidden_full', 'hidden_full_constant')


STEPS = 6400


VALIDATION_STEPS = (0, 50, 100, 200, 400, 800, 1600, 3200, 6400)


WEIGHTS = {'exposed_BCE': (0., 0.), 'hidden_BCE': (0., 0.), 'hidden_con': (.5, 0.),
           'hidden_full': (.5, .3), 'hidden_full_constant': (.5, .3)}


class MainCore(pilot.PaperCore):
    """Original core, optionally zeroing tuple output while retaining its marker."""
    def __init__(self, *args, constant_tuple=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.constant_tuple = bool(constant_tuple)

    def encode(self, batch):
        if not self.constant_tuple:
            return super().encode(batch)
        if batch.distances[batch.valid].max() >= self.distance_embedding.num_embeddings:
            raise ValueError('Distance table too small')
        node_tokens = self.node_input(batch.nodes) + self.distance_embedding(batch.distances)
        conn_tokens = self.connection_embedding(batch.connections) * 0. + self.connection_marker
        size, length = batch.nodes.shape
        tokens = node_tokens.new_zeros(size, 2 * length - 1, self.d)
        tokens[:, ::2], tokens[:, 1::2] = node_tokens, conn_tokens
        valid = torch.zeros(tokens.shape[:2], dtype=torch.bool, device=tokens.device)
        valid[:, ::2], valid[:, 1::2] = batch.valid, batch.connection_valid
        states = self.encoder(tokens, src_key_padding_mask=~valid)
        return Encoded(batch, states[:, ::2], states[:, 1::2])


def query_block_labels(block, nd, nt):
    # Accept only this block, never the whole fold or validation/test labels.
    return train_label_index.declared_training_mask(block['positive'], block['negative'], nd, nt)


def objective(model, encoded, plans, view_ids, labels, block_labels, nd, pos_weight, arm):
    logits = reduce_encoded_plans(model, encoded, plans, view_ids)
    bce = F.binary_cross_entropy_with_logits(logits, logits.new_tensor(labels),
                                            pos_weight=logits.new_tensor(pos_weight))
    known, positive_map = block_labels
    index = train_label_index.training_tail_index(encoded.batch, positive_map, known, nd)
    cw, ow = WEIGHTS[arm]
    con = pilot.paper_fast.contrastive_loss(encoded, index, tau=.2, normalize=True) if cw else logits.new_zeros(())
    obs = model.observation_loss(encoded, normalize=True)[0] if ow else logits.new_zeros(())
    total = bce + cw * con + ow * obs
    positives = index.positive.detach().cpu().numpy()
    pr = np.bincount(index.row_cpu, weights=positives.astype(np.int64), minlength=index.batch_size)
    nr = np.bincount(index.row_cpu, weights=(~positives).astype(np.int64), minlength=index.batch_size)
    stats = {'unique_views': index.batch_size, 'current_block_positive_tail_groups': int(positives.sum()),
             'current_block_negative_tail_groups': int((~positives).sum()),
             'positive_bearing_views': int((pr > 0).sum()),
             'positive_and_negative_bearing_views': int(((pr > 0) & (nr > 0)).sum())}
    return total, (bce, con, obs), stats
