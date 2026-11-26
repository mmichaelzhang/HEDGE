# -*- coding: utf-8 -*-
"""Legacy demonstration encoder retained for API compatibility.

It interleaves three edge types with node tokens, uses walk-position indices,
and averages encoded head states across views. It does not implement the later
reference core's graph-distance tokens, ordered relation tuples, or pair-wise
probability ensemble. See reference/README.md for the implementation mapping.
"""
import torch
import torch.nn as nn

from .sampling import sample_walks

N_CONN_TYPES = 3  # drug-target / drug-drug / target-target


class HedgeDTI(nn.Module):
    def __init__(self, N, d=128, nlayers=2, nheads=4, L=12, m=6, dropout=0.1):
        """
        N       number of nodes of interest (nD + nT)
        d       token width (nodes and connections share it, by design)
        L       retained nodes per sampled sequence -> 2L-1 tokens
        m       sampled views per head node (multi-view ensemble)
        """
        super().__init__()
        self.N, self.d, self.L, self.m = N, d, L, m
        self.node_emb = nn.Embedding(N, d)
        self.type_emb = nn.Embedding(N_CONN_TYPES, d)     # connection tokens
        self.pos_emb = nn.Embedding(L + 1, d)             # walk-position index
        self.conn_marker = nn.Parameter(torch.zeros(d))   # shared connection marker
        nn.init.normal_(self.node_emb.weight, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d, nheads, 2 * d, dropout, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, nlayers)
        self.norm = nn.LayerNorm(d)
        self.score = nn.Sequential(
            nn.Linear(3 * d, d), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d, 1))

    def encode(self, node_seq, type_seq):
        """(B,L) nodes + (B,L-1) connections -> (B,d) head-token representation."""
        B, L = node_seq.shape
        pos = torch.arange(L, device=node_seq.device)
        nemb = self.node_emb(node_seq) + self.pos_emb(pos)[None]   # (B,L,d)
        cemb = self.type_emb(type_seq) + self.conn_marker          # (B,L-1,d)
        tok = torch.empty(B, 2 * L - 1, self.d,
                          device=node_seq.device, dtype=nemb.dtype)
        tok[:, 0::2] = nemb                                        # interleave
        tok[:, 1::2] = cemb
        out = self.enc(tok)
        return self.norm(out[:, 0])                                # head token

    def node_reps(self, nodes, G, rng, device):
        """Contextual representation of each node, averaged over its m views."""
        offset, nbr, ntype = G
        ns, ts = sample_walks(nodes, offset, nbr, ntype, self.m, self.L, rng)
        ns = torch.as_tensor(ns, device=device)
        ts = torch.as_tensor(ts, device=device)
        h = self.encode(ns, ts).view(len(nodes), self.m, self.d).mean(1)
        return h

    def pair_score(self, hd, ht):
        """Interaction logit for a (drug, target) pair from their representations."""
        return self.score(torch.cat([hd, ht, hd * ht], -1)).squeeze(-1)
