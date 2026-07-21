# -*- coding: utf-8 -*-
"""Concentrated heterogeneous graph sampling  (paper Sec. 4.1, Algorithm 1).

Implements the sampler Q : (G, v_h) -> S of the preliminaries, which maps the
heterogeneous network and a head node to a multiset of ordered sub-structures.
Each sub-structure is an alternating node--connection sequence; a connection
carries the *type* of the traversed relation, so no relational evidence is lost
when intervening nodes are collapsed. No meta-path template is required.
"""
import numpy as np

# edge (connection) types kept by the sampler
DT, DD, TT = 0, 1, 2  # drug-target, drug-drug, target-target


def build_typed_graph(mask, topk=10):
    """Build a CSR typed graph over N = nD + nT nodes from the *training* adjacency.

    Parameters
    ----------
    mask : (nD, nT) array
        Leak-free training drug-target adjacency. Only training positives may be
        present -- held-out edges must never enter the message/sampling graph.
    topk : int
        Degree cap for the induced co-interaction edges.

    Returns
    -------
    offset : (N+1,) int64 CSR offsets
    nbr    : (E,)   int64 neighbour ids
    ntype  : (E,)   int8  connection type per incident edge
    N      : int    number of nodes
    """
    nD, nT = mask.shape
    N = nD + nT
    M = mask.astype(np.float32)
    adj = [[] for _ in range(N)]

    di, ti = np.where(M > 0)
    for d, t in zip(di.tolist(), ti.tolist()):
        tt = t + nD
        adj[d].append((tt, DT))
        adj[tt].append((d, DT))

    # induced same-type relations (co-interaction); skipped on very large graphs
    if nD <= 6000:
        dd = M @ M.T
        np.fill_diagonal(dd, 0)
        for d in range(nD):
            row = dd[d]
            idx = np.argpartition(-row, min(topk, N - 1))[:topk]
            for j in idx[row[idx] > 0]:
                adj[d].append((int(j), DD))
    if nT <= 6000:
        tc = M.T @ M
        np.fill_diagonal(tc, 0)
        for t in range(nT):
            row = tc[t]
            idx = np.argpartition(-row, min(topk, N - 1))[:topk]
            for j in idx[row[idx] > 0]:
                adj[t + nD].append((int(j) + nD, TT))

    offset = np.zeros(N + 1, np.int64)
    for i in range(N):
        offset[i + 1] = offset[i] + len(adj[i])
    E = int(offset[-1])
    nbr = np.zeros(max(E, 1), np.int64)
    ntype = np.zeros(max(E, 1), np.int8)
    for i in range(N):
        for j, (nb, ty) in enumerate(adj[i]):
            nbr[offset[i] + j] = nb
            ntype[offset[i] + j] = ty
    return offset, nbr, ntype, N


def sample_walks(heads, offset, nbr, ntype, m, L, rng):
    """Draw m concentrated typed walks of length L from each head node.

    Fully vectorised over (heads x m): every walk advances in lock-step, so the
    cost is O(m * L * |heads|) and is independent of the size of the network.

    Returns
    -------
    node_seq : (len(heads)*m, L)   int64 -- nodes of interest v_1..v_L, v_1 = head
    type_seq : (len(heads)*m, L-1) int64 -- connection types c_1..c_{L-1}
    """
    cur = np.repeat(heads, m).astype(np.int64)
    seqs = [cur.copy()]
    types = []
    deg_all = offset[1:] - offset[:-1]
    for _ in range(L - 1):
        deg = deg_all[cur]
        has = deg > 0
        r = (rng.random(len(cur)) * np.maximum(deg, 1)).astype(np.int64)
        pick = offset[cur] + np.minimum(r, np.maximum(deg - 1, 0))
        pick = np.minimum(pick, len(nbr) - 1)
        nxt = np.where(has, nbr[pick], cur)          # dead ends stay put
        ety = np.where(has, ntype[pick].astype(np.int64), DT)
        seqs.append(nxt.copy())
        types.append(ety.copy())
        cur = nxt
    return np.stack(seqs, 1), np.stack(types, 1)
