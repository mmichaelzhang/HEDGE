# -*- coding: utf-8 -*-
"""Legacy demonstration loop, retained for compatibility.

This is not the later audited experiment runner. It samples unlabeled negatives
and has no validation selection or test evaluation. See reference/README.md for
the source-bound support/query core and its synthetic example.
"""
import argparse
import numpy as np
import torch

from hedge.sampling import build_typed_graph, sample_walks
from hedge.model import HedgeDTI
from hedge.losses import (bce_with_hard_negatives, listwise_ranking,
                          contrastive_infonce, observation)
from hedge.data import load_fold, check_no_leakage


def train_one_fold(fold, device, epochs=40, d=128, L=12, m=6, neg_k=10, bs=256,
                   lam_list=1.0, lam_con=0.5, lam_obs=0.3, seed=0):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    nD, nT = int(fold["nD"]), int(fold["nT"])
    mask = fold["train_pos_mask"]

    offset, nbr, ntype, N = build_typed_graph(mask)
    G = (offset, nbr, ntype)

    pos_by_drug = {}
    for d0, t0 in fold["train_pos"]:
        pos_by_drug.setdefault(int(d0), []).append(int(t0) + nD)
    train_drugs = np.array(sorted(pos_by_drug.keys()))
    all_t = np.arange(nD, nD + nT)

    model = HedgeDTI(N, d=d, L=L, m=m).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)

    for ep in range(epochs):
        model.train()
        rng.shuffle(train_drugs)
        tot, nb = 0.0, 0
        for s in range(0, len(train_drugs), bs):
            heads = train_drugs[s:s + bs]
            rows, labels, drow, uniq = [], [], [], {}

            def uid(n):
                if n not in uniq:
                    uniq[n] = len(uniq)
                return uniq[n]

            for r, dg in enumerate(heads):
                uid(int(dg))
                pos = pos_by_drug[int(dg)]
                n_want = neg_k * max(len(pos), 1)
                pool = rng.choice(all_t,
                                  size=min(n_want + len(pos) + 1, len(all_t)),
                                  replace=False).tolist()
                negs = [x for x in pool if mask[int(dg), x - nD] == 0][:n_want]
                for t in pos:
                    rows.append((uid(int(dg)), uid(int(t)))); labels.append(1.0); drow.append(r)
                for t in negs:
                    rows.append((uid(int(dg)), uid(int(t)))); labels.append(0.0); drow.append(r)

            nodes = np.array(list(uniq.keys()), np.int64)
            H = model.node_reps(nodes, G, rng, device)
            ridx = torch.tensor([a for a, _ in rows], device=device)
            tidx = torch.tensor([b for _, b in rows], device=device)
            y = torch.tensor(labels, device=device)
            drow_t = torch.tensor(drow, device=device)
            logit = model.pair_score(H[ridx], H[tidx])

            l_bce = bce_with_hard_negatives(logit, y)
            l_list = listwise_ranking(logit, y, drow_t, len(heads))
            l_con = contrastive_infonce(H[ridx], H[tidx], y, drow_t, len(heads))

            # observational view over sampled connections
            wn = nodes[rng.integers(0, len(nodes), size=min(256, len(nodes)))]
            ns, _ = sample_walks(wn, offset, nbr, ntype, 1, L, rng)
            a = torch.as_tensor(ns[:, :-1].reshape(-1), device=device)
            b = torch.as_tensor(ns[:, 1:].reshape(-1), device=device)
            alln = torch.unique(torch.cat([a, b]))
            hmap = {int(x): i for i, x in enumerate(alln.tolist())}
            Ho = model.node_reps(alln.cpu().numpy(), G, rng, device)
            ai = torch.tensor([hmap[int(x)] for x in a.tolist()], device=device)
            bi = torch.tensor([hmap[int(x)] for x in b.tolist()], device=device)
            bneg = torch.randint(0, len(alln), (len(ai),), device=device)
            l_obs = observation(Ho, ai, bi, bneg)

            loss = l_bce + lam_list * l_list + lam_con * l_con + lam_obs * l_obs
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            tot += float(loss); nb += 1
        sched.step()
        if ep % 5 == 0 or ep == epochs - 1:
            print(f"  ep{ep:03d} loss={tot / max(nb, 1):.4f}", flush=True)
    return model, G, (nD, nT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="directory containing <dataset>/fold_k.npz")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--folds", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--gpu", type=int, default=0)
    a = ap.parse_args()

    device = torch.device(f"cuda:{a.gpu}" if torch.cuda.is_available() else "cpu")
    for k in range(a.folds):
        fold = load_fold(a.root, a.dataset, k)
        check_no_leakage(fold)
        print(f"[{a.dataset}] fold {k} on {device}", flush=True)
        train_one_fold(fold, device, epochs=a.epochs, seed=k)


if __name__ == "__main__":
    main()
