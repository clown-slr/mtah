"""Change-aware Viterbi decoder: per-frame evidence (deep players model, tree ball-control model, two ball rules)
-> H / A / D per frame of each half, X elsewhere. Switch costs vary per frame with the tree change model.

    python -m mtah.decoder [out/submission.csv]
"""
import os, sys
import numpy as np
from .config import HALVES, N_FRAMES, PITCH, OUT, ensure
from .trees import predict as tree_probs
from .deep.predict import run as deep_probs

SETTINGS = dict(
    w_players=0.25,   # weight of the players model (frames are strongly correlated, so < 1)
    w_control=0.5,    # weight of the ball-control model
    k_out=1.0,        # evidence per frame for D while the tracked ball is off the pitch
    out_margin=0.5,   # m
    c_switch=6.0,     # cost of H <-> A
    c_dead=10.0,      # cost of entering or leaving D
    w_miss=0.25,      # weight of the tracker-lost-the-ball evidence (likelihood ratio estimated on the match itself)
    eps=0.02,         # probability floor
    b_switch=1.0,     # how strongly the change model moves the home <-> away cost
    b_dead=2.0,       # how strongly the change model moves the live <-> dead cost
    c_floor=0.5,      # lowest cost a switch can have
)


def viterbi(ll, cs, cd):
    """Viterbi over H, A, D with a per-frame cost: cs[t] for home <-> away, cd[t] for live <-> dead (paid at frame t)."""
    n = len(ll); score = ll[0].copy(); back = np.zeros((n, 3), np.int8)
    for t in range(1, n):
        c = np.array([[0, cs[t], cd[t]], [cs[t], 0, cd[t]], [cd[t], cd[t], 0]])
        cand = score[:, None] - c; back[t] = cand.argmax(0); score = cand.max(0) + ll[t]
    path = np.empty(n, np.int8); path[-1] = score.argmax()
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def switch_costs(r, p):
    """Lower the switch cost where the change model expects a change, raise it where it does not:
    cost - b * log(p_change / prior), clipped to [c_floor, 3 * cost]."""
    pc, prior = r["pc"], r["prior"]
    cs = p["c_switch"] - p["b_switch"] * np.log(np.clip(pc[:, 1], 1e-3, 1) / prior[1])
    cd = p["c_dead"] - p["b_dead"] * np.log(np.clip(pc[:, 2], 1e-3, 1) / prior[2])
    return np.clip(cs, p["c_floor"], 3 * p["c_switch"]), np.clip(cd, p["c_floor"], 3 * p["c_dead"])


def emissions(r, p):
    """Per-frame log-evidence for H / A / D."""
    p1 = np.clip(r["p1"], p["eps"], 1); q = r["q"]
    ll = p["w_players"] * np.log(p1)
    ok = np.isfinite(q)
    qc = np.clip(np.where(ok, q, 0.5), p["eps"], 1 - p["eps"])
    ll[:, 0] += np.where(ok, p["w_control"] * np.log(qc), 0.0)
    ll[:, 1] += np.where(ok, p["w_control"] * np.log(1 - qc), 0.0)
    b = r["ball"]                                   # a tracked ball well outside the lines is evidence for D
    out_m = np.maximum.reduce([-b[:, 0], b[:, 0] - PITCH[0], -b[:, 1], b[:, 1] - PITCH[1]])
    ll[:, 2] += np.where(np.isfinite(out_m) & (out_m > p["out_margin"]), p["k_out"], 0.0)
    # Ball visibility as evidence for D. The likelihood ratio is estimated on this match from the players model's soft
    # labels (no ground truth involved): the tracker loses the ball far more often while it is dead than in play.
    g = np.concatenate([np.arange(a, b + 1) for a, b in HALVES])
    pD = r["p1"][g, 2]; miss = ~r["filled"][g]
    pm_d = (pD * miss).sum() / pD.sum(); pm_a = ((1 - pD) * miss).sum() / (1 - pD).sum()
    lr_miss, lr_seen = np.log(pm_d / pm_a), np.log((1 - pm_d) / (1 - pm_a))
    ll[:, 2] += p["w_miss"] * np.where(r["filled"], lr_seen, lr_miss)
    return ll


def decode(r, **over):
    p = dict(SETTINGS, **over)
    ll = emissions(r, p)
    cs, cd = switch_costs(r, p)
    lab = np.full(N_FRAMES, "X", dtype="<U1")
    for a, b in HALVES:
        lab[a:b + 1] = np.array(["H", "A", "D"])[viterbi(ll[a:b + 1], cs[a:b + 1], cd[a:b + 1])]
    return lab


def write(path, lab):
    with open(path, "w") as fh:
        fh.write("frame,label\n")
        fh.writelines("%d,%s\n" % (i, l) for i, l in enumerate(lab))


def summary(lab):
    out = {}
    for a, b in HALVES:
        s = lab[a:b + 1]
        for i in np.r_[0, np.flatnonzero(s[1:] != s[:-1]) + 1]:
            out[str(s[i])] = out.get(str(s[i]), 0) + 1
    g = np.concatenate([lab[a:b + 1] for a, b in HALVES])
    return {"runs": out, "share": {k: round(float((g == k).mean()), 3) for k in "HAD"}}


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(OUT, "submission.csv")
    ensure(os.path.dirname(os.path.abspath(out)))
    r = dict(tree_probs())
    r["p1"] = deep_probs()          # players evidence: the deep model in place of the tree players model
    lab = decode(r)
    write(out, lab); print(out, summary(lab))
