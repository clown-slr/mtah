"""Gradient-boosted models, trained only on the open matches (never on the competition match):
players model (H/A/D; its features also feed the change model), ball-control model (home vs away, live frames with
a ball) and change model (near a home<->away / live<->dead change within 0.48 s).

    python -m mtah.trees train      # train on the open matches, save models/models.pkl
    python -m mtah.trees predict    # apply to the competition match, cache per-frame probabilities
"""
import os, sys, pickle
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from .config import CACHE, MODELS, HALVES, ensure
from .open_data import load_game
from .tracking import load, interp_ball
from .features import player_features, ball_features, orient, swap_teams, velocities

CP_W = 12   # frames either side of a change that count as "near a change" (0.48 s)


def change_labels(lab, w=CP_W):
    """0: no change nearby; 1: home <-> away change within w frames; 2: live <-> dead change within w frames."""
    a, b = lab[:-1], lab[1:]
    ok = (a >= 0) & (b >= 0) & (a != b)
    team = np.r_[False, ok & (a < 2) & (b < 2)]; dl = np.r_[False, ok & ((a == 2) | (b == 2))]
    k = np.ones(2 * w + 1)
    near_t = np.convolve(team, k, "same") > 0; near_d = np.convolve(dl, k, "same") > 0
    return np.where(near_t, 1, np.where(near_d, 2, 0)).astype(np.int8)


def cp_frame(X1, X2):
    """Inputs of the change model: both feature sets side by side."""
    return pd.concat([X1.reset_index(drop=True), X2.reset_index(drop=True).add_prefix("b_")], axis=1)


def game_frames(g, rng):
    """Features and labels of one open match, as played and mirrored, every third frame."""
    d = load_game(g)
    P, team, ball, lab, period = d["P"], d["team"], d["ball"], d["label"].copy(), d["period"]
    blocks = [(np.flatnonzero(period == p)[0], np.flatnonzero(period == p)[-1]) for p in (1, 2)]
    P, ball = orient(P, team, ball, blocks)
    out = []
    for sw in (0, 1):
        if sw:
            P, team, ball = swap_teams(P, team, ball)
            lab = np.where(lab == 0, 1, np.where(lab == 1, 0, lab))
        Pd = P.copy()                                     # mimic the automatic tracking: players drop out for stretches
        n, S = P.shape[:2]
        for _ in range(int(n * S * 0.08 / 100)):
            t0 = rng.integers(0, n); s = rng.integers(0, S); Pd[t0:t0 + rng.integers(20, 180), s] = np.nan
        V = velocities(Pd)
        X1 = player_features(Pd, team, V); X2 = ball_features(Pd, team, ball, V)
        keep = lab >= 0
        sub = np.zeros(len(lab), bool); sub[::3] = True
        k = keep & sub
        out.append((X1[k], X2[k], lab[k], change_labels(lab)[k]))
    return out


def fit(X, y):
    m = HistGradientBoostingClassifier(learning_rate=0.06, max_iter=250, max_leaf_nodes=31, min_samples_leaf=400,
                                       l2_regularization=10.0, max_features=0.5, early_stopping=False, random_state=0)
    return m.fit(X, y)


def train():
    rng = np.random.default_rng(0)
    parts = [p for g in (1, 2, 3) for p in game_frames(g, rng)]
    X1 = pd.concat([p[0] for p in parts]); X2 = pd.concat([p[1] for p in parts])
    y = np.concatenate([p[2] for p in parts]); yc = np.concatenate([p[3] for p in parts])
    m1 = fit(X1, y); al = y < 2; m2 = fit(X2[al], (y[al] == 0).astype(int))
    X3 = cp_frame(X1, X2); m3 = fit(X3, yc)
    pickle.dump({"players": m1, "control": m2, "change": m3, "change_prior": np.bincount(yc, minlength=3) / len(yc),
                 "cols1": list(X1.columns), "cols2": list(X2.columns), "cols3": list(X3.columns)},
                open(os.path.join(ensure(MODELS), "models.pkl"), "wb"))
    print("trees: trained on %d frames (three open matches, each also mirrored)" % len(y))


def predict(max_gap=50, force=False):
    """Per-frame probabilities on the competition match: p1 (players, H/A/D), q (P(home in control), where a ball
    exists), pc (change classes), plus the ball track. Cached in cache/probs.npz."""
    cp = os.path.join(ensure(CACHE), "probs.npz")
    if os.path.exists(cp) and not force:
        z = np.load(cp); return {k: z[k] for k in z.files}
    d = load(); P = d["P"][..., :2].astype(np.float64); team = d["team"]
    xy, obs, filled = interp_ball(d["B"], max_gap)
    ball = np.where(filled[:, None], xy, np.nan)
    Po, bo = orient(P, team, ball, HALVES)
    V = velocities(Po)
    M = pickle.load(open(os.path.join(MODELS, "models.pkl"), "rb"))
    X1 = player_features(Po, team, V)[M["cols1"]]; X2 = ball_features(Po, team, bo, V)[M["cols2"]]
    p1 = M["players"].predict_proba(X1)
    q = np.full(len(P), np.nan); ok = np.isfinite(bo[:, 0])
    q[ok] = M["control"].predict_proba(X2[ok])[:, 1]
    pc = M["change"].predict_proba(cp_frame(X1, X2)[M["cols3"]])
    out = dict(p1=p1, q=q, ball=ball, obs=obs, filled=filled, pc=pc, prior=M["change_prior"])
    np.savez_compressed(cp, **out)
    return out


if __name__ == "__main__":
    step = sys.argv[1] if len(sys.argv) > 1 else "train"
    if step == "train":
        train()
    elif step == "predict":
        r = predict(force=True)
        g = np.concatenate([r["p1"][a:b + 1] for a, b in HALVES])
        print("trees: predicted the competition match; mean H/A/D (tree players model)", g.mean(0).round(3))
    else:
        sys.exit("usage: python -m mtah.trees [train|predict]")
