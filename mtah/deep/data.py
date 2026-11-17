"""Player tensors for the deep players model.

A sample is a 4 s window (50 steps, every 2nd frame) of up to 12 players per team: position, velocity, speed,
presence and team. The model labels every step of the window, so each window gives 50 supervised frames.
"""
import numpy as np
from ..open_data import load_game
from ..features import orient, velocities, swap_teams

W, SUB = 100, 2          # 4 s window, every 2nd frame -> 50 steps
K_TEAM = 12              # players kept per team per window (most present first)
C = 7                    # x, y, vx, vy, speed, present, team


def frame_tensor(P, team):
    """F [T, S, C] float32 with NaN-free values and a presence channel. P oriented, metres."""
    V = velocities(P)
    T, S = P.shape[:2]
    F = np.zeros((T, S, C), np.float32)
    ok = np.isfinite(P[..., 0])
    vok = ok & np.isfinite(V[..., 0])
    F[..., 0] = np.where(ok, P[..., 0] / 52.5 - 1, 0); F[..., 1] = np.where(ok, P[..., 1] / 34.0 - 1, 0)
    F[..., 2] = np.where(vok, V[..., 0] / 5, 0); F[..., 3] = np.where(vok, V[..., 1] / 5, 0)
    F[..., 4] = np.where(vok, np.hypot(V[..., 0], V[..., 1]) / 5, 0)
    F[..., 5] = ok
    F[..., 6] = np.where(ok, np.where(team[None] == 0, 1.0, -1.0), 0)
    return F


def open_game(g, swap=False):
    """An open training match, oriented (and optionally mirrored with teams exchanged)."""
    d = load_game(g)
    P, team, ball, lab, period = d["P"], d["team"], d["ball"], d["label"].copy(), d["period"]
    blocks = [(np.flatnonzero(period == p)[0], np.flatnonzero(period == p)[-1]) for p in (1, 2)]
    P, ball = orient(P, team, ball, blocks)
    if swap:
        P, team, ball = swap_teams(P, team, ball)
        lab = np.where(lab == 0, 1, np.where(lab == 1, 0, lab))
    return dict(F=frame_tensor(P, team), team=team, lab=lab.astype(np.int64))


def batch(G, centres, rng=None, drop=0.0):
    """X [B, N, L, C], node mask [B, N], labels [B, L] for windows centred on `centres`."""
    F, team, lab = G["F"], G["team"], G["lab"]
    T = len(F)
    idx = np.clip(centres[:, None] + np.arange(-W // 2, W // 2, SUB)[None], 0, T - 1)      # [B, L]
    Xw = F[idx]                                                                       # [B, L, S, C]
    pres = Xw[..., 5].sum(1)                                                          # [B, S]
    sel = []
    for g in (0, 1):
        s = np.flatnonzero(team == g)
        order = np.argsort(-pres[:, s], 1, kind="stable")[:, :K_TEAM]
        sel.append(s[order])
    sel = np.concatenate(sel, 1)                                                      # [B, 2K]
    X = np.take_along_axis(Xw, sel[:, None, :, None], 2).transpose(0, 2, 1, 3)        # [B, N, L, C]
    mask = np.take_along_axis(pres, sel, 1) > 0
    if rng is not None and drop > 0:                                                  # tracker-like dropouts
        kill = rng.random(mask.shape) < drop
        X = X * (~kill)[:, :, None, None]; mask = mask & ~kill
    y = lab[idx]
    return X.astype(np.float32), mask, y
