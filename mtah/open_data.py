"""Metrica Sports' open sample data (three anonymised matches, not the competition match): tracking plus
possession labels derived mechanically from their event data. No label is produced by hand.

    python -m mtah.open_data       # load, derive labels and cache
"""
import os, json
import numpy as np, pandas as pd
from .config import EXT_DIR, CACHE, ensure

L, W = 105.0, 68.0
ONBALL = {"PASS", "RECOVERY", "SET PIECE", "SHOT", "CARRY", "BALL LOST", "BALL OUT"}


def _csv_game(g):
    d = os.path.join(EXT_DIR, "Sample_Game_%d" % g)
    out = []
    for side in ("Home", "Away"):
        df = pd.read_csv(os.path.join(d, "Sample_Game_%d_RawTrackingData_%s_Team.csv" % (g, side)), skiprows=2)
        a = df.values.astype(np.float64)
        period = a[:, 0].astype(int)
        out.append(a[:, 3:-2].reshape(len(a), -1, 2))
        ball = a[:, -2:]
    ev = pd.read_csv(os.path.join(d, "Sample_Game_%d_RawEventsData.csv" % g))
    ev = pd.DataFrame({"team": ev["Team"].map({"Home": 0, "Away": 1}), "type": ev["Type"], "sub": ev["Subtype"].fillna(""),
                       "start": ev["Start Frame"] - 1, "end": ev["End Frame"] - 1})
    return out[0], out[1], ball, period, ev


def _game3():
    d = os.path.join(EXT_DIR, "Sample_Game_3")
    rows, fids = [], []
    with open(os.path.join(d, "Sample_Game_3_tracking.txt")) as fh:
        for line in fh:
            p = line.rstrip("\n").split(":")
            fids.append(int(p[0]))
            rows.append([float(v) for g in p[1].split(";") for v in g.split(",")] + [float(v) for v in p[2].split(",")])
    a = np.array(rows); fids = np.array(fids)
    n = fids.max()
    full = np.full((n, a.shape[1]), np.nan); full[fids - 1] = a
    pl = full[:, :-2].reshape(n, -1, 2)
    ball = full[:, -2:]
    js = json.load(open(os.path.join(d, "Sample_Game_3_events.json")))["data"]
    teams = sorted({e["team"]["id"] for e in js})
    ev = pd.DataFrame({"team": [teams.index(e["team"]["id"]) for e in js], "type": [e["type"]["name"] for e in js],
                       "sub": [(e["subtypes"] or {}).get("name", "") if not isinstance(e["subtypes"], list) else "" for e in js],
                       "start": [e["start"]["frame"] - 1 for e in js], "end": [e["end"]["frame"] - 1 for e in js],
                       "period": [e["period"] for e in js]})
    p2 = ev.loc[ev.period == 2, "start"].min()                   # period from the events
    period = np.where(np.arange(n) >= p2, 2, 1)
    return pl[:, :11], pl[:, 11:22], ball, period, ev            # first 11 slots home, next 11 away


def load_game(g):
    """Returns dict with P [T,S,2] metres, team [S], ball [T,2] metres, label [T] in {0:H, 1:A, 2:D, -1:unknown}, period.

    Labels: D where the open data has no ball position (it publishes the ball only while in play); otherwise the team
    of the most recent on-ball event. BALL LOST / BALL OUT do not change the team: the side that lost the ball keeps
    possession until the other side acts on it."""
    cp = os.path.join(ensure(CACHE), "ext_%d.npz" % g)
    if os.path.exists(cp):
        z = np.load(cp); return {k: z[k] for k in z.files}
    h, a, ball, period, ev = _game3() if g == 3 else _csv_game(g)
    P = np.concatenate([h, a], 1); P[..., 0] *= L; P[..., 1] *= W
    team = np.r_[np.zeros(h.shape[1], np.int8), np.ones(a.shape[1], np.int8)]
    ball = ball * np.array([L, W])
    n = len(P)
    lab = np.full(n, -1, np.int8)
    ev = ev[ev["type"].isin(ONBALL) & ev["team"].notna()].sort_values("start")
    cur = -1; last = 0
    for r in ev.itertuples():
        if r.type in ("BALL LOST", "BALL OUT"):
            continue
        s = int(r.start)
        if cur >= 0:
            lab[last:s] = cur
        cur, last = int(r.team), s
    lab[last:] = cur
    lab[~np.isfinite(ball[:, 0])] = 2
    out = dict(P=P, team=team, ball=ball, label=lab, period=period)
    np.savez_compressed(cp, **out)
    return out


if __name__ == "__main__":
    for g in (1, 2, 3):
        lab = load_game(g)["label"]; k = lab >= 0
        print("open match %d: %d frames, home / away / dead %s" % (g, len(lab), np.round(np.bincount(lab[k], minlength=3) / k.sum(), 3)))
