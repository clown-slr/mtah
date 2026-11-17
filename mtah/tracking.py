"""Parse the competition's automatic tracking (Metrica ATD export) into numpy arrays, and fill short ball gaps.

    python -m mtah.tracking        # parse and cache
"""
import os
import xml.etree.ElementTree as ET
import numpy as np
from .config import DATA_DIR, CACHE, PITCH, ensure


def paths(source="tactical"):
    d = os.path.join(DATA_DIR, source)
    p = "BAR-ATH_%s" % source
    return {"meta": os.path.join(d, p + "_atd.xml"), "raw": os.path.join(d, p + "_atd.txt")}


def slot_teams(meta_path):
    """Team of each tracking slot: 0 for the first team in the metadata (home), 1 for the other."""
    root = ET.parse(meta_path).getroot()
    team_ids = [t.get("id") for t in root.iter("Team")]
    return np.array([0 if p.get("teamId") == team_ids[0] else 1 for p in root.iter("Player")], dtype=np.int8)


def load(source="tactical", force=False):
    """Returns dict: P [T,S,3] (x, y in metres, speed), B [T,3] (ball x, y in metres, third field), team [S]."""
    cp = os.path.join(ensure(CACHE), "atd_%s.npz" % source)
    if os.path.exists(cp) and not force:
        z = np.load(cp)
        return {k: z[k] for k in z.files}
    p = paths(source)
    team = slot_teams(p["meta"])
    S = len(team)
    rows_p, rows_b, fids = [], [], []
    with open(p["raw"]) as fh:
        for line in fh:
            parts = line.rstrip("\n").split(":")
            if len(parts) != 3:
                continue
            fids.append(int(parts[0]))
            g = parts[1].split(";")
            rows_p.append(np.array([[float(v) for v in x.split(",")] for x in g], dtype=np.float32))
            rows_b.append(np.array([float(v) for v in parts[2].split(",")], dtype=np.float32))
    fids = np.array(fids)
    T = fids.max()
    P = np.full((T, S, 3), np.nan, np.float32)
    B = np.full((T, rows_b[0].shape[0]), np.nan, np.float32)
    for f, rp, rb in zip(fids, rows_p, rows_b):
        P[f - 1, :rp.shape[0]] = rp
        B[f - 1] = rb
    P[..., 0] *= PITCH[0]; P[..., 1] *= PITCH[1]
    B[:, 0] *= PITCH[0]; B[:, 1] *= PITCH[1]
    out = {"P": P, "B": B, "team": team}
    np.savez_compressed(cp, **out)
    return out


def interp_ball(B, max_gap=50, max_speed=36.0):
    """Linear interpolation of the ball across gaps of at most max_gap frames, unless the ball would have had to
    travel faster than max_speed m/s. Returns xy [T,2], observed mask, filled mask."""
    xy = B[:, :2].astype(np.float64).copy()
    obs = np.isfinite(xy[:, 0])
    idx = np.flatnonzero(obs)
    filled = obs.copy()
    for a, b in zip(idx[:-1], idx[1:]):
        g = b - a
        if 1 < g <= max_gap:
            dist = np.hypot(*(xy[b] - xy[a]))
            if dist / (g / 25.0) <= max_speed:
                t = (np.arange(a + 1, b) - a)[:, None] / g
                xy[a + 1:b] = xy[a] + t * (xy[b] - xy[a])
                filled[a + 1:b] = True
    return xy, obs, filled


if __name__ == "__main__":
    d = load(force=True)
    print("tracking:", d["P"].shape, "ball seen on %.1f%% of frames" % (100 * np.isfinite(d["B"][:, 0]).mean()),
          "| slots per team", np.bincount(d["team"]))
