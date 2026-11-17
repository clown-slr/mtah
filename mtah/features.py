"""Features shared by the open training matches and the competition match.

Everything is computed from positions, after orienting each half so the home team attacks towards +x. No player
identities are used: every feature is a statistic over a team.
"""
import warnings
import numpy as np, pandas as pd

warnings.simplefilter("ignore")
LAGS = (-100, -50, -25, -12, 12, 25, 50, 100)      # context offsets in frames: +-0.5, 1, 2, 4 s


def velocities(P):
    v = np.full_like(P, np.nan)
    v[2:-2] = (P[4:] - P[:-4]) * (25.0 / 4.0)
    sp = np.hypot(v[..., 0], v[..., 1])
    v[sp > 12.0] = np.nan  # identity swaps and tracker jumps
    return v


def orient(P, team, ball, blocks):
    """Flip x (and y) per block so that the home team attacks +x. blocks: list of (lo, hi) inclusive."""
    P = P.copy(); ball = None if ball is None else ball.copy()
    Lx = 105.0
    for lo, hi in blocks:
        s = slice(lo, hi + 1)
        hx = np.nanmean(P[s][:, team == 0, 0]); ax = np.nanmean(P[s][:, team == 1, 0])
        if hx > ax:
            P[s, :, 0] = Lx - P[s, :, 0]; P[s, :, 1] = 68.0 - P[s, :, 1]
            if ball is not None:
                ball[s, 0] = Lx - ball[s, 0]; ball[s, 1] = 68.0 - ball[s, 1]
    return P, ball


def swap_teams(P, team, ball):
    """Mirror a match: exchange the teams and rotate the pitch, so the models see both roles."""
    P2 = P.copy(); P2[..., 0] = 105.0 - P2[..., 0]; P2[..., 1] = 68.0 - P2[..., 1]
    b2 = ball.copy(); b2[:, 0] = 105.0 - b2[:, 0]; b2[:, 1] = 68.0 - b2[:, 1]
    return P2, 1 - team, b2


def _team_stats(P, V, m, name):
    X = np.where(m[None, :, None], P, np.nan); Vv = np.where(m[None, :, None], V, np.nan)
    sp = np.hypot(Vv[..., 0], Vv[..., 1])
    xs = np.sort(X[..., 0], 1)  # NaN last
    n = np.isfinite(X[..., 0]).sum(1)
    f = {}
    f[name + "_cx"] = np.nanmean(X[..., 0], 1); f[name + "_cy"] = np.nanmean(X[..., 1], 1)
    f[name + "_sx"] = np.nanstd(X[..., 0], 1); f[name + "_sy"] = np.nanstd(X[..., 1], 1)
    f[name + "_x2"] = xs[:, 1]  # second deepest (first is usually the keeper for the team attacking +x)
    idx = np.clip(n - 2, 0, xs.shape[1] - 1)
    f[name + "_xm2"] = xs[np.arange(len(xs)), idx]
    f[name + "_wid"] = np.nanmax(X[..., 1], 1) - np.nanmin(X[..., 1], 1)
    f[name + "_sp"] = np.nanmean(sp, 1); f[name + "_spmax"] = np.nanmax(np.where(np.isfinite(sp), sp, -1), 1)
    f[name + "_vx"] = np.nanmean(Vv[..., 0], 1); f[name + "_vy"] = np.nanmean(Vv[..., 1], 1)
    f[name + "_fast"] = np.nanmean(sp > 4.0, 1); f[name + "_slow"] = np.nanmean(sp < 1.0, 1)
    f[name + "_n"] = n
    return f


def reaction_features(V, team, k=6):
    """Reaction spread: the leading edge of a change (first reactors), how coherent it is, and which team moved first."""
    sp = np.hypot(V[..., 0], V[..., 1])
    T, S = sp.shape
    acc = np.full((T, S), np.nan); acc[k:-k] = (sp[2 * k:] - sp[:-2 * k]) * (25.0 / (2 * k))      # m/s^2 over ~0.5 s
    srt = np.sort(acc, 1)                                   # NaN last
    n = np.isfinite(acc).sum(1)
    f = {}
    f["acc_lo"] = srt[:, 1]                                 # second strongest deceleration: the first reactors
    f["acc_hi"] = srt[np.arange(T), np.clip(n - 2, 0, S - 1)]
    f["acc_sd"] = np.nanstd(acc, 1)
    f["dec_share"] = np.nanmean(acc < -1.5, 1); f["acc_share"] = np.nanmean(acc > 1.5, 1)
    ang = np.arctan2(V[..., 1], V[..., 0])                  # turning of moving players over the same window
    da = np.full((T, S), np.nan); da[k:-k] = np.abs(np.angle(np.exp(1j * (ang[2 * k:] - ang[:-2 * k]))))
    mov = sp > 1.5
    f["turn"] = np.nanmean(np.where(mov, da, np.nan), 1)
    f["turn_hi"] = np.nanmax(np.where(mov & np.isfinite(da), da, -1), 1)
    ux = np.where(mov, V[..., 0] / np.maximum(sp, 1e-6), np.nan); uy = np.where(mov, V[..., 1] / np.maximum(sp, 1e-6), np.nan)
    f["align"] = np.hypot(np.nanmean(ux, 1), np.nanmean(uy, 1))   # order parameter: do players move one way
    for nm, m in (("h", team == 0), ("a", team == 1)):
        f[nm + "_align"] = np.hypot(np.nanmean(ux[:, m], 1), np.nanmean(uy[:, m], 1))
        f[nm + "_acc"] = np.nanmean(acc[:, m], 1)
        f[nm + "_dir"] = np.nanmean(ux[:, m], 1)
    f["cross"] = np.nanmean(ux[:, team == 0], 1) * np.nanmean(ux[:, team == 1], 1) + np.nanmean(uy[:, team == 0], 1) * np.nanmean(uy[:, team == 1], 1)
    hs = np.nanmean(np.where((team == 0)[None], sp, np.nan), 1); as_ = np.nanmean(np.where((team == 1)[None], sp, np.nan), 1)
    dh = pd.Series(hs).diff(12).values; da_ = pd.Series(as_).diff(12).values
    f["lead"] = pd.Series(dh).shift(12).values * da_ - pd.Series(da_).shift(12).values * dh      # >0: home changed first
    return f


def _rmean(x, w):
    return pd.DataFrame(x).rolling(w, center=True, min_periods=w // 3).mean().values


def delay_features(P, V, team, lags=(6, 12, 18), w=75, hp=62):
    """Movement order: does each player's velocity change follow its own team's or the other team's (shifted
    0.24 / 0.48 / 0.72 s earlier vs later)? The team without the ball usually follows."""
    T, S = V.shape[:2]
    Vh = V - np.stack([_rmean(V[..., c], 2 * hp + 1) for c in (0, 1)], -1)          # keep only changes
    ref = {g: np.stack([np.nanmean(Vh[:, team == g, c], 1) for c in (0, 1)], -1) for g in (0, 1)}
    en = np.sqrt(_rmean(np.nansum(Vh ** 2, -1) + 0 * Vh[..., 0], w))                   # NaN where player missing

    def follow(R):
        """>0: player follows R (matches R's past better than R's future)."""
        R = np.nan_to_num(R); rn = np.sqrt(_rmean((R ** 2).sum(-1)[:, None], w))
        out = 0
        for L in lags:
            past = np.roll(R, L, 0); fut = np.roll(R, -L, 0)
            d = Vh[..., 0] * (past[:, None, 0] - fut[:, None, 0]) + Vh[..., 1] * (past[:, None, 1] - fut[:, None, 1])
            out = out + _rmean(d, w)
        return out / (len(lags) * np.maximum(en * rn, 1e-3))

    own = np.where((team == 0)[None], follow(ref[0]), follow(ref[1]))
    opp = np.where((team == 0)[None], follow(ref[1]), follow(ref[0]))
    f = {}
    for nm, g in (("h", 0), ("a", 1)):
        m = team == g
        f[nm + "_fol"] = np.nanmean(opp[:, m], 1)                 # this team follows the other
        f[nm + "_fol_hi"] = np.nanmax(np.where(np.isfinite(opp[:, m]), opp[:, m], -9), 1)
        f[nm + "_dl_sd"] = np.nanstd(own[:, m], 1)                # how staggered the team's own reaction is
        x = P[:, m, 0]; d = own[:, m]; ok = np.isfinite(x) & np.isfinite(d)
        xc = np.where(ok, x, np.nan); xc = xc - np.nanmean(xc, 1, keepdims=True); dc = np.where(ok, d, np.nan)
        f[nm + "_dl_x"] = np.nansum(xc * dc, 1) / np.maximum(np.nansum(xc ** 2, 1), 1.0)   # front or back first
    f["fol"] = f["a_fol"] - f["h_fol"]                           # >0: away follows home
    return f


def roll(x, k):
    return pd.Series(x).rolling(2 * k + 1, center=True, min_periods=1).mean().values


def add_context(f, keys):
    out = dict(f)
    for k in keys:
        x = pd.Series(f[k])
        for lag in LAGS:
            out["%s_l%d" % (k, lag)] = x.shift(-lag).values
        out[k + "_r25"] = roll(f[k], 25); out[k + "_r75"] = roll(f[k], 75)
    return out


def player_features(P, team, V=None):
    """Players-only features (no ball): team shape and motion, reaction spread, movement order, with context."""
    V = velocities(P) if V is None else V
    f = {}
    f.update(_team_stats(P, V, team == 0, "h")); f.update(_team_stats(P, V, team == 1, "a"))
    sp = np.hypot(V[..., 0], V[..., 1])
    f["sp"] = np.nanmean(sp, 1); f["sp_sd"] = np.nanstd(sp, 1)
    f["d_cx"] = f["h_cx"] - f["a_cx"]; f["d_sx"] = f["h_sx"] - f["a_sx"]; f["d_sy"] = f["h_sy"] - f["a_sy"]
    f["d_wid"] = f["h_wid"] - f["a_wid"]; f["d_sp"] = f["h_sp"] - f["a_sp"]; f["vx"] = (f["h_vx"] + f["a_vx"]) / 2
    f["line_gap"] = f["a_xm2"] - f["h_x2"]
    f.update(reaction_features(V, team))
    f.update(delay_features(P, V, team))
    ctx = ["sp", "h_sp", "a_sp", "vx", "d_cx", "d_sx", "d_wid", "h_cx", "a_cx", "h_fast", "a_fast", "h_slow", "a_slow",
           "line_gap", "acc_lo", "acc_hi", "turn", "align", "lead", "fol", "h_fol", "a_fol"]
    return pd.DataFrame(add_context(f, ctx)).astype(np.float32)


def ball_features(P, team, ball, V=None):
    """Ball-control features: only meaningful where a ball position exists (NaN elsewhere)."""
    V = velocities(P) if V is None else V
    d = np.hypot(P[..., 0] - ball[:, None, 0], P[..., 1] - ball[:, None, 1])
    d = np.where(np.isfinite(d), d, np.inf)
    dh = np.sort(np.where(team[None] == 0, d, np.inf), 1)[:, :2]; da = np.sort(np.where(team[None] == 1, d, np.inf), 1)[:, :2]
    dh = np.minimum(dh, 60.0); da = np.minimum(da, 60.0)
    vb = np.full(len(ball), np.nan); vb[1:-1] = np.hypot(*(ball[2:] - ball[:-2]).T) * 12.5
    vbx = np.full(len(ball), np.nan); vbx[1:-1] = (ball[2:, 0] - ball[:-2, 0]) * 12.5
    f = {"dh1": dh[:, 0], "da1": da[:, 0], "dh2": dh[:, 1], "da2": da[:, 1], "dd": dh[:, 0] - da[:, 0],
         "vb": vb, "vbx": vbx, "bx": ball[:, 0], "by": ball[:, 1],
         "nh5": (np.where(team[None] == 0, d, np.inf) < 5).sum(1), "na5": (np.where(team[None] == 1, d, np.inf) < 5).sum(1)}
    nan = ~np.isfinite(ball[:, 0])
    for k in f:
        f[k] = np.where(nan, np.nan, f[k]).astype(np.float64)
    f = add_context(f, ["dd", "dh1", "da1", "vb", "vbx", "bx"])
    return pd.DataFrame(f).astype(np.float32)
