"""Train the deep players model on the three open matches (each also mirrored).

Not needed to rebuild the submission: the weights used for it are committed in weights/deep_players.pt. GPU training
is not bit-for-bit repeatable, so re-running this gives a model of the same kind but not identical weights.

    python -m mtah.deep.train                  # all three matches -> weights/deep_players.pt
    python -m mtah.deep.train --holdout 2      # leave match 2 out (held-out evaluation)
"""
import os, sys, time
import numpy as np, torch, torch.nn.functional as Fn
from .data import open_game, batch, W, SUB
from .model import DeepGTW
from ..config import ROOT, ensure

DEV = os.environ.get("DEVICE") or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
EPOCHS, PER_EPOCH, BS, LR = 6, 40000, 128, 2e-3


def train(games, seed=0):
    """Cross-entropy with label smoothing 0.05, AdamW (weight decay 0.05), one-cycle learning rate peaking at 2e-3,
    6 epochs of 40,000 windows, batch 128, 8% of players dropped per window."""
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    net = DeepGTW().to(DEV)
    steps = EPOCHS * (PER_EPOCH // BS)
    opt = torch.optim.AdamW(net.parameters(), LR, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, LR, total_steps=steps, pct_start=0.1)
    valid = [np.flatnonzero(G["lab"] >= 0) for G in games]
    t0 = time.time(); run = 0.0
    for s in range(steps):
        gi = rng.integers(len(games)); G = games[gi]
        X, m, y = batch(G, rng.choice(valid[gi], BS), rng, drop=0.08)
        out = net(torch.tensor(X, device=DEV), torch.tensor(m, device=DEV))
        loss = Fn.cross_entropy(out.reshape(-1, 3), torch.tensor(y, device=DEV).reshape(-1), ignore_index=-1, label_smoothing=0.05)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sched.step()
        run = 0.98 * run + 0.02 * loss.item() if s else loss.item()
        if (s + 1) % (PER_EPOCH // BS) == 0:
            print("  epoch %d loss %.4f  %.0fs" % ((s + 1) // (PER_EPOCH // BS), run, time.time() - t0), flush=True)
    return net


@torch.no_grad()
def predict(net, G, stride=10):
    """Per-frame H/A/D probabilities: overlapping windows every `stride` frames, averaged per frame."""
    net.eval()
    dev = next(net.parameters()).device
    T = len(G["lab"]); acc = np.zeros((T, 3)); cnt = np.zeros(T)
    centres = np.arange(0, T, stride)
    for i in range(0, len(centres), 256):
        c = centres[i:i + 256]
        X, m, _ = batch(G, c)
        p = torch.softmax(net(torch.tensor(X, device=dev), torch.tensor(m, device=dev)), -1).cpu().numpy()
        idx = np.clip(c[:, None] + np.arange(-W // 2, W // 2, SUB)[None], 0, T - 1)
        np.add.at(acc, idx.ravel(), p.reshape(-1, 3)); np.add.at(cnt, idx.ravel(), 1)
    have = cnt > 0
    for k in range(3):
        acc[:, k] = np.interp(np.arange(T), np.flatnonzero(have), acc[have, k] / cnt[have])
    return acc / acc.sum(1, keepdims=True)


if __name__ == "__main__":
    hold = int(sys.argv[sys.argv.index("--holdout") + 1]) if "--holdout" in sys.argv else None
    games = [open_game(g, s) for g in (1, 2, 3) if g != hold for s in (False, True)]
    net = train(games, seed=hold or 0)
    out = os.path.join(ensure(os.path.join(ROOT, "weights")), "deep_players%s.pt" % ("" if hold is None else "_holdout%d" % hold))
    torch.save(net.state_dict(), out); print("saved", out)
