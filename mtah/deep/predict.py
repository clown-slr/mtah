"""Apply the committed deep players model to the competition match, on CPU (exactly repeatable).

    python -m mtah.deep.predict    # writes cache/deep_p1.npz
"""
import os
import numpy as np, torch
from ..config import CACHE, HALVES, WEIGHTS, ensure
from ..tracking import load
from ..features import orient
from .data import frame_tensor
from .model import DeepGTW
from .train import predict


def run(force=False):
    cp = os.path.join(ensure(CACHE), "deep_p1.npz")
    if os.path.exists(cp) and not force:
        return np.load(cp)["p1"]
    torch.set_num_threads(4)
    d = load(); P = d["P"][..., :2].astype(np.float64); team = d["team"]
    Po, _ = orient(P, team, None, HALVES)
    G = dict(F=frame_tensor(Po, team), team=team, lab=np.full(len(P), -1, np.int64))
    net = DeepGTW(); net.load_state_dict(torch.load(WEIGHTS, map_location="cpu"))
    p1 = predict(net, G)
    np.savez_compressed(cp, p1=p1)
    return p1


if __name__ == "__main__":
    p = run(force=True)
    print("deep players model: mean H/A/D", p[np.concatenate([np.arange(a, b + 1) for a, b in HALVES])].mean(0).round(3))
