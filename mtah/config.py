"""Paths and match constants. Data locations come from environment variables; nothing is downloaded."""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# competition data (not in the repository): <MTAH_DATA_DIR>/tactical/BAR-ATH_tactical_atd.{xml,txt}
DATA_DIR = os.environ.get("MTAH_DATA_DIR", os.path.join(ROOT, "data"))
# Metrica open sample data: <MTAH_EXT_DIR>/Sample_Game_{1,2,3}/...
EXT_DIR = os.environ.get("MTAH_EXT_DIR", os.path.join(ROOT, "sample-data", "data"))

CACHE = os.path.join(ROOT, "cache")       # parsed tracking and per-frame probabilities
MODELS = os.path.join(ROOT, "models")     # trained tree models
OUT = os.path.join(ROOT, "out")           # the submission file
WEIGHTS = os.path.join(ROOT, "weights", "deep_players.pt")   # committed deep model

N_FRAMES = 199319                                 # frames of the tactical video
PITCH = (105.3, 68.0)                             # metres; normalised coordinates are scaled by this
HALVES = [(11185, 90651), (110808, 190197)]       # graded frames (inclusive); everything else is X


def ensure(path):
    os.makedirs(path, exist_ok=True)
    return path
