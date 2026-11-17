# mtah — possession labelling for More Than A Hack 2027

**Team jsp** · Pratyush Nayak, Sivalinga Raja Shanmugam, Junaid Ahmed

Frame-by-frame possession labels (`H` home, `A` away, `D` dead ball, `X` no answer) for the More Than A Hack 2027
Initial Challenge (FC Barcelona vs Athletic Club), from the automatic tracking only.

- **Method:** a deep players model (TSLANet-style temporal encoder per player + attention across players, no ball),
  a gradient-boosted ball-control model and a gradient-boosted change model, combined by a change-aware Viterbi decoder.
- **Training data:** only Metrica Sports' three open matches, with labels derived mechanically from their event data.
  No hand labels, and nothing trained on the provided match.
- **Results:** held-out spellF1 0.717 / macroF1 0.896 (leave-one-match-out on the open matches); 0.739 / 0.942 on the
  public five minutes (organisers' baseline: 0.326 / 0.621).

Full method, validation and negative results: [TECHNICAL.md](TECHNICAL.md) (PDF: [docs/technical_report.pdf](docs/technical_report.pdf)).

**No match data is in this repository.** The competition rules forbid redistributing it; `.gitignore` excludes it.

## Rebuild the submission

```bash
pip install -r requirements.txt                    # Python 3.11
git clone https://github.com/metrica-sports/sample-data
git -C sample-data checkout e706dd506b360d69d9d123d5b8026e7294b13996
export MTAH_DATA_DIR=/path/to/data                 # contains tactical/BAR-ATH_tactical_atd.{xml,txt}
export MTAH_EXT_DIR=$PWD/sample-data/data
./run_all.sh                                       # ~5 minutes -> out/submission.csv
```

The rebuild is exact: two rebuilds from scratch give byte-identical files. The deep model's weights are committed
(`weights/deep_players.pt`, trained on the open matches only) and applied on CPU, because GPU training is not
bit-for-bit repeatable. To retrain them (optional): `python -m mtah.deep.train`.

## Layout

| path | what it does |
|---|---|
| `mtah/config.py` | data paths (from environment variables) and match constants |
| `mtah/tracking.py` | parses the competition's tracking export; fills short ball gaps |
| `mtah/open_data.py` | loads Metrica's open matches and derives possession labels from their events |
| `mtah/features.py` | team shape and motion, reaction spread, movement order, ball-control features |
| `mtah/trees.py` | trains and applies the gradient-boosted players, ball-control and change models |
| `mtah/deep/model.py` | the deep players model (TSLANet-style blocks + attention across players) |
| `mtah/deep/data.py` | 4-second player windows for the deep model |
| `mtah/deep/train.py` | retrains the deep model (not needed to rebuild) |
| `mtah/deep/predict.py` | applies the committed weights to the competition match, on CPU |
| `mtah/decoder.py` | change-aware Viterbi decoder; writes the submission |
| `run_all.sh` | rebuilds everything from scratch, one step per process |
| `weights/deep_players.pt` | committed deep model weights |
| `docs/technical_report.pdf` | the technical report |
