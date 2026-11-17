#!/bin/sh
# Rebuild the submission from scratch: parse the tracking -> load the open matches -> train the tree models ->
# predict (tree models, and the deep players model from the committed weights, on CPU) -> decode.
# Each step runs in its own process. Writes out/submission.csv.
set -e
cd "$(dirname "$0")"
PY="${PYTHON:-python}"
rm -rf cache models
"$PY" -m mtah.tracking
"$PY" -m mtah.open_data
"$PY" -m mtah.trees train
"$PY" -m mtah.trees predict
"$PY" -m mtah.deep.predict
"$PY" -m mtah.decoder out/submission.csv
