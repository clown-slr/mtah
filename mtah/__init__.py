"""Possession labelling for More Than A Hack 2027 (team jsp).

Pipeline (each step is a module run with `python -m`, see run_all.sh):
    tracking    parse the competition's automatic tracking (players, ball)
    open_data   Metrica's open matches and possession labels derived from their events
    trees       gradient-boosted players / ball-control / change models (train, predict)
    deep        TSLANet-style player encoder with attention across players (committed weights)
    decoder     change-aware Viterbi decoder that writes the submission
"""
