# Possession labelling for BAR-ATH — technical document

**Team jsp** · Authors: Pratyush Nayak, Sivalinga Raja Shanmugam, Junaid Ahmed

More Than A Hack 2027, Initial Challenge. Output: one label (`H`, `A`, `D`, `X`) per frame of the
tactical video.

## Approach

Four parts, all algorithmic:

1. **Players-only model.** A small neural network (see "Deep players model" below) predicts `H` /
   `A` / `D` for every frame from the raw positions and velocities of the players alone: a
   TSLANet-style encoder reads each player's movement over 4 s, and attention across players lets
   each player's representation depend on who it is reacting to. It never sees the ball. It replaced
   a gradient-boosted classifier on hand-made team features (centroids, spread, lines, speeds, and
   the reaction and movement-order features below), which is still trained because the change model
   uses its features.
2. **Ball-control model.** A second gradient-boosted classifier predicts which team controls the
   ball on frames where a ball position exists: distance of the two nearest players of each team,
   their difference, ball speed, ball position, players within 5 m, again with temporal context.
3. **Change model.** A third gradient-boosted classifier, on both feature sets, predicts for every
   frame whether a home↔away change, or a live↔dead change, happens within 0.48 s.
4. **Decoder.** A three-state Viterbi pass per half combines the first two models' log-probabilities
   with two ball rules (below) and a cost for changing state. The cost is what turns per-frame
   evidence into possessions: a brief touch by the other team does not pay for a switch, sustained
   control does. The cost varies per frame with the change model: lower where it expects a change,
   higher where it does not (see "Change-aware switch cost").

Why this rather than a rule on the ball alone: the ball is missing on 41% of graded frames, and
those are the frames a ball-only method cannot label. Player geometry is available on every frame
and turned out to carry most of the answer. Why two models rather than one: in the training
matches the ball is absent exactly when it is dead, while the automatic tracking of the provided
match often keeps tracking a dead ball. A single model given the ball would learn "ball visible
means live", which is false here. Keeping the ball out of the dead/live model avoids that.

## Reaction and movement order

A possession change or a whistle does not reach every player at once: the players nearest the ball
react first and the reaction spreads outward, and defenders move in response to attackers. Team
averages hide both. Two groups of features describe them (`mtah/features.py`). This is the idea behind
graphical time warping (per-player time offsets, smooth across neighbouring players) without the
solver.

- **Reaction spread** (`reaction_features`): each player's change in speed over about 0.5 s; the
  second-strongest deceleration and acceleration (the first reactors), the spread of those changes,
  the share of players braking or accelerating hard, how much moving players turn, how aligned
  their directions are, per team and between teams, and which team's mean speed changed first.
- **Movement order** (`delay_features`): each player's velocity, with the slow trend removed
  (2.5 s mean), is compared with the average velocity change of their own team and of the
  opposing team shifted 0.24, 0.48 and 0.72 s earlier and later. If the player matches the
  reference's past better than its future, the player follows it. Summarised over a 3 s window:
  how strongly each team follows the other (mean and strongest player), how staggered each team's
  own reaction is, and whether it starts at the front or the back of the team. The difference
  between the teams' following scores (`fol`) gets the same ±0.5–4 s context as the other key
  features. The team without the ball usually follows.

Both are statistics over a team; no player identity is used. Held-out effect on the open matches,
at the chosen decoder settings:

| features | spellF1 mean | spellF1 worst match | macroF1 | dead/live accuracy per match |
|---|---|---|---|---|
| base | 0.652 | — | 0.874 | 0.879 / 0.912 / 0.923 |
| + reaction spread | 0.657 | 0.641 | 0.882 | 0.889 / 0.923 / 0.925 |
| + movement order (submitted) | 0.662 | 0.653 | 0.882 | 0.889 / 0.921 / 0.925 |

Each step is inside the 0.04 noise band on its own; they were kept because every check moved the
same way, on all three held-out matches and on the public window.

## Assets used

| asset | use |
|---|---|
| tactical ATD (players, ball) | the only input to the models |
| tactical homography, video | debugging overlays and the two experiments below (ball from video, shirt-colour audit); not an input to the submission |
| Smart Tagging | tried as a per-frame team prior in a first rule-based version, **dropped**: it is too coarse to place transitions and the players-only model already carries the same information |
| broadcast video, ATD, homography | **not used**: not synchronised to the tactical timeline |
| per-slot speed field of the ATD | **dropped**: speeds are recomputed from positions so the features match the training matches; jumps above 12 m/s (identity swaps) are discarded |

No player identities are used anywhere; every feature is a statistic over a team.

## Training data

Both models are trained only on **Metrica Sports' open sample data** (github.com/metrica-sports/sample-data):
three anonymised matches with tracking at 25 fps and event data. None of it is the provided game,
and no label of any kind was produced by us by hand, on any game.

Labels are derived mechanically from that dataset (`mtah/open_data.py`):

- `D` where the dataset has no ball position (it publishes the ball only while in play);
- otherwise the team of the most recent on-ball event (pass, carry, recovery, shot, set piece).
  A `BALL LOST` or `BALL OUT` event does not change the team: the side that lost the ball keeps
  possession until the other side acts on it. Challenges, fouls and cards are ignored.

The event annotations are Metrica's. Size: about 430,000 frames across the three matches; every
third frame is used, and each match is also used mirrored (teams exchanged, pitch rotated), giving
286,618 training frames. To mimic the automatic tracking, players are randomly removed for
stretches of 1–7 s during training.

Each half is flipped so the home team attacks towards +x, in both the training matches and the
provided match (direction inferred from the teams' mean positions).

## Possession rules implemented

- **Loose and contested balls** stay with the last team: with no clear evidence either way the
  decoder has no reason to pay the switch cost.
- **Change of possession** needs sustained evidence. The switch cost (6 log-units against
  per-frame evidence weighted 0.25–0.5) means roughly half a second of clear control by the other
  team; a deflection or a single touch is not enough. Viterbi places the change where the evidence
  flips, not half a second later.
- **Dead ball** comes from the players-only model (collective slowing, shape), plus two rules:
  a tracked ball more than 0.5 m outside the lines is evidence for `D`; and whether the tracker
  sees the ball at all is evidence, with a likelihood ratio estimated on the provided match from
  the players-only model's own soft labels (ball lost on about 70% of frames it calls dead, 17%
  of frames it calls live). No ground truth enters that estimate.
- **Restarts**: a possession ends at a stoppage and a new one starts after it, even for the same
  team. Entering or leaving `D` costs 10.
- **Outside the two halves** every frame is `X`. There are no abstentions inside the halves.

## Change-aware switch cost

With a flat cost, the decoder knows how much evidence a change needs but not where changes tend to
happen, so it often merges two possessions of the same team across a short stoppage, or places the
change a little early or late. The change model (`mtah/trees.py`, labels from `change_labels`) is trained
on the open matches: a frame is "near a home↔away change" or "near a live↔dead change" if one occurs
within 12 frames. In the decoder (`mtah/decoder.py`, `switch_costs`):

    cost_switch[t] = 6  - 1 · log(p_team_change[t] / prior)     clipped to [0.5, 18]
    cost_dead[t]   = 10 - 2 · log(p_dead_change[t] / prior)     clipped to [0.5, 30]

On held-out open matches the change model reaches AUC 0.88–0.93 for home↔away and 0.91–0.95 for
live↔dead changes. The two weights (1 and 2) were chosen from a 5 × 5 grid on the same
leave-one-match-out protocol as the other settings (`cp_exp.py`):

| decoder | spellF1 mean | per held-out match | macroF1 | possessions at IoU 0.5: precision / recall |
|---|---|---|---|---|
| flat cost | 0.662 | 0.653 / 0.676 / 0.656 | 0.882 | 0.814 / 0.570 |
| change-aware (submitted) | 0.691 | 0.681 / 0.711 / 0.682 | 0.891 | 0.839 / 0.597 |

All three matches improve, and most of the grid beats the flat cost, so the gain does not depend on
one setting; because the weights were picked on these matches the 0.03 is somewhat optimistic. The
number of predicted possessions barely changes (752 → 764 over the three matches): the gain comes
from placing changes better, mostly fewer true possessions lost to dead ball. With both weights at 0
the pipeline reproduces the previous file exactly.

## Deep players model

`mtah/deep/model.py`, `mtah/deep/data.py`, `mtah/deep/train.py`, `mtah/deep/predict.py`; about 98,000 parameters.

- **Input.** A 4 s window (every 2nd frame, 50 steps) of up to 12 players per team: position,
  velocity, speed, presence and team, after orienting home to attack +x. No identities, no ball.
- **Over time, per player:** two blocks after TSLANet (Eldele et al., ICML 2024): an adaptive
  spectral block (FFT along time, learned complex filters, and a learned energy threshold that treats
  high-frequency content separately, which suits tracker jitter) and an interactive convolution
  block (two convolution branches that gate each other).
- **Across players, per time step:** multi-head attention over the players with a team embedding.
  This is a learned form of the graphical-time-warping idea: rather than estimating explicit
  per-player time offsets, each player's representation attends to the players it reacts to.
- **Output:** `H` / `A` / `D` for every one of the 50 steps, pooled from home, away and all players.
  Predictions from overlapping windows (every 10 frames) are averaged per frame.
- **Training.** Only the three open matches, each also mirrored with teams exchanged; every frame can
  centre a window and every step of the window is supervised, so the ~1,000 possessions give
  hundreds of thousands of labelled frames. Cross-entropy with label smoothing 0.05, AdamW (weight
  decay 0.05), one-cycle learning rate peaking at 2e-3, 6 epochs of 40,000 windows, batch 128,
  8% of players randomly dropped per window to mimic tracker losses. About 8 minutes on an Apple
  M-series GPU.

Held-out (leave-one-match-out, each fold trained on the other two matches), through the same
change-aware decoder:

| players model | spellF1 mean | per held-out match | macroF1 | possessions at IoU 0.5: precision / recall | held-out log-loss per match |
|---|---|---|---|---|---|
| gradient-boosted trees | 0.691 | 0.681 / 0.711 / 0.682 | 0.891 | 0.840 / 0.599 | 0.682 / 0.338 / 0.338 |
| deep model (submitted) | 0.717 | 0.708 / 0.750 / 0.692 | 0.896 | 0.845 / 0.642 | 0.491 / 0.325 / 0.320 |

All three matches improve, mostly through recall. This was the first configuration trained, with no
search over the model or the decoder, so the gain is not inflated by selection. Three loss variants
were tried afterwards and dropped: adding a soft-F1 term per class (spellF1 0.701), weighting
frames near possession changes three times (0.718, mixed across matches), and both together
(0.713).

**Reproducibility.** Training on the GPU is not bit-for-bit repeatable, so the weights used for the
submission are committed (`weights/deep_players.pt`, 411 KB, trained with `python -m mtah.deep.train` on the
three open matches only; they contain no data of the provided match). `run_all.sh` applies them on
CPU, which is exactly repeatable: two rebuilds from scratch give byte-identical files, and CPU and
GPU inference give the same label on every graded frame. Re-running `python -m mtah.deep.train` gives a model of
the same kind but not identical weights.

## Choosing the decoder settings

*Scripts named in this section and in the experiment sections below (`tune_ext.py`, `cp_exp.py`, and the analysis and experiment scripts) belong to the development workspace; this repository contains only the final method.*

Weights and costs were chosen by leave-one-match-out on the three open matches, with the ball
masked on about 13% of frames, scored with the organisers' `possession_metrics.py`
(`tune_ext.py`). The surface is flat: spellF1 between 0.62 and 0.66 across the grid. The chosen
setting (players 0.25, control 0.5, switch 6, dead 10) scored spellF1 0.652 and macroF1 0.874
there with the base features, and 0.662 and 0.882 with the submitted features. The settings were
not re-tuned after adding features.

The ball-visibility weight cannot be chosen on the open matches, where a missing ball means a dead
ball by construction. It is set to 0.25, the same weight as the players-only model: both are
per-frame evidence from strongly correlated neighbouring frames, so neither should count more per
frame than the other. In the interest of full disclosure: early in development three values (0,
0.25, 0.5) were also scored on the public five minutes; the differences (spellF1 0.66–0.69) are far
inside the noise of 17 possessions, and the value is the same.

How the public five minutes were used, all of it here: to check that the pipeline was aligned; to
see how dead balls look in the automatic tracking; as a second check next to the held-out matches
when deciding whether to keep each idea. Every idea that was kept had first improved the held-out
open matches. Two ideas were dropped partly because of the public window: the ball-moving-with-player
features (held-out results mixed, public window lower) and the removal of extra tracks (which the
clean held-out tracking cannot measure). Nothing was trained on it, and no setting of the submitted
pipeline was chosen by its score. On that window the submission scores spellF1 0.739 and macroF1
0.942 (tree players model: 0.694 and 0.930; flat-cost decoder: 0.703 and 0.930; base features:
0.663 and 0.914; `baseline_smooth50`: 0.326 and 0.621). We do not treat it as an estimate of the
full-match score.

Because over a dozen ideas were evaluated on the same three held-out matches, the held-out numbers
are somewhat optimistic. Two gains are clearly beyond the noise: the change-aware decoder (+0.03,
all three matches up) and the deep players model (+0.026, all three matches up, first configuration
tried). The reaction and movement-order features (+0.005 each) were kept because every check moved
the same way.

## Tried and dropped: robot-motion kinematics

We tested features from robot navigation among people: a Kalman filter and smoother on the ball
with outlier rejection, collision detection (a jump in ball velocity next to a player, as a touch),
and time to intercept (which team can reach the ball first on its predicted path). For a fair test
the open-match ball was first degraded to match the provided tracking's statistics (tracker
smoothing, registration drift, dropouts).

- On the clean open-match ball the touch detector finds 88% of pass and shot starts with 97% of
  touches next to an annotated event. The provided ball track is far smoother (99th percentile of
  the 3-frame velocity jump 4.4 m/s against 10.0 m/s in the open data), so sharp touches barely
  exist in it.
- Leave-one-match-out on the degraded ball: ball-control accuracy rose from 0.925 / 0.939 / 0.928
  to 0.931 / 0.944 / 0.934, and best spellF1 from 0.620 to 0.624. That is well inside the 0.04
  noise band, so the features are **not** in the submission.

The code for this experiment is kept outside the submission pipeline.

## Tried and dropped: ball from the video

The tracker loses the ball on 41% of graded frames, so we built a ball detector on the tactical
video. Its training labels are algorithmic: the tracker's ball projected through the homography and
snapped to the nearest white blob (difference of Gaussians on whiteness). A small fully
convolutional network scores candidate blobs on every frame (PyTorch, Apple GPU), and a Viterbi
pass links them into a track, removing static pitch markings.

- Coverage of graded frames rose from 59.3% (tracker) to 73.7%; 53.1% of the frames the tracker
  lost now have a ball. Median distance to the tracker where both exist: 0.83 m.
- In six sampled frames where the two disagree by more than 10 m the detector was on the ball in
  all six; where the tracker has no ball, about half of the sampled detections were the ball and
  half were boots or objects near the touchline.
- Touch detection now fires at a realistic rate (25.7 per minute, against 21–28 in the open
  data). But on the public window, filling gaps scored spellF1 0.663 / macroF1 0.901, replacing the
  tracker 0.686 / 0.884, and adding touch features 0.580. None beats the submission, so the video
  ball is **not** an input.

## Tried and dropped: shirt-colour team check

Shirt colour was sampled under every tracked player in 398 frames (7,674 samples, grass pixels
removed, median colour in Lab space) and compared with the tracking file's team tags. Two-means
clustering on colour alone agrees with the tags on 94.8% of samples; every outfield slot agrees on
88–99%, the misses being occlusion and overlapping players. The one slot that disagrees is the
home goalkeeper, whose shirt differs from his team's by design. The team tags are therefore
correct, and a colour feature would duplicate them, so it is not used.

## Tried and dropped: ball moving with the player

To test control rather than nearness, the ball-control model was given, for the nearest player of
each team: the ball's velocity relative to that player, how long the same player has stayed within
2 m of the ball, and whether the ball is within 2 m and moving with them (`control_features`,
`MTAH_CTRL=1`). On the held-out open matches, spellF1 at the chosen settings went from 0.662 to
0.664, but ball-control accuracy fell on one match (0.948 → 0.931) while rising on the other two.
On the provided match the control model's team accuracy on the public window rose from 0.909 to
0.922, yet the decoded labels scored lower (spellF1 0.703 → 0.640, macroF1 0.930 → 0.923, with
fewer possessions found). The provided ball track is heavily smoothed, so relative velocity is less
informative on it than on the training matches. The features are off in the submission.

## Tried and dropped: restart shapes, stoppage length, cleaned labels

A breakdown of missed possessions on the held-out matches (`misses_by_situation.py`) shows that two
thirds are possessions with no completed pass (median 1.8 s: an interception or clearance lost at
once), and that most of the rest are found but merged with a neighbour or mistimed around stoppages.
Four ideas aimed at the second group, all tested with the change-aware decoder on held-out matches
(current: spellF1 0.691):

- **Restart shapes** (`line_features`, `MTAH_LINES=1`): players standing on a touchline, at a corner
  flag, in the six-yard box, packed in a penalty area; ball distance to the nearest line. 0.689: up
  on one match, down on two.
- **Separate costs for entering and leaving dead ball**: best 0.693 over a 4 × 4 grid. Within noise.
- **Minimum dead-ball length** in the decoder, 1–4 s (`viterbi_mind`, `min_dead`; exact, checked
  against a brute-force solver): 0.691–0.694. The decoder already rarely outputs short stoppages.
- **Cleaned training labels** (`clean_labels`, `MTAH_CLEAN=1`): the open data marks the ball dead
  wherever it has no ball position, which creates 188 dead stretches under 2 s in three matches,
  while the shortest stoppage in the public reference is 5.5 s. Giving those back to play raised
  recall at IoU 0.5 from 0.597 to 0.605 but lowered precision from 0.839 to 0.827; spellF1 0.6925,
  and 0.698 against 0.698 when scored on equally cleaned labels (`mind_exp.py`).

None moves spellF1 by more than 0.003, so all are off. The remaining misses are limited by the
evidence (what the players and the smoothed ball show), not by the decoder's constraints.

## Tried and dropped: full graphical time warping

The movement-order features estimate each player's lead or lag independently. Full GTW (`gtw_full.py`,
`MTAH_GTW=1`) solves all players' offsets jointly in each 3 s window (every 1 s): each player's data
cost is minus the correlation between their acceleration and the opposing team's, shifted by −1.44 to
+1.44 s; offsets are expressed in one frame ("home ahead by τ") so teammates and opponents can be
compared; each player is linked to their 3 nearest players with a penalty λ·|τᵢ − τⱼ|. The problem is
solved exactly by one minimum cut on Ishikawa's layered graph (checked against brute force), in about
5 s per match. Features: mean offset, per team, spread, front-to-back slope, agreement.

Held-out, with the change-aware decoder (current 0.691): with graph smoothing (λ = 0.05) 0.681,
without (λ = 0) 0.678. The smoothing helps a little relative to independent offsets and sharpens the
dead-ball signal (AUC 0.687 → 0.728 on one match), but on top of the existing features both versions
are worse, so full GTW is off.

## Tried and dropped: removing extra tracks

On 8.1% (home) and 4.7% (away) of graded frames the automatic tracking has more than 11 players per
team; the open training matches never do. Removing near-duplicate tracks (two slots within 1 m) and
tracks more than 2 m off the pitch (`clean_tracks.py`, `MTAH_DEDUP=1`) removed 50,000 positions but
only lowered the home rate to 7.3%: most extras are tracks about 4 s old, several metres from any
teammate, likely re-acquired players whose old track has not ended. It changed 0.3% of frames and the
public window by one possession (spellF1 0.694 → 0.682), so it is off.

## Limitations

- **Short possessions are missed.** The submission has 283 possessions of mean 11.8 s. With the
  tree players model, recall at IoU 0.5 on held-out open matches was 0.26 for possessions under 3 s
  and 0.86 for those over 12 s; two thirds of the misses are possessions with no completed pass
  (`analyse_errors.py`, `misses_by_situation.py`, `loss_breakdown.py`).
- **Dead-ball boundaries are soft.** Player slowing lags the whistle by a second or two, and quick
  restarts are sometimes labelled late.
- **Training labels differ from the reference.** Event-derived possession on three matches from one
  provider is not the corrected-tracking definition of this challenge.
- **The training ball is clean; the provided ball is not.** False ball detections can still pull
  the control model.
- With more time: the broadcast recording, synchronised to the tactical timeline from the two
  tracking files (per-stretch time offsets chained by dynamic programming, replays left out), for
  touch evidence on quick turnovers, which are most of the remaining misses; a confidence filter so
  the video ball is used only where it is reliable; a second stage trained on the provided match
  with pseudo-labels from this pipeline, which could use ball visibility and Smart Tagging as
  features.

## Reproduction

```
pip install -r requirements.txt                    # Python 3.11
git clone https://github.com/metrica-sports/sample-data
git -C sample-data checkout e706dd506b360d69d9d123d5b8026e7294b13996   # the version used (April 2021)
export MTAH_DATA_DIR=/path/to/data                 # contains tactical/BAR-ATH_tactical_atd.{xml,txt}
export MTAH_EXT_DIR=/path/to/sample-data/data      # default: ./sample-data/data
./run_all.sh                                       # writes out/submission.csv
```

`run_all.sh` deletes all caches and models and rebuilds everything, each step in its own process:
`mtah.tracking` (parse the tracking), `mtah.open_data` (open matches and their labels), `mtah.trees train`
and `mtah.trees predict` (the three gradient-boosted models), `mtah.deep.predict` (the committed deep players model
`weights/deep_players.pt`, applied on CPU) and `mtah.decoder` (the change-aware Viterbi decoder). All random seeds are
fixed, and two rebuilds from scratch give byte-identical files. It takes about five minutes on a laptop.
`python -m mtah.deep.train` retrains the deep model (about 8 minutes on an Apple GPU; `--holdout N` for the
held-out folds); this is not needed to rebuild the submission.
