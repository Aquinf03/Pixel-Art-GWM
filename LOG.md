# Build log — pixel-art character world model

Running record of what was done, why, what broke, and what the numbers were.
Newest entries at the bottom. Every measurement here was produced by a command
in this repo, not estimated.

---

## 2026-09-01 — Session 1

### Step 0. Installed the aq framework
`curl -fsSL https://aq.aquin.app/framework/install.sh | bash`, after reading the
script first (no sudo; installs to `~/.local/share/aquin-framework`, launcher at
`~/.local/bin/aq`).

Kernel venv landed on Python 3.14 with real wheels, no source builds:
torch 2.13.0 (**MPS available: True**), transformers 5.16.1, datasets 5.0.1,
peft 0.20.0, scikit-learn 1.9.0, xgboost 3.4.1.

Note: there is no `aq version` subcommand — verify with `aq help` / `aq doctor`.

### Step 1. Verified the brief's claims before building on them
The brief's §10 asked for these to be checked first.

| Claim | Status |
|---|---|
| arXiv 2606.27326 | **Real.** Hansen & Wang, *Hallucination in World Models is Predictable and Preventable*, submitted 25 Jun 2026 |
| `nicklashansen/dreamer4` | **Real.** MIT, 390 stars. *Unofficial* PyTorch reimplementation of Dreamer 4, applied to DMControl |
| dreamer4 dataset | **Real.** MIT, 30.9 GB, 7,200 trajectories / 3.6M frames |
| "cheap rental, 2 GB GPU" | **Only true for inference.** README says training the tokenizer/dynamics wants >256 GB RAM and 8 GPUs >24 GB; ~24 h tokenizer + ~48 h dynamics on 8×3090 |

**Open question #1 answered — the dev machine:** Apple M2, 8 cores, 16 GB RAM,
macOS 26.3.1. The binding constraint is not the chip: **7.7 GB free disk**
(228 GB volume, 96% full). The dreamer4 dataset alone is 30.9 GB, so it cannot
be downloaded locally. This is why everything moves to Modal.

**Modal:** CLI 1.4.3, authenticated. Three profiles; active `sachin-37-73-17`
verified live. The `aquin` profile fails authentication (unused).

### Step 2. Environment — `env/pixel_world.py`
128×128 RGB at native resolution (no upscaling, so the tokenizer sees real
pixel art rather than interpolation blur), 15 fps to match MMBench2, 7 discrete
actions, deterministic given a seed, gym-like `reset`/`step` so the same object
can serve as the live simulator phase 5 needs.

Deliberate design choices:
- **Actions must be visually consequential**, or the action-marginalisation
  probe has nothing to detect. Attack shows a slash, crouch changes the
  silhouette, walking cycles the legs.
- **A low-coverage corner by construction** — a high ledge (P4) that takes a
  4-hop climb to reach.

### Step 3. Action encoding — `env/actions.py`
7 discrete buttons ↔ MMBench2's 16-dim zero-padded continuous format with a
validity mask. We claim 4 dims (move / vertical / attack / interact) and mask
the other 12. The mapping is invertible, which the action-inversion probe needs.

### Step 4. First smoke test — **failed**
20 episodes ran, but **total reward across all of them was 1.0**. Three real
bugs, found by tracing one episode step by step:

1. **Policy.** Picked the nearest coin by *horizontal* distance only, so from
   the start it locked onto a coin 60px overhead on an unreachable platform and
   jumped in place forever.
2. **Level.** The spawn sat under a platform with 6px of head clearance —
   jumping just bonked.
3. **Reachability was assumed, not checked.** I had hand-computed the jump arcs.

### Step 5. Fix for (3): derive reachability instead of asserting it
Wrote `env/nav.py`, which simulates the *actual physics* from every launch x
and every hold direction and records where the player lands. The hop graph is
now a measured property, so changing a physics constant can never silently
strand a platform.

It immediately paid for itself: the level was **only 2/5 platforms reachable**.
Ground→P1 worked, but P1 was a dead end upward because P3 overhung P1's launch
edge and capped the jump.

### Step 6. Searched the layout space
Rather than hand-tune arcs a second time, searched candidate layouts and kept
only those `nav.py` proved fully connected. Added a `render=False` path to
`step()` first — rasterising every frame made the search ~100× slower than it
needed to be.

Result: 13,337 layouts sampled in 90 s, 3,902 fully connected.

### Step 7. Rewrote the policy to route over the graph
`ScriptedPolicy` now BFS-routes platform to platform. Reward went from
**1.0 total across 20 episodes → 3.80 mean per episode** (random baseline 0.93).

### Step 8. Looked at the frames
Rendered a 12-frame contact sheet (`artifacts/contact_sheet.png`). Art is
coherent, the climb reads clearly, coins vanish when collected. But frames
145→200 were **pixel-identical**: the agent reached P3 and idled for 60 steps.

**Bug 4:** `LAUNCH_TOL` was 1.2 while `MOVE_SPEED` is 1.6 — the player steps
*over* the tolerance band and oscillates in front of the jump forever. Static
frames are worse than useless as training data; they teach the model that
nothing moves.

**Bug 5:** `Nav` defaulted to `step=2`, scanning only even launch positions,
while the band-finder needed *consecutive* integers — so every band collapsed
to width 0. The real bands are wide (ground→P1 works from x=9..33).

### Step 9. Re-searched, scoring robustness rather than mere connectivity
New score: fully connected, 4-hop route to P4, and **every launch band on that
route ≥ 10px** (must exceed `MOVE_SPEED` or the oscillation returns).
8,304 sampled → 2,286 connected → final layout has bands 10–24px wide.

Also added a stuck-detector: 12 steps without moving triggers a random action.

### Step 10. Goal-conditioned episodes
The robust layout created a *new* problem: the scripted policy now reached P4
in **100%** of episodes, which destroys the coverage experiment.

Fix, and it matches the paper better than what it replaced: each episode targets
one coin, sampled from a skewed distribution (30/28/22/15/**5**%), and ends when
that coin is collected. **A goal is a "task"**, so the paper's mitigation #1
(resample uniform across tasks rather than frames) becomes directly
implementable instead of something we would have had to fake.

Measured over 300 episodes — note episode share and frame share are deliberately
mismatched, the same heavy-tailed structure the brief describes for MMBench2:

| goal | designed | episodes | frame share | success |
|---|---|---|---|---|
| P0 | 30% | 22.7% | 3.7% | 97% |
| P1 | 28% | 28.3% | 28.9% | 73% |
| P2 | 22% | 24.0% | 25.9% | 78% |
| P3 | 15% | 19.0% | 30.5% | 68% |
| **P4** | **5%** | **6.0%** | **11.0%** | **50%** |

P4 is attempted in 6% of episodes and succeeds in half of them, so the ledge
region is reached in ~3% — genuinely low coverage.

Mean per-frame pixel change 0.48% (0% would mean dead frames).

### Step 11. Storage and throughput, measured
2,865 frames/s on one core; **291 bytes/frame** compressed (vs 49,152 raw at
128×128 — 169× compression, because flat pixel art deflates hard).

| target | frames | size | cost |
|---|---|---|---|
| brief min | 200k | 0.06 GB | 0.02 core-h |
| brief max | 500k | 0.15 GB | 0.05 core-h |
| **chosen** | **2M** | **0.58 GB** | **0.19 core-h** (~25 s on 32 cores) |

2M frames is effectively free, so the brief's 200k–500k was conservative.
Resolution costs *training* compute, not storage.

### Status
No training has started. No GPU has been used. Phase 1 only.

### Step 12. Modal app and the corpus
`modal_app.py` — debian_slim + pygame, `SDL_VIDEODRIVER=dummy`, Volume
`pixel-world-data`, one container per shard. Shards are kept small: frames sit
uncompressed in RAM before `savez_compressed`, and 49 KB/frame raw means 200
episodes already peaks ~1.5 GB.

**Bug 6, caught by `check_splits.py`:** the human-play seed block
(900,000..900,499) collided with train shard 9 (900,027..900,226) — 200 shared
seeds. A train seed leaking into an eval split silently inflates every rollout
metric we would later report. Moved the human base to 500,000,000,000; all five
blocks now provably disjoint.

Also fixed: the manifest writer only saw the current run's shards, so topping a
split up would have shrunk its manifest. It now scans the volume.

**Corpus (all on Modal, nothing on the Mac):**

| split | shards | episodes | frames | size |
|---|---|---|---|---|
| train | 150 | 30,000 | **2,005,419** | 0.604 GB |
| val | 12 | 2,400 | 154,379 | 0.047 GB |
| test | 12 | 2,400 | 163,803 | 0.049 GB |
| probe_ledge | 16 | 2,400 | 306,106 | 0.091 GB |

Train coverage, episode share vs frame share (the mismatch is the point):

| task | episodes | frames | ratio |
|---|---|---|---|
| P0 ground | 29.8% | 7.2% | 0.24 |
| P1 | 28.2% | 31.0% | 1.10 |
| P2 | 22.2% | 26.0% | 1.17 |
| P3 | 14.8% | 26.7% | 1.80 |
| **P4 ledge** | **4.9%** | **9.2%** | **1.88** |

`probe_ledge` is 100% P4, held out — the hallucination predictors need ground
truth in the region the model is expected to hallucinate in, which by
construction the training corpus does not cover.

### Step 13. aq wiring
`instructions.md`, `recipe.yaml` (data section live, model section marked
phase 3), and three tools: `aq tool collect`, `aq tool modal_collect`,
`aq tool coverage`. Tools are `.sh`, not `.py`, because aq runs `.py` tools
with its kernel python, which has no pygame.

### Phase 1: complete
Remaining optional input: human play (`play.py`) runs on the Mac because it
needs a real display. Not required to start phase 2.

Still no training. Still no GPU.

---

## 2026-09-01 — Session 2: attacking the action-conditioning risk

Phase 2's biggest risk is **action marginalisation** — the model ignoring what
you press. The paper traces it to the dynamics component. Question asked: would
fewer buttons reduce it?

### Step 14. Measured it instead of guessing — `measure_actions.py`
From states sampled out of real rollouts, branch on every action, render each
outcome, and measure how many pixels separate the outcomes of two actions taken
from the *same* state. An action whose branch collapses onto another's is
unlearnable regardless of how its vector is encoded.

Original 7-action env, single step:

```
least separable pair: idle vs interact (0.00%)
left 0.41%  right 0.39%  jump 0.28%  attack 0.22%  crouch 0.22%  idle 0.17%  interact 0.17%
```

**`interact` was a literal no-op**: 0.00% different from `idle`, because it
rendered nothing except in the single frame it opened the door. And the best
pair in the whole action set, left vs right, differed in **0.66%** of pixels —
about 108 of 16,384.

### Step 15. Action repeat helps, but cannot fix a no-op
Holding each action for 1/2/4/8 steps: left/right separability 0.41% → 1.21%
(~3×), most of the gain by hold=4. But `idle` vs `interact` stayed at exactly
**0.00% at every horizon**. Holding a no-op longer is still a no-op.

### Step 16. Found the actual ceiling
Separability is capped by how much of the frame the sprite occupies — two
disjoint sprite positions can differ by at most twice its area.

| sprite | area | ceiling | measured | % of cap |
|---|---|---|---|---|
| 9×13 | 0.71% | 1.43% | 1.49% | **104%** |
| 14×20 | 1.71% | 3.42% | 2.65% | 77% |
| 18×26 | 2.86% | 5.71% | 4.99% | 87% |

At 9×13 we were **already saturated**. No encoding change could have helped;
the fix had to change the pixels.

**So the answer to "would fewer buttons help": only in one specific way.**
Cutting cardinality does nothing on its own — 7 corners of a hypercube versus 6
is not what makes an action hard to learn. Cutting the *unlearnable* action
does, and enlarging the sprite does far more.

### Step 17. Swept the two real levers
| sprite | speed | mean sep | left\|right |
|---|---|---|---|
| 9×13 | 1.6 | 0.32% | 0.69% (1.0×) |
| 14×20 | 2.6 | 0.59% | 1.35% (2.0×) |
| **14×20** | **3.6** | **0.70%** | **1.75% (2.6×)** |

### Step 18. Implemented the fix
1. **Merged `attack` + `interact` into one context-sensitive `use`** — it breaks
   a crate or opens a door depending on proximity, and always draws its swing.
   7 actions → 6.
2. **Sprite 9×13 → 14×20** (0.71% → 1.71% of frame).
3. **Move speed 1.6 → 3.6**; jump velocity −6.0 → −6.6 to clear the new spacing.
4. **Five tiers → four** — a 20px sprite needs ~28px of vertical spacing.
   Re-searched the layout (8,412 candidates): 3-hop route to the ledge, every
   launch band ≥24px. Graph is 5 edges, sparse and shortcut-free.
5. Action vector now claims 3 valid dims of 16 (move / vertical / use).

### Result

| metric | before | after | |
|---|---|---|---|
| worst pair | **0.00%** (idle/interact) | **0.22%** (idle/use) | no longer a no-op |
| left vs right | 0.66% | **2.12%** | 3.2× |
| idle | 0.16% | 0.65% | 4.1× |
| left | 0.41% | 1.36% | 3.3× |

Every action pair is now distinguishable from the frames alone.

### Step 19. Rebuilt the corpus
The v1 data has a 7-action space that no longer exists, so it was deleted
rather than left to mix in. Every shard now carries `action_space_version=2`
and `n_actions`, so a stale shard cannot silently join a training run.

### What this does *not* fix
The remaining half of the risk is a **distribution** gap, not a visibility one:
dreamer4's conditioning was trained on dense continuous DMControl vectors,
while ours are sparse corners of a hypercube ({−1,0,1}). Fewer buttons does not
change that. The available lever is feeding the model a temporally smoothed
action vector alongside the raw discrete one — continuous trajectories in
action space, closer to DMControl statistics — while keeping the raw discrete
channel for the action-inversion probe. Not implemented; worth deciding in
phase 2 once we can read dreamer4's action-conditioning code.

### Step 20. Corpus rebuilt on the 6-action space

| split | shards | episodes | frames | size |
|---|---|---|---|---|
| train | 175 | 35,000 | **2,025,385** | 0.568 GB |
| val | 12 | 2,400 | 136,620 | 0.038 GB |
| test | 12 | 2,400 | 139,157 | 0.039 GB |
| probe_ledge | 16 | 2,400 | 364,684 | 0.101 GB |

Coverage held its shape: P3 ledge 6.0% of episodes, and the episode/frame
mismatch is sharper than before (P0 is 31.9% of episodes but 5.4% of frames;
P3's ratio is 2.59).

---

## 2026-09-01 — Session 3: phase 2 reconnaissance

Read the repo before spending any GPU money. This changed the plan twice.

### Step 21. Repo health (brief §10)
MIT, 390 stars. **Last code commit 2026-03-25 — five months stale**; the July
push was README-only. 3 open issues, and the author has replied to none of the
two that matter:

- **#9 "World model rollouts are blurry"** — a user trained 145k dynamics steps
  on 4×A100 with a healthy tokenizer (400k steps, loss <0.005) and total loss
  converged below 0.003, and rollouts are *still blurry*. A second user reports
  worse. **This is the single biggest risk to the whole plan** and it is
  unanswered.
- **#11** — unanswered question about a train/inference mismatch in the dynamics
  model: latents are all-noisy in training but clean-concatenated-with-noise at
  inference.

Consequence: the first thing phase 2 must do is check whether the *released*
checkpoint's own rollouts are sharp. If they are not, dreamer4 is the wrong
base and we pivot to from-scratch flow-matching before spending real money.

### Step 22. The action risk is smaller than assessed — evidence, not reasoning
`interactive.py:379 build_action_from_keys` maps keyboard keys to the action
vector by selecting **one** action dimension and driving it to ±1. Then
`interactive.py:626` EMA-smooths it:

```python
st.a_smooth = (beta * st.a_smooth + (1.0 - beta) * a_raw)   # beta default 0.817
```

So dreamer4's own inference loop already feeds the model **sparse ±1 corner
vectors**, exactly the shape our discrete buttons produce — not the dense
continuous DMControl vectors I assumed. The action vector is literally 16-dim
(`torch.zeros(16, ...)`), matching our encoding.

Two consequences:
1. The discrete-vs-continuous gap I flagged as the top risk is **much narrower
   than it looked**, and the mitigation I proposed (temporal smoothing) is
   already implemented upstream with a tuned default.
2. There is a mismatch in *their* setup worth avoiding in ours: training reads
   raw dataset actions while inference sees EMA-smoothed ones. We should write
   **EMA-smoothed actions into our training data at beta=0.817**, so train and
   inference see the same action statistics. Cheap, and it costs nothing to do
   at collection time.

### Step 23. Found the undocumented custom-env registration path (brief §10)
The README says only "modify data directory paths". The real path is
**`tasks.json`** — 234 registered tasks, each:

```
embodiment         '2D walker with 6 controllable joints across 2 legs'
instruction        'Stand upright and maintain balance'
action_dim         6
max_episode_steps  500
text_embedding     <512 floats>
```

- `action_dim` across existing tasks spans 1–12, so **our 3 valid dims fit an
  existing bucket**; `max_episode_steps` spans 25–1000, so our 200 fits.
- `text_embedding` is **optional** — `wm_dataset.py:291` falls back to a
  512-dim zero vector when absent. We have one task and no language ambition,
  so we can register without producing an embedding.

### Step 24. Dataset format for the adapter
- `<data_dir>/<task>.pt` — a TensorDict with `episode`, actions, rewards
- `<frames_dir>/<task>/*shard*.pt` — `{"frames": tensor}`, 2048 frames/shard
- `TARGET_SIZE = 128` — **matches our render resolution exactly**, so no
  resampling and no tokenizer resolution mismatch

Writing `npz -> .pt` is the concrete adapter job.

### Step 25. Probed the released checkpoint — the base model fails on our domain

Ran the released `tokenizer.pt` (257 MB, 90k steps) on an A10G against six of
our own frames. Tokenizer config: 128×128×3, patch 4, 16 latents, d_bottleneck
32 — resolution matches ours exactly.

**Round-trip PSNR: 18.56 dB** (`artifacts/roundtrip.png`).

The number understates it. The reconstructions are not blurred pixel art —
they are **a different scene**. Platforms, coins, crate and door are gone, and
the orange character is rebuilt as what looks like an articulated robot arm on
a blue-grey robotics background. The reconstruction tracks the sprite's
*position* but rebuilds the *scene* from the training distribution.

Two tells that this is category error rather than mere quality loss:
- PSNR is 18.62 on five of six different frames — nearly constant, i.e. largely
  independent of frame content.
- The output contains structures that appear nowhere in our environment.

This is textbook **perceptual hallucination** — the exact failure the paper
attributes to the tokenizer, and precisely what brief §6 predicted: "the frozen
encoder may snap our sprite onto something it knows."

**Consequences, and they are significant:**
1. Tokenizer finetuning is **mandatory**, not the "expect to" hedge in §6.
2. The released dynamics checkpoint was trained on the *frozen* tokenizer's
   latent space. Finetuning the tokenizer moves that space, so the pretrained
   dynamics stops being valid — brief §4's warning that "stages compose
   sequentially, an early failure propagates and amplifies downstream."
3. So the pretrained prior buys much less than the plan assumed: it is a prior
   over *robotics video*, and we would end up retraining both stages anyway.

### Step 26. Dynamics probe — sharp, but in the wrong universe

Loaded `dynamics.pt` (512 MB, 40k steps, action-conditioned; k_max 8, n_spatial
8, d_spatial 64, packing_factor 2), seeded from *our* encoded frame and rolled
12 steps holding "move right" (`artifacts/rollout.png`).

The rollout is **sharp, coherent and temporally consistent** — and it is a
**DMControl walker**: a two-legged articulated figure on a checkered floor,
moving plausibly. From our pixel-art seed, the model immediately reconstructs a
walker and continues in *that* world.

Two findings:
1. **Issue #9 does not apply to the released checkpoint.** The published model
   is not blurry. The blurriness reports concern users reproducing the
   *training*, which is a different risk (one we now inherit, since we must
   train).
2. **The pretrained weights are unusable for us.** Not degraded — categorically
   wrong. The prior is over robotics video.

### Phase 2 decision: keep the architecture, drop the weights

The evidence points somewhere other than either option in the brief's §5:

- **Not "finetune the pretrained checkpoint"** (§5's recommended path). The
  tokenizer cannot encode our domain, and finetuning it invalidates the
  pretrained dynamics that was trained on its frozen latents. We would be
  retraining both stages regardless, having gained a robotics prior we do not
  want.
- **Not IRIS from scratch** (§5's fallback). A discrete VQ + categorical
  transformer has no denoiser, so `u_f` (flow instability) and `u_s`
  (inter-seed variance) would not exist. Two of the three predictors would be
  unportable.
- **Instead: dreamer4's architecture and training code, trained from scratch on
  our corpus, at reduced size.** The rollout above proves the architecture
  produces sharp action-conditioned video; its shortcut flow-matching dynamics
  is exactly what `u_f` and `u_s` need; and our domain is far simpler than the
  30-task DMControl corpus it was sized for (one environment, 6 actions, flat
  art, 2M frames), so it should not need the 8-GPU budget the README quotes.

Both remaining risks are now training risks, not transfer risks, and issue #9
is the one to watch: it says reproducing their training is where people get
stuck. Mitigation is that our target is much easier than theirs.

---

## 2026-09-01 — Session 4: the character, and phase 3 wiring

### Step 27. Redrew the character
Feedback: the sprite was too basic. It was — a block with one eye, drawn by
stacking rectangles. Worse, it was a measurable problem: **facing right vs left
changed only 8 pixels (2.9% of the sprite)**, the eye sliding across the head.
`left` and `right` are the two most-used actions, so the model could tell the
character had moved but not which way it was looking.

Rewrote the art as hand-authored pixel maps (`env/sprite.py`) rather than
composed rectangles — at 14x20 every pixel is a decision, and rectangles cannot
express an outline, a shaded belly or a tail. Frames are validated at import;
the width check caught three of my own errors immediately.

Three passes:
1. Blocky creature with snout + pack: facing 2.9% -> 29.3%.
2. Added a directional *lean* (shift the sprite toward its facing): 75.7%.
3. **Dropped the lean.** It was compensating for a body that was still
   symmetric. Re-authored as a genuine side profile — mass pushed forward,
   tail sweeping back and up — and mirroring alone now moves **71.4%** of the
   sprite's pixels, with `sprite.LEAN` left at 0.

Full action separability improved across the board (`measure_actions.py`):
left 1.36% -> 1.56%, use 0.78% -> 0.93%, worst pair 0.22% -> 0.32%.

Published the character at `artifacts/character.html`.

Corpus rebuilt again (art changed the pixels). Shards now carry `art_version`
as well as `action_space_version`, so v2 art cannot silently join a v3 run.
Train: 2,025,385 frames, 0.746 GB — larger than v2's 0.604 GB, since the
detailed sprite compresses less (368 vs 301 bytes/frame).

### Step 28. Stage-0 blocker found
The smoke run failed with "no usable sequences found in outdirs". Two mistakes,
both mine: `ShardedFrameDataset` takes `outdirs`, not `data_dirs`, and it wants
the **root** containing `<task>/*.pt`, not the task directory itself.

### Step 29. aq wired for real
Until now aq was a folder skeleton — honestly so, because phases 1 and 2 had no
model to fit. Phase 3 does, so:

- **`methods/world_model.py`** — `fit()` shells out to `modal run` (aq runs
  methods with its kernel python, which has no modal package) and returns a
  JSON record. aq stores checkpoints as JSON (`engine/step.py` uses
  `json.loads`), so the checkpoint is a **pointer** into the `d4-runs` volume
  plus the numbers describing the run. That split is right: the Mac has 7.7 GB
  free, and what deserves versioning is which run produced which score.
- **`evaluate()`** returns round-trip PSNR — the stage-1 gate. `psnr` is not in
  aq's `LOWER` set, so `eval.min_score` acts as a floor.
- **`evals/roundtrip.jsonl`** — the probe spec.
- **`recipe.yaml`** is now live: `min_score: 30.0` against the released
  checkpoint's 18.56 dB on our art.
- **`write_inspect()`** records the run and, deliberately, says to look at the
  image rather than trust the number — the released checkpoint scored a
  plausible 18.56 dB while drawing something else entirely.

Stage 2 (dynamics) is deliberately not wired: the stages compose sequentially,
so a weak tokenizer would silently waste the dynamics run.

---

## 2026-09-02 — Session 5: phase 3 launch

### Step 30. Moved to the aquin workspace
The original workspace blocked A100/H100 behind a payment method, forcing an
A10G whose 22 GiB cannot hold dreamer4's default batch 8 x seq 8 at 128x128.
`aquin` has both available (verified: A100-SXM4-40GB and H100 80GB), so stage 1
runs their intended config rather than a shrunken one. Volumes are
per-workspace, so the whole corpus was re-collected and re-converted there.

### Step 31. Overfit sanity gate — passed on the evidence that matters
256 frames, 600 steps, A10G. Loss **0.4487 -> 0.0711, an 84% drop**, cleanly
descending; LPIPS fell fastest (0.474 -> 0.200), which is the term that fights
blur on hard edges. The pipeline trains.

Two process failures worth recording:
- I had wrapped the trainer in `subprocess.run(capture_output=True)`, so a run
  that went 17 minutes against a 5-minute estimate showed **nothing** on the
  dashboard. Replaced with a streaming runner. Do not run long jobs blind.
- The trainer **hangs after its final step** — teardown never returns. The
  runner now detects completion from the output and terminates the hang,
  treating it as success since checkpoints are already on disk. Without this a
  60k-step run would have hung forever at the finish line.

### Step 32. aq's recipe validator
`aq train` refused with "recipe.yaml must set data.target". Its validator
(`protocol/recipe.py:103`) falls through to a tabular branch for any method it
does not recognise, and `world_model` is not one of its built-ins. Added
`data.target: frames` with a comment; our method ignores it.

### Step 33. The 100 GB mistake

**Stage 1 would not start, twice, and the cause was mine.**

`torch.save` does not compress. Our corpus is 0.75 GB as npz (368 B/frame,
because flat pixel art deflates hard) but the dreamer4 frame format stores raw
uint8: 2048 frames x 3 x 128 x 128 = **96 MB per shard**, x 989 shards =
**~100 GB**. And `ShardedFrameDataset.__init__` `torch.load`s *every shard* to
count frames — so each run read ~100 GB off a network volume before step 0,
printing nothing while it did.

I copied their `preprocess_dataset.py` format without doing the arithmetic on
what it costs at our scale.

This also explains their README's otherwise odd **">256 GB RAM"** recommendation:
it is not for the model, it is to hold the dataset in memory so this scan is
survivable. On Modal with a network volume that is not available to us, so the
format needed adapting rather than copying.

**Fix:** patch the loader to `torch.load(..., mmap=True)` in the image, so the
scan reads tensor headers rather than payloads and pixels load lazily per batch.

Worth revisiting later: 100 GB for a corpus whose information content is ~26
bits of state per frame is absurd. Storing PNG-encoded or npz shards and
decoding in the dataloader would cut it ~130x at negligible CPU cost.

### Step 34. Stage 1 ran — and the gate I wrote was worthless

Training worked. Loss fell 0.4344 -> 0.0080 mse, cleanly. Then the
reconstruction image contradicted the number, exactly as in phase 2.

The whole-frame palette gate said **0.7549** at step 2000. The picture showed
platforms, coins, crate and door reconstructed — and **no fox at all**.

**The gate was passable with no character in it.** The character is 1.30% of
the frame, so a reconstruction that omits it entirely scores **0.987**
whole-frame, above the 0.95 threshold I had set. I built a gate that measures
the easy 98.7% and barely notices the subject.

Replaced with `char_acc`: exact-colour match restricted to pixels whose GROUND
TRUTH is one of the character's own palette entries (indices 8-15, 19).

### Step 35. The real finding: the model *unlearns* the character

| step | char_acc | whole-frame | psnr |
|---|---|---|---|
| 1000 | **0.0151** | 0.4245 | 16.93 |
| 2000 | 0.0015 | 0.7549 | 20.51 |
| 3000 | 0.0010 | 0.7656 | 21.09 |

char_acc **peaks at step 1000 and then falls 15x**, monotonically, while
whole-frame accuracy and PSNR climb throughout
(`artifacts/char_regression.png`).

This is not underfitting. The model learns the fox partially, then actively
trades it away: at 1.3% of pixels, spending latent capacity on the character is
net-negative for a mean-over-patches MSE. **Training longer makes it worse.**

It is the project's own subject appearing one level down — a rare, important
region being averaged away by a loss that treats all pixels alike.

### Step 36. Six-variant sweep — stalled by my own storage bug

Launched `base` / `detail5` / `detail20` / `latent32` / `bneck64` / `wide384`
in parallel, 2000 steps each, ranked on char_acc.

All six stalled at ~step 1800. Zero progress in 100 s across every container.
Cause: six containers doing random reads against the **679 GB** corpus (the
`.contiguous()` slice bug, step 33) collapsed the volume's I/O. One run
tolerated it at 8.7% MFU; six did not.

Two process notes:
- I evaluated `latest.pt` early and reported a ranking. **Those numbers were
  invalid** — `latest.pt` was still the step-0 checkpoint, so I had evaluated
  six untrained models. Identical values across three variants was the tell.
- Modal volume writes are not visible outside the container until `commit()`,
  which my code only called at the end. Checkpoints should be committed at each
  save so progress is externally observable.

### Step 37. Main run relaunched with the fix
`detail_weight: 5.0` — each patch's loss scaled by its own variance, so the
sprite's outline and eye outweigh flat sky and ground. Hang grace cut 300s ->
60s. Gate every 1000 steps against the baseline's known curve, so the verdict
arrives at step 3000 (~18 min) rather than at 60k.

### Step 38. Stopped adapting dreamer4's tokenizer; built one for this domain

The variance-weighted loss was a measured failure: char_acc 0.0151 -> **0.0075**
at step 1000 while whole-frame rose 0.42 -> 0.67 and PSNR 16.9 -> 22.1. It
upweighted every edge in the frame, and there are far more background edges
(platform boundaries, sky bands, coins, crate, door) than character pixels, so
the extra gradient went to them. I fixed "flat regions dominate" without fixing
"the character is 1.3% of pixels" -- different problems.

**The literature says this is a known, published failure.** DIAMOND
(arXiv 2405.12399, *Diffusion for World Modeling: Visual Details Matter*)
documents it in Figure 5: in Asterix "an enemy (orange) becomes a reward (red)
in the second frame"; in Road Runner "the rewards (small blue dots) are
inconsistently rendered between frames". Their diagnosis is ours: "compression
into a compact discrete representation may ignore visual details that are
important for reinforcement learning."

Their fix is to drop the tokenizer entirely for pixel-space diffusion -- which
would also delete `u_r`, since there would be no round trip to measure. The
VQGAN lineage's fix is an adversarial patch discriminator. Both are large.

**What we built instead** (`tokenizer/`, ~2.9M params), exploiting two
properties of this domain that dreamer4's general-purpose design cannot use:

1. **Spatial latents** (8x16x16 = 2048 numbers) instead of 16 *global* vectors
   (512). The fox occupies its own latent cells and cannot be averaged into
   the background -- directly addressing DIAMOND's stated cause.
2. **The decoder classifies over the 21-colour palette** instead of regressing
   RGB. Cross-entropy cannot blur: it picks the right colour or it does not.
   Class weights then express "the character's colours matter more", which MSE
   had no vocabulary for.

It also reads the **npz corpus directly** (0.75 GB), skipping the 679 GB `.pt`
expansion and the 989-shard volume scan that stalled every parallel run.

Result, 1500 steps, ~5 minutes, well under a dollar:

| step | loss | char_acc | palette_acc |
|---|---|---|---|
| 250 | 0.0933 | 0.9094 | 0.9955 |
| 500 | 0.0272 | 0.9842 | 0.9988 |
| 1000 | 0.0094 | 0.9971 | 0.9998 |
| 1500 | 0.0084 | **0.9976** | **0.9998** |

Held out: char_acc **0.99935** (val), **0.99805** (probe_ledge). Reconstruction
is pixel-identical by eye (`artifacts/pal_recon_val.png`). At step 250 it was
already 60x dreamer4's best-ever char_acc, and it climbs monotonically rather
than collapsing.

**A bug worth recording:** the first launch produced no steps at all. The data
loader picked a random shard per batch, and `np.load(p)["frames"]` decompresses
the whole ~570 MB array -- so nearly every step paid a full decompression.
Fixed by drawing 200 batches from each loaded shard.

### `u_r` has room to work now
Round-trip residual, normalised: **0.0048** (val), 0.0036 (probe_ledge).
dreamer4's tokenizer scored 0.996 on our frames against a noise floor of 1.029
-- saturated, with no resolution left to detect anything. Two orders of
magnitude of headroom is exactly what phase 4 needs.

**Stage 1 gate: PASSED** (0.999 against a 0.90 threshold).
