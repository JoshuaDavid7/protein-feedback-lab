# Experimental protocol

## Question

Under an equal measurement budget, do PPO or DPO improve the discovery and
recommendation of GB1 variants compared with random sampling, a frozen sequence
prior, reward-weighted supervised learning, and random-forest acquisition?

This is a retrospective, finite-candidate experimental-feedback benchmark.
Measurements were published in 2016. No new experiments are performed. It is
not evidence that a small model can replace a frontier protein foundation model.

## Data and partitions

Use all 149,361 measured four-site variants in the pinned GB1 archive. Do not use
FLIP's `keep` field: its construction preferentially retains variants above a
fitness threshold. That mask would change the candidate-pool prevalence using
answers unavailable to a prospective acquisition policy.

The policy receives the 56-residue GB1 domain, not FLIP's longer fusion construct.
Its actions are amino acids at positions 39, 40, 41 and 54 (wild type VDGV).

For each seed:

1. Hash-order low-order variants (Hamming distance <= 2). Reserve 10% of their
   sequences to select the initial model checkpoint using action NLL. Train the
   sequence prior on the remainder, without fitness labels.
2. Select 128 low-order training variants as the common initial labeled set.
   This set does not overlap the initial sequence-validation partition.
3. Hash-order variants with distance >= 3. Take the registered acquisition and
   evaluation pool sizes from disjoint consecutive slices. No pool is selected
   or balanced using fitness.
4. Every method receives the same initial labeled set and candidate sequences.
   New labels are available only through a budgeted, no-repeat measurement API.
5. Freeze every method's recommendations for every seed before reading the
   evaluation labels. Save the recommendations and their hashes to disk.

The data are public: this is an auditable software access boundary, not
independent external blinding. Low input-read-count measurements are retained;
their noise is an explicit limitation, not silently removed after seeing results.

## Model and training

The causal transformer uses learned token/position embeddings, pre-layernorm,
multihead causal attention and GELU feed-forward blocks. It is initialized from
random weights. No pretrained ESM or other downloaded weights are used.

The CPU pilot trains a small domain policy directly. The GPU study first trains
a larger small transformer on a reservoir sample of UniProt/Swiss-Prot sequences,
then adapts it on low-order GB1 sequences. Pretraining splits accessions by hash;
this is not a homology-disjoint protein-generalization experiment. The corpus
checkpoint is shared across the five GPU study seeds; subsequent adaptation,
pool assignment and acquisition are seeded separately. Do not describe those
five runs as five independent corpus-pretraining replications.

GB1 adaptation maximizes only the four action-site conditional log probabilities.
Reporting a low loss on 52 fixed residues would exaggerate the learned task.
Checkpoint selection uses low-order validation action NLL, never high-order
evaluation fitness. Model counts, losses, gradient norms, software versions,
source checksums, wall times and seeds are recorded.

## Acquisition distribution and PPO

For remaining measured candidates C, let p_theta(v) be the product of the four
causal amino-acid probabilities. Acquisition samples q_theta(v) proportional to
p_theta(v) within C. Candidate sequences are known; their fitness is hidden until
queried. We do not claim unrestricted de novo generation with measured rewards
for sequences outside this historical dataset.

A batch is drawn without replacement. For ordered draws v_j, log probability is
log q(v_j) minus log(1 - sum of q(v_i) over earlier draws). Both old and updated
policy probabilities include this changing denominator. Treating these draws as
iid would give the wrong PPO likelihood ratio.

PPO uses the clipped surrogate (epsilon 0.2) and within-batch reward-to-go on
log(1 + measured fitness). Earlier choices affect later availability, so using
only each draw's immediate reward would omit future-choice effects. The baseline
is a scalar mean reward from previous measurements, multiplied by remaining
batch length; reward scale is estimated from the initial labeled set. The
reference KL averages the conditional distributions at the visited prefix states.
Each acquisition batch is a short episode; there is no cross-batch credit
assignment, GAE, learned critic, or novel RL algorithm. An enumeration test checks
that the reward-to-go gradient matches the exact expected batch-return gradient.

## DPO and baselines

DPO draws strict preference pairs from observed measurements only, skips ties,
and applies the reference-relative logistic objective with beta 0.1 to the
normalized causal four-action policy. Initial low-order observations and newly
measured high-order observations are both valid preference examples in this
20^4 action space. Acquisition still conditions on the remaining measured pool.
These are measurement-derived preferences, not human preferences.

Reward-weighted SFT updates the same architecture on observed variants, using
softmax weights from scaled log fitness. Frozen-prior sampling performs no
updates. RF-UCB refits a 64-tree forest on one-hot action residues and uses mean
plus one tree standard deviation for acquisition. Tree dispersion is a heuristic,
not a calibrated uncertainty statement. RF's final held-out recommendations use
its predictive mean. Random sampling is uniform. All methods use exactly the
same new-measurement budget; wall times are reported separately.

## Outcomes and interpretation

Primary: best newly measured fitness, number of acquired variants above wild
type, and mean measured fitness of the model's top 20 unqueried recommendations.
Secondary: top-20 hit count, rank correlation, acquisition curves and wall time.
All registered methods and seeds appear in the report. Show per-seed points and
mean/sample SD, without calling a small seed set proof of statistical superiority.

The configuration and environment are frozen before a run. Do not tune on the
held-out results or suppress a losing method. Bug fixes require a new code
revision and a new output directory. Completed result files are never overwritten.

## What this supports

- Implementing and training a small causal transformer from scratch.
- Implementing PPO and preference optimization, with tests of the actual objectives.
- A measured-data feedback loop, equal-budget comparisons and reproducible evaluation.
- For an executed GPU run: single-GPU training, checkpointing and profiling.

It does not establish wet-lab experience, prospective biological validation,
frontier-scale training, multi-node GPU experience, diffusion-model experience,
or leadership of an industrial AI research program.

