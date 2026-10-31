# Questions to be able to answer without the code open

1. What exactly does the transformer predict? Why is its GB1 loss computed on
   four sites, even though it conditions on a 56-residue domain?
2. Why would filtering candidates with FLIP's fitness-dependent `keep` flag
   make a retrospective discovery benchmark easier?
3. What changes between a joint sequence probability and a distribution
   conditioned on a finite candidate pool?
4. Derive the denominator for the second and third draw in a batch sampled
   without replacement. Where does it enter the PPO importance ratio?
5. What do positive and negative advantages do at PPO's clipping boundaries?
   Why do earlier draws also receive credit for rewards later in the same batch?
6. Why can the reference-relative log-probability difference be used in DPO?
   What are the limitations of treating noisy assay ranks as preferences?
7. Why include reward-weighted supervised learning and RF-UCB? What would an
   RL loss tell us if a simpler method wins?
8. Which labels influence model fitting, checkpoint selection, acquisition, and
   final evaluation? What remains unblinded because this is a public dataset?
9. How much compute did the actual runs use, and what did their learning curves
   and gradient norms show? Which runs share a corpus-pretraining checkpoint?
10. What would change with genuinely prospective experiments: assay replicates,
    batch effects, unmeasured candidates, an experimental failure rate, and
    delayed/noisy observations?

The project was developed with AI-assisted coding. A useful portfolio claim
requires understanding, reproducing and being able to modify the implementation.
Do not claim independent manual authorship of every line or experience that
the experiments do not demonstrate.

