# Development record

The first implementation passed 20 objective, oracle, causality and agent tests
and completed CPU/GPU smoke runs. Its three-seed CPU pilot used immediate-reward
advantages. A subsequent mathematical review identified that without-replacement
batch draws also change the opportunities available to later draws. For the
specified expected batch-return objective, the gradient must include reward-to-go.

Before the registered GPU study, PPO was corrected to use within-batch returns,
and its reference KL was made conditional on every visited prefix state. An
exhaustive small-state test compares the resulting gradient to the exact gradient
of expected batch return. Conditional probabilities now use masked logsumexp
rather than subtracting accumulated probability mass, avoiding cancellation when
the policy concentrates on a few candidates.

The corrected CPU study uses a new output directory. This change corrects credit
assignment; model sizes, dataset partitions, query budgets, learning rates and
other registered hyperparameters were not changed based on test outcomes. Earlier
diagnostic outputs are retained locally and are not presented as final evidence.
