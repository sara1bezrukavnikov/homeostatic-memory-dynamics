# Project Goal

Simulating the homeostatic recovery or loss of a Hebbian-encoded memory post-perturbation.

# Experiment Stages

Build three modules: neuronal dynamics, Hebbian plasticity, and homeostatic regulation.

1: Apply an input `I_m` that drives the network into a stable attractor state (memory formation).
2: Apply strong inhibition `I_p` that suppresses activity and causes the stored pattern, as well as everything else, to decay.
3: Apply the homeostatic function while inhibition is still present and see whether restoring the MFR allows the original pattern to re-emerge or not.

# Metrics

- Memory overlap: mmem(t) = rm · r(t)
- Mean firing rate: mmean(t) = (1/N) Σ ri(t)
- Memory weight strength: leading singular value of ΔW = W - W_struct

# Phases

0. Build a working model of a pair of neurons, testing out the homeostatic, Hebbian, and baseline functions on it. Expecting to see: Hebbian strengthening; ability to restore to baseline post-perturbation; ability to average out to a stable baseline (summing both neurons' activity into one function)
1. Build a simple network, try out the perturbation + homeostasis on it
2. In the same network, capture its ability to arrive to a stable attractor state with `I_m` injection
3. "Run the experiment": capture memory, apply perturbation, watch the results as the homeostasis takes course
