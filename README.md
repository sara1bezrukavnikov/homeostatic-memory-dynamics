# Project Goal

Simulating the homeostatic recovery or loss of a Hebbian-encoded memory post-perturbation.

# Experiment Stages

Build three modules: neuronal dynamics, Hebbian plasticity, and homeostatic regulation.

1. Apply an input `I_m` that drives the network into a stable attractor state (memory formation).
2. Apply strong inhibition `I_p` that suppresses activity and causes the stored pattern, as well as everything else, to decay.
3. Apply the homeostatic function while inhibition is still present and see whether restoring the MFR allows the original pattern to re-emerge or not.

# Metrics

- Memory overlap: mmem(t) = rm · r(t)
- Mean firing rate: mmean(t) = (1/N) Σ ri(t)
- Memory weight strength: leading singular value of ΔW = W - W_struct

# Phases

0. Build a working model of a pair of neurons, testing out the homeostatic, Hebbian, and baseline functions on it.
  - Verify Hebbian strengthening (plot W_12 over time during co-activation)
  - Verify homeostatic convergence (h(t) → steady state, mmean → m*)
  - Verify weight decay toward W_struct when activity = 0
  - Define and fix all time constants

1. Build a simple network, try out the perturbation + homeostasis on it
  - Inject `I_m`, confirm r → rm (stable attractor)
  - Plot mmem(t) rising to 1; plot ΔW singular values showing rank-1 structure

2. In the same network, capture its ability to arrive to a stable attractor state with `I_m` injection
  - Apply `I_p`, confirm activity collapse and weight decay
  - Plot mmem(t) falling; track how much of the rank-1 piece survives as a function of T2

3. "Run the experiment": capture memory, apply perturbation, watch the results as the homeostasis takes course
  - Turn on `h(t)` with `Ip` still present
  - Primary readout: mmem(t) — does memory re-emerge?
  - Secondary readout: `yu(t)`, `yv(t)` from two-mode projection
  - Sweep one parameter (e.g. overlap of rp with rm) to find rescue vs. corruption boundary
