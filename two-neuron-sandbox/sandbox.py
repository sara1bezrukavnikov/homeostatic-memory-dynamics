"""
Two-Neuron Sandbox
==================
Tests three modules in isolation and together:
  1. Hebbian plasticity (weight strengthening during co-activation)
  2. Homeostatic regulation (mean firing rate restoration)
  3. Baseline dynamics (rate model with structural weights)

Follows the framework in: RNN with Hebbian Plasticity and Homeostasis (Feb 2026)

Equations
---------
Rate dynamics:   tau_r * dr/dt = -r + W @ phi(r) + I(t)
Hebbian:         tau_W * dW/dt = -lambda * (W - W_struct) + eta * outer(phi(r), phi(r))
Homeostasis:     tau_h * dh/dt = kappa * (m_star - mean(r))
                 I_h(t)        = h(t) * ones(N)   [uniform injection]
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ─────────────────────────────────────────────
# 0.  NONLINEARITY
# ─────────────────────────────────────────────

def phi(r: np.ndarray, kind: str = "relu") -> np.ndarray:
    """Elementwise nonlinearity."""
    if kind == "relu":
        return np.maximum(r, 0.0)
    elif kind == "tanh":
        return np.tanh(r)
    else:
        raise ValueError(f"Unknown nonlinearity: {kind}")


# ─────────────────────────────────────────────
# 1.  HEBBIAN PLASTICITY MODULE
# ─────────────────────────────────────────────

def hebbian_update(
    W: np.ndarray,
    W_struct: np.ndarray,
    phi_r: np.ndarray,
    eta: float,
    lam: float,
    tau_W: float,
    dt: float,
) -> np.ndarray:
    """
    One Euler step of the Hebbian weight update.

    dW/dt = (-lambda * (W - W_struct) + eta * outer(phi_r, phi_r)) / tau_W

    Parameters
    ----------
    W        : current weight matrix (N x N)
    W_struct : structural baseline weights (N x N)
    phi_r    : phi(r), current nonlinear activity (N,)
    eta      : Hebbian learning rate
    lam      : decay rate toward W_struct
    tau_W    : weight timescale (should be >> tau_r)
    dt       : time step

    Returns
    -------
    W_new    : updated weight matrix
    """
    dW = (-lam * (W - W_struct) + eta * np.outer(phi_r, phi_r)) / tau_W
    return W + dW * dt


# ─────────────────────────────────────────────
# 2.  HOMEOSTATIC MODULE
# ─────────────────────────────────────────────

def homeostatic_update(
    h: float,
    r: np.ndarray,
    m_star: float,
    kappa: float,
    tau_h: float,
    dt: float,
) -> float:
    """
    One Euler step of the homeostatic controller.

    dh/dt = kappa * (m_star - mean(r)) / tau_h

    The controller adjusts a scalar h that injects a uniform current
    h * ones(N) into all neurons, steering the mean firing rate toward m_star.

    Parameters
    ----------
    h      : current homeostatic drive (scalar)
    r      : current firing rates (N,)
    m_star : target mean firing rate
    kappa  : homeostatic gain
    tau_h  : homeostatic timescale (should be >> tau_r, < tau_W)
    dt     : time step

    Returns
    -------
    h_new  : updated homeostatic drive
    """
    m_mean = np.mean(r)
    dh = kappa * (m_star - m_mean) / tau_h
    return h + dh * dt


def homeostatic_current(h: float, N: int) -> np.ndarray:
    """Uniform current injected by homeostasis: h * ones(N)."""
    return h * np.ones(N)


# ─────────────────────────────────────────────
# 3.  RATE DYNAMICS MODULE
# ─────────────────────────────────────────────

def rate_update(
    r: np.ndarray,
    W: np.ndarray,
    I: np.ndarray,
    tau_r: float,
    dt: float,
    nl: str = "relu",
) -> np.ndarray:
    """
    One Euler step of the rate dynamics.

    dr/dt = (-r + W @ phi(r) + I) / tau_r

    Parameters
    ----------
    r     : current firing rates (N,)
    W     : weight matrix (N x N)
    I     : total external input (N,)  -- sum of all input sources
    tau_r : rate timescale
    dt    : time step
    nl    : nonlinearity type ('relu' or 'tanh')

    Returns
    -------
    r_new : updated firing rates
    """
    phi_r = phi(r, nl)
    dr = (-r + W @ phi_r + I) / tau_r
    return r + dr * dt


# ─────────────────────────────────────────────
# 4.  TWO-NEURON SANDBOX
# ─────────────────────────────────────────────

def run_two_neuron_sandbox():
    """
    Runs the full three-phase protocol on a 2-neuron network.

    Phase 1 (write):     inject I_m, watch Hebbian strengthen W
    Phase 2 (suppress):  inject I_p, watch activity collapse and W decay
    Phase 3 (rescue):    keep I_p, turn on homeostasis, watch whether
                         mean rate recovers and memory re-emerges

    What to expect
    --------------
    - Phase 1: both neurons activate; W[0,1] and W[1,0] grow (co-activation)
    - Phase 2: rates drop toward 0; Hebbian drive vanishes; W decays back
    - Phase 3: h ramps up; mean rate recovers toward m_star;
               W may partially re-strengthen if phi(r) has memory overlap
    """

    # ── Parameters ──────────────────────────────────────────────────────
    N       = 2
    tau_r   = 1.0       # fast: rate timescale
    tau_h   = 30.0      # medium: homeostatic timescale
    tau_W   = 500.0     # slow: weight timescale
    eta     = 0.05      # Hebbian learning rate (small — weights are slow)
    lam     = 0.02      # weight decay rate
    kappa   = 0.8       # homeostatic gain
    m_star  = 0.5       # target mean firing rate
    dt      = 0.02      # smaller step for stability

    T1 = 400   # write phase duration
    T2 = 400   # suppress phase duration
    T3 = 600   # rescue phase duration
    T_total = T1 + T2 + T3

    # ── Initial conditions ───────────────────────────────────────────────
    W_struct = np.array([[0.0, 0.1],
                         [0.1, 0.0]])   # weak symmetric baseline
    W = W_struct.copy()
    r = np.zeros(N)
    h = 0.0

    # ── Inputs ───────────────────────────────────────────────────────────
    I_m = np.array([1.2, 0.8])    # memory cue: asymmetric to create a real pattern
    I_p = np.array([-2.0, -2.0])  # strong uniform suppression

    # ── Storage ──────────────────────────────────────────────────────────
    steps = int(T_total / dt)
    t_arr    = np.zeros(steps)
    r_arr    = np.zeros((steps, N))
    W_arr    = np.zeros((steps, N, N))
    h_arr    = np.zeros(steps)
    mmean_arr = np.zeros(steps)
    # Memory overlap: dot product of current r with r at end of phase 1
    # (computed post-hoc once we know r_memory)
    mmem_arr  = np.zeros(steps)

    r_memory = None  # will be set at end of phase 1

    # ── Simulation loop ──────────────────────────────────────────────────
    for i in range(steps):
        t = i * dt
        t_arr[i] = t

        # Determine phase and input
        if t < T1:
            # Phase 1: write
            I_ext = I_m
            homeostasis_on = False
        elif t < T1 + T2:
            # Phase 2: suppress
            I_ext = I_p
            homeostasis_on = False
            if r_memory is None:
                r_memory = r.copy()   # snapshot at transition
        else:
            # Phase 3: rescue
            I_ext = I_p
            homeostasis_on = True

        # Homeostatic current
        if homeostasis_on:
            h = homeostatic_update(h, r, m_star, kappa, tau_h, dt)
        I_h = homeostatic_current(h, N)

        # Total input
        I_total = I_ext + I_h

        # Rate update
        phi_r = phi(r)
        r = rate_update(r, W, I_total, tau_r, dt)
        r = np.maximum(r, 0.0)  # rates can't go negative (physical constraint)

        # Hebbian update (always running — driven to zero by phi(r) during suppression)
        W = hebbian_update(W, W_struct, phi_r, eta, lam, tau_W, dt)

        # Store
        t_arr[i]     = t
        r_arr[i]     = r
        W_arr[i]     = W
        h_arr[i]     = h
        mmean_arr[i] = np.mean(r)

    # Memory overlap (defined once r_memory is known)
    if r_memory is not None and np.linalg.norm(r_memory) > 1e-8:
        r_mem_norm = r_memory / np.linalg.norm(r_memory)
        for i in range(steps):
            mmem_arr[i] = np.dot(r_mem_norm, r_arr[i])

    return t_arr, r_arr, W_arr, h_arr, mmean_arr, mmem_arr, T1, T2, T3, r_memory


# ─────────────────────────────────────────────
# 5.  PLOTTING
# ─────────────────────────────────────────────

def plot_results(t, r, W, h, mmean, mmem, T1, T2, T3, r_memory):

    phase_boundaries = [T1, T1 + T2]
    phase_labels     = ["Write", "Suppress", "Rescue"]
    phase_centers    = [T1 / 2, T1 + T2 / 2, T1 + T2 + T3 / 2]
    colors           = ["#d4edda", "#f8d7da", "#d1ecf1"]  # green, red, blue

    fig = plt.figure(figsize=(12, 10))
    gs  = gridspec.GridSpec(4, 2, figure=fig, hspace=0.55, wspace=0.35)

    def shade_phases(ax):
        ylim = ax.get_ylim()
        starts = [0, T1, T1 + T2]
        ends   = [T1, T1 + T2, T1 + T2 + T3]
        for s, e, c in zip(starts, ends, colors):
            ax.axvspan(s, e, alpha=0.25, color=c, zorder=0)
        for pb in phase_boundaries:
            ax.axvline(pb, color="gray", lw=0.8, ls="--", zorder=1)
        ax.set_ylim(ylim)

    # ── (A) Individual firing rates ──────────────────────────────────────
    ax_r = fig.add_subplot(gs[0, :])
    ax_r.plot(t, r[:, 0], label="Neuron 1", color="#2c7bb6", lw=1.5)
    ax_r.plot(t, r[:, 1], label="Neuron 2", color="#d7191c", lw=1.5)
    ax_r.set_ylabel("Firing rate r(t)")
    ax_r.set_title("A — Individual neuron firing rates")
    ax_r.legend(loc="upper right", fontsize=9)
    shade_phases(ax_r)
    for xc, lbl in zip(phase_centers, phase_labels):
        ax_r.text(xc, ax_r.get_ylim()[1] * 0.92, lbl,
                  ha="center", fontsize=8, color="dimgray")

    # ── (B) Mean firing rate + homeostatic target ────────────────────────
    ax_m = fig.add_subplot(gs[1, :])
    ax_m.plot(t, mmean, color="purple", lw=1.5, label="Mean rate")
    ax_m.axhline(0.5, color="purple", lw=1, ls=":", alpha=0.6, label="m* (target)")
    ax_m.plot(t, h, color="orange", lw=1.2, ls="--", label="h(t) homeostatic drive")
    ax_m.set_ylabel("Rate / drive")
    ax_m.set_title("B — Mean firing rate and homeostatic drive h(t)")
    ax_m.legend(loc="upper right", fontsize=9)
    shade_phases(ax_m)

    # ── (C) Memory overlap ───────────────────────────────────────────────
    ax_mem = fig.add_subplot(gs[2, :])
    ax_mem.plot(t, mmem, color="#1a9641", lw=1.5)
    ax_mem.set_ylabel("mmem(t) = rm · r(t)")
    ax_mem.set_title("C — Memory overlap (does the pattern re-emerge?)")
    ax_mem.axhline(0, color="gray", lw=0.7, ls=":")
    shade_phases(ax_mem)

    # ── (D) Weight evolution: W[0,1] and W[1,0] (off-diagonal = cross-neuron) ──
    ax_w = fig.add_subplot(gs[3, 0])
    ax_w.plot(t, W[:, 0, 1], label="W[0,1]", color="#2c7bb6", lw=1.5)
    ax_w.plot(t, W[:, 1, 0], label="W[1,0]", color="#d7191c", lw=1.5)
    ax_w.axhline(0.1, color="gray", lw=0.8, ls=":", label="W_struct baseline")
    ax_w.set_ylabel("Weight")
    ax_w.set_xlabel("Time")
    ax_w.set_title("D — Off-diagonal weights\n(Hebbian cross-neuron strengthening)")
    ax_w.legend(fontsize=8)
    shade_phases(ax_w)

    # ── (E) Weight matrix heatmaps at end of each phase ─────────────────
    ax_wh = fig.add_subplot(gs[3, 1])
    phase_end_indices = [
        int(T1 / (t[1] - t[0])) - 1,
        int((T1 + T2) / (t[1] - t[0])) - 1,
        len(t) - 1,
    ]
    W_snapshots = np.stack([W[i] for i in phase_end_indices])  # (3, 2, 2)
    # Show as a 2x6 tiled image (3 matrices side by side)
    tile = np.concatenate([W_snapshots[k] for k in range(3)], axis=1)  # (2, 6)
    im = ax_wh.imshow(tile, cmap="RdBu", vmin=-0.5, vmax=1.0, aspect="auto")
    ax_wh.set_xticks([0.5, 2.5, 4.5])
    ax_wh.set_xticklabels(["After\nWrite", "After\nSuppress", "After\nRescue"],
                           fontsize=8)
    ax_wh.set_yticks([0, 1])
    ax_wh.set_yticklabels(["N1", "N2"])
    ax_wh.set_title("E — Weight matrix snapshots")
    plt.colorbar(im, ax=ax_wh, fraction=0.03, pad=0.04)

    fig.suptitle("Two-Neuron Sandbox: Write → Suppress → Homeostatic Rescue",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.savefig("/mnt/user-data/outputs/two_neuron_sandbox.png",
                dpi=150, bbox_inches="tight")
    plt.show()
    print("Figure saved.")


# ─────────────────────────────────────────────
# 6.  MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    results = run_two_neuron_sandbox()
    plot_results(*results)

    # ── Quick sanity checks ──────────────────────────────────────────────
    t, r, W, h, mmean, mmem, T1, T2, T3, r_memory = results
    dt = t[1] - t[0]

    end_write    = int(T1 / dt) - 1
    end_suppress = int((T1 + T2) / dt) - 1
    end_rescue   = len(t) - 1

    print("\n── Sanity checks ──────────────────────────────────────")
    print(f"  r at end of write:    {r[end_write]}")
    print(f"  r at end of suppress: {r[end_suppress]}")
    print(f"  r at end of rescue:   {r[end_rescue]}")
    print(f"  Mean rate at end of rescue: {mmean[end_rescue]:.3f}  (target: 0.5)")
    print(f"  Memory overlap at end of write:    {mmem[end_write]:.3f}")
    print(f"  Memory overlap at end of suppress: {mmem[end_suppress]:.3f}")
    print(f"  Memory overlap at end of rescue:   {mmem[end_rescue]:.3f}")
    print(f"  W[0,1] — after write: {W[end_write,0,1]:.3f} | "
          f"after suppress: {W[end_suppress,0,1]:.3f} | "
          f"after rescue: {W[end_rescue,0,1]:.3f}")
    print(f"  W_struct baseline: 0.100")
    print("───────────────────────────────────────────────────────")