"""
MCMC fit of the cubic polynomial model to data.hdf.

(1) Gaussian-noise likelihood + priors
(2) ensemble sampler with many walkers from random starts, trace plots, burn-in
(3) corner plot after burn-in, compared to the Newton best fit
(4) convergence test #5: do all walkers spend the same fraction of time
    within X of the peak? (binomial scatter using n_eff per walker)

Model, SSR and the Newton minimizer are the same as in cubic_bestfit.py.
The sampler is an affine-invariant "stretch move" ensemble sampler
(Goodman & Weare 2010, the algorithm behind emcee), written out by hand.
"""

import os
import h5py
import numpy as np
from matplotlib import pyplot as plt

here = os.path.dirname(os.path.abspath(__file__))
with h5py.File(os.path.join(here, "data.hdf"), "r") as f:
    xpos = f["data/xpos"][:]
    ypos = f["data/ypos"][:]

order = np.argsort(xpos)
xpos, ypos = xpos[order], ypos[order]
N = len(xpos)

X_MEAN = xpos.mean()
X_STD = xpos.std()

rng = np.random.default_rng(42)


# ---------------------------------------------------------------------------
# Model + Newton minimizer (from cubic_bestfit.py)
# ---------------------------------------------------------------------------
def cubic(values, x):
    a, b, c, d = values
    x_coeff = (x - X_MEAN) / X_STD
    return a * x_coeff**3 + b * x_coeff**2 + c * x_coeff + d


def ssr(params, x, y, model_fn=cubic):
    residuals = y - model_fn(params, x)
    return np.sum(residuals**2)


def num_grad(f, p, h=1e-6):
    g = np.zeros(len(p))
    for i in range(len(p)):
        hi = h * max(1.0, abs(p[i]))
        e = np.zeros(len(p))
        e[i] = hi
        g[i] = (f(p + e) - f(p - e)) / (2 * hi)
    return g


def num_hess(f, p, h=1e-4):
    n = len(p)
    H = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            hi = h * max(1.0, abs(p[i]))
            hj = h * max(1.0, abs(p[j]))
            ei = np.zeros(n)
            ei[i] = hi
            ej = np.zeros(n)
            ej[j] = hj
            H[i, j] = H[j, i] = (
                f(p + ei + ej) - f(p + ei - ej) - f(p - ei + ej) + f(p - ei - ej)
            ) / (4 * hi * hj)
    return H


def newton(f, p0, max_runs=200, tol=1e-8):
    p = np.array(p0, dtype=float)
    val = f(p)
    for i in range(1, max_runs + 1):
        g = num_grad(f, p)
        H = num_hess(f, p)
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            return p, val, None
        if g @ step <= 0:
            step = g / (np.linalg.norm(g) + 1e-12)
        t = 1.0
        while t > 1e-10:
            p_new = p - t * step
            v_new = f(p_new)
            if np.isfinite(v_new) and v_new < val:
                break
            t /= 2
        else:
            return p, val, i
        moved = np.linalg.norm(p_new - p)
        p, val = p_new, v_new
        if moved < tol * (1 + np.linalg.norm(p)):
            return p, val, i
    return p, val, None


p0 = np.polyfit((xpos - X_MEAN) / X_STD, ypos, 3)
newton_best, newton_ssr, newton_iters = newton(lambda p: ssr(p, xpos, ypos), p0)
# maximum-likelihood noise level for Gaussian residuals
sigma_ml = np.sqrt(newton_ssr / N)
best_full = np.append(newton_best, np.log(sigma_ml))
print(f"Newton best fit: a,b,c,d = {newton_best}, SSR = {newton_ssr:.2f}")
print(f"ML noise sigma = {sigma_ml:.4f}")


# ---------------------------------------------------------------------------
# (1) Likelihood and priors
# ---------------------------------------------------------------------------
# theta = (a, b, c, d, ln_sigma)
# The noise level is unknown, so it is fit too (as ln sigma so it stays > 0).
#
# Gaussian likelihood, y_i = f(x_i; a,b,c,d) + eps_i,  eps_i ~ N(0, sigma^2):
#   ln L = -1/2 * sum[(y_i - f(x_i))^2 / sigma^2] - N ln(sigma) - N/2 ln(2 pi)
#
# Priors (my choice):
#   a, b, c, d : flat on [-100, 100]  (data |y| is a few tens at most, and x is
#                standardized, so this is very wide but proper)
#   ln sigma   : flat on [-10, 5]     (Jeffreys / scale-invariant prior on sigma)
LABELS = ["a", "b", "c", "d", r"$\ln\sigma$"]
NDIM = 5
PRIOR_LO = np.array([-100, -100, -100, -100, -10.0])
PRIOR_HI = np.array([100, 100, 100, 100, 5.0])


def log_prior(theta):
    if np.all(theta > PRIOR_LO) and np.all(theta < PRIOR_HI):
        return 0.0
    return -np.inf


def log_likelihood(theta):
    params, ln_sigma = theta[:4], theta[4]
    sigma2 = np.exp(2 * ln_sigma)
    return (
        -0.5 * ssr(params, xpos, ypos) / sigma2
        - N * ln_sigma
        - 0.5 * N * np.log(2 * np.pi)
    )


def log_posterior(theta):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta)


# ---------------------------------------------------------------------------
# (2) Ensemble sampler, many walkers, random initial positions
# ---------------------------------------------------------------------------
def stretch_move_sampler(log_prob, initial, n_steps, a=2.0):
    """Affine-invariant ensemble sampler (Goodman & Weare stretch move).

    Walkers are split into two halves; each walker in one half proposes a
    move along the line to a random walker in the other half:
        Y = X_j + z (X_k - X_j),   z ~ g(z) ∝ 1/sqrt(z) on [1/a, a]
    accepted with probability min(1, z^(ndim-1) p(Y)/p(X_k)).
    """
    n_walkers, ndim = initial.shape
    pos = initial.copy()
    lp = np.array([log_prob(p) for p in pos])

    chain = np.empty((n_steps, n_walkers, ndim))
    lnp = np.empty((n_steps, n_walkers))
    n_accept = np.zeros(n_walkers)
    half = n_walkers // 2
    halves = [np.arange(half), np.arange(half, n_walkers)]

    for step in range(n_steps):
        for s in range(2):
            active, other = halves[s], halves[1 - s]
            # draw z from g(z) via inverse CDF
            z = ((a - 1.0) * rng.random(len(active)) + 1.0) ** 2 / a
            partners = pos[rng.choice(other, size=len(active))]
            proposal = partners + z[:, None] * (pos[active] - partners)
            lp_new = np.array([log_prob(p) for p in proposal])
            log_ratio = (ndim - 1) * np.log(z) + lp_new - lp[active]
            accept = np.log(rng.random(len(active))) < log_ratio
            idx = active[accept]
            pos[idx] = proposal[accept]
            lp[idx] = lp_new[accept]
            n_accept[idx] += 1
        chain[step] = pos
        lnp[step] = lp
        if (step + 1) % 1000 == 0:
            print(f"  step {step + 1}/{n_steps}")

    return chain, lnp, n_accept / n_steps


N_WALKERS = 200
N_STEPS = 5000

# random starts spread over a wide box, nowhere near tuned to the best fit
init = np.column_stack(
    [
        rng.uniform(-30, 30, N_WALKERS),
        rng.uniform(-30, 30, N_WALKERS),
        rng.uniform(-30, 30, N_WALKERS),
        rng.uniform(-30, 30, N_WALKERS),
        rng.uniform(-2, 3, N_WALKERS),
    ]
)

print(f"Running {N_WALKERS} walkers for {N_STEPS} steps...")
chain, lnp, acc_frac = stretch_move_sampler(log_posterior, init, N_STEPS)
print(f"mean acceptance fraction = {acc_frac.mean():.3f}")

# Trace plot per parameter (+ log posterior)
fig, axes = plt.subplots(NDIM + 1, 1, sharex=True, figsize=(10, 12))
for j in range(NDIM):
    axes[j].plot(chain[:, :, j], color="k", alpha=0.05, lw=0.5)
    axes[j].axhline(best_full[j], color="r", lw=1)
    axes[j].set_ylabel(LABELS[j])
axes[-1].plot(lnp, color="k", alpha=0.05, lw=0.5)
axes[-1].set_ylabel("ln posterior")
axes[-1].set_yscale("symlog")
axes[-1].set_xlabel("step")
axes[0].set_title("Trace plots, all walkers (red = Newton best fit)")
fig.tight_layout()
fig.savefig(os.path.join(here, "cubic_mcmc_trace.png"), dpi=120)

# Zoom on the first steps to choose the burn-in by eye
fig, axes = plt.subplots(NDIM, 1, sharex=True, figsize=(10, 10))
for j in range(NDIM):
    axes[j].plot(chain[:1500, :, j], color="k", alpha=0.05, lw=0.5)
    axes[j].axhline(best_full[j], color="r", lw=1)
    axes[j].set_ylabel(LABELS[j])
axes[-1].set_xlabel("step")
axes[0].set_title("Trace plots, first 1500 steps (zoom to pick burn-in)")
fig.tight_layout()
fig.savefig(os.path.join(here, "cubic_mcmc_trace_zoom.png"), dpi=120)

# BURN-IN, chosen by eye from the zoomed trace plot: the walkers start out
# spread over +/-30 (ln sigma is the slowest) and the bulk of them have
# collapsed onto the red lines and stopped drifting by about step 200.
# 1000 is used to be safe; there are 5000 steps, so it costs little.
# The few flat lines that never come in are stuck walkers (handled below).
BURN_IN = 1000

# Stuck walkers: ensemble samplers sometimes leave a few walkers stranded in a
# low-probability region far from the peak (they show up as flat lines away
# from the red line in the trace plot). The convergence test in (4) flags them;
# here they are identified by their mean ln posterior after burn-in being far
# below the ensemble's, and are removed before the corner plot.
mean_lnp = lnp[BURN_IN:].mean(axis=0)
good = mean_lnp > np.median(mean_lnp) - 10 * NDIM
print(f"burn-in = {BURN_IN} steps; stuck walkers removed: {np.where(~good)[0]}")

all_after_burn = chain[BURN_IN:]  # (steps, walkers, ndim), every walker
samples_by_walker = all_after_burn[:, good]
flat = samples_by_walker.reshape(-1, NDIM)
print(f"{good.sum()} walkers, {len(flat)} samples kept")


# ---------------------------------------------------------------------------
# (3) Corner plot after burn-in
# ---------------------------------------------------------------------------
def corner_plot(samples, labels, truths=None, bins=40):
    nd = samples.shape[1]
    fig, axes = plt.subplots(nd, nd, figsize=(2.2 * nd, 2.2 * nd))
    for i in range(nd):
        for j in range(nd):
            ax = axes[i, j]
            if j > i:
                ax.axis("off")
                continue
            if i == j:
                ax.hist(samples[:, i], bins=bins, histtype="step", color="k")
                q = np.percentile(samples[:, i], [16, 50, 84])
                for v in q:
                    ax.axvline(v, color="k", ls="--", lw=0.7)
                ax.set_title(
                    f"{labels[i]} = {q[1]:.3f}$^{{+{q[2] - q[1]:.3f}}}"
                    f"_{{-{q[1] - q[0]:.3f}}}$",
                    fontsize=8,
                )
                ax.set_yticks([])
                if truths is not None:
                    ax.axvline(truths[i], color="r")
            else:
                H, xe, ye = np.histogram2d(samples[:, j], samples[:, i], bins=bins)
                # contour levels enclosing 39.3% / 86.5% (1 and 2 sigma in 2D)
                Hs = np.sort(H.ravel())[::-1]
                cum = np.cumsum(Hs) / Hs.sum()
                levels = sorted({Hs[np.searchsorted(cum, c)] for c in (0.865, 0.393)})
                ax.hist2d(samples[:, j], samples[:, i], bins=bins, cmap="Greys")
                xc, yc = 0.5 * (xe[1:] + xe[:-1]), 0.5 * (ye[1:] + ye[:-1])
                if len(levels) > 1:
                    ax.contour(xc, yc, H.T, levels=levels, colors="k", linewidths=0.8)
                if truths is not None:
                    ax.axvline(truths[j], color="r", lw=0.8)
                    ax.axhline(truths[i], color="r", lw=0.8)
                    ax.plot(truths[j], truths[i], "rs", ms=3)
            if i == nd - 1:
                ax.set_xlabel(labels[j])
            else:
                ax.set_xticklabels([])
            if j == 0 and i > 0:
                ax.set_ylabel(labels[i])
            elif j > 0:
                ax.set_yticklabels([])
            ax.tick_params(labelsize=6)
    fig.suptitle("Posterior after burn-in (red = Newton best fit)")
    fig.tight_layout()
    return fig


fig = corner_plot(flat, LABELS, truths=best_full)
fig.savefig(os.path.join(here, "cubic_mcmc_corner.png"), dpi=120)

# Does the peak sit where the minimizer put the best fit?
print("\nPosterior vs. Newton best fit")
print(f"{'param':>8s}{'Newton':>12s}{'median':>12s}{'16-84% width':>15s}{'offset/sigma':>14s}")
for j in range(NDIM):
    lo, med, hi = np.percentile(flat[:, j], [16, 50, 84])
    width = 0.5 * (hi - lo)
    name = "ln_sigma" if j == 4 else LABELS[j]
    print(
        f"{name:>8s}{best_full[j]:12.4f}{med:12.4f}{width:15.4f}"
        f"{(med - best_full[j]) / width:14.2f}"
    )
map_idx = np.unravel_index(np.argmax(lnp[BURN_IN:]), lnp[BURN_IN:].shape)
print("highest-posterior sample:", all_after_burn[map_idx])

# Are any parameters correlated?
corr = np.corrcoef(flat.T)
print("\nCorrelation matrix (a, b, c, d, ln_sigma):")
print(np.array2string(corr, precision=2, suppress_small=True))
print("Strongly correlated pairs (|r| > 0.5):")
for i in range(NDIM):
    for j in range(i + 1, NDIM):
        if abs(corr[i, j]) > 0.5:
            print(f"  {LABELS[i]} - {LABELS[j]}: r = {corr[i, j]:+.2f}")


# ---------------------------------------------------------------------------
# (4) Convergence test #5: do all walkers spend the same fraction of time
#     within X of the peak?
# ---------------------------------------------------------------------------
# For each parameter, "within X of the peak" = within 1 posterior std of the
# posterior median. If every walker samples the same distribution, each walker's
# fraction f_w is a binomial estimate of the same p, with n_eff = n_steps / tau
# effectively independent draws, so
#     var(f_w) ≈ p (1 - p) / n_eff.
# We compare the observed scatter of f_w across walkers with that expectation
# (ratio ~ 1 means converged, >> 1 means some walkers are stuck elsewhere).


def acl_batch_means(x, b):
    # batch-means autocorrelation length (same idea as professor_example_2.py)
    m = len(x) // b
    means = x[: m * b].reshape(m, b).mean(axis=1)
    return b * means.var(ddof=1) / x.var(ddof=1)


def time_near_peak_test(walkers, title, fname):
    """walkers: (steps, n_walkers, ndim). Prints the table, saves a figure."""
    n_steps, n_w, _ = walkers.shape
    print(f"\nConvergence test ({title}): fraction of time within 1 std of peak")
    print(f"{'param':>8s}{'p':>8s}{'tau':>8s}{'n_eff/walker':>14s}"
          f"{'obs std':>10s}{'binom std':>11s}{'ratio':>8s}{'chi2/dof':>10s}")
    fig, axes = plt.subplots(1, NDIM, figsize=(16, 3), sharey=True)
    for j in range(NDIM):
        x = walkers[:, :, j]
        # peak and width from the pooled samples (median / 16-84% half width,
        # robust to a few outlying walkers)
        lo, centre, hi = np.percentile(x, [16, 50, 84])
        width = 0.5 * (hi - lo)
        inside = np.abs(x - centre) < width  # (steps, walkers) booleans
        f_w = inside.mean(axis=0)  # fraction per walker
        p = inside.mean()

        tau = np.mean([acl_batch_means(x[:, w], 200) for w in range(n_w)])
        tau = max(tau, 1.0)
        n_eff = n_steps / tau
        expected_std = np.sqrt(p * (1 - p) / n_eff)
        obs_std = f_w.std(ddof=1)
        chi2_dof = np.sum((f_w - p) ** 2 / expected_std**2) / (n_w - 1)

        name = "ln_sigma" if j == 4 else LABELS[j]
        print(f"{name:>8s}{p:8.3f}{tau:8.1f}{n_eff:14.1f}{obs_std:10.4f}"
              f"{expected_std:11.4f}{obs_std / expected_std:8.2f}{chi2_dof:10.2f}")

        ax = axes[j]
        ax.hist(f_w, bins=25, density=True, color="0.7", label="walkers")
        grid = np.linspace(min(f_w.min(), p - 4 * expected_std),
                           max(f_w.max(), p + 4 * expected_std), 300)
        ax.plot(grid, np.exp(-0.5 * ((grid - p) / expected_std) ** 2)
                / (np.sqrt(2 * np.pi) * expected_std), "r", label="binomial")
        ax.set_title(LABELS[j])
        ax.set_xlabel("fraction within 1 std of peak")
    axes[0].legend(fontsize=7)
    fig.suptitle(f"Convergence test, {title}: per-walker time near peak vs. binomial")
    fig.tight_layout()
    fig.savefig(os.path.join(here, fname), dpi=120)


time_near_peak_test(all_after_burn, "all walkers", "cubic_mcmc_convergence_all.png")
time_near_peak_test(samples_by_walker, "stuck walkers removed",
                    "cubic_mcmc_convergence.png")

print("\nratio ≈ 1 (chi2/dof ≈ 1): walkers agree to within binomial noise -> converged.")
print("ratio >> 1: some walkers spend very different time near the peak -> not converged.")

# Posterior predictive: a handful of fits drawn from the posterior
plt.figure(figsize=(8, 4))
plt.scatter(xpos, ypos, s=6, color="0.5", label="data")
for k in rng.choice(len(flat), 100, replace=False):
    plt.plot(xpos, cubic(flat[k, :4], xpos), color="C0", alpha=0.05)
plt.plot(xpos, cubic(newton_best, xpos), "r", label="Newton best fit")
plt.title("Posterior draws of the cubic model")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(here, "cubic_mcmc_draws.png"), dpi=120)

plt.show()
