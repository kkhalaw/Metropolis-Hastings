import os
import numpy as np
import scipy
import tqdm
import numpy.random
import h5py
from matplotlib import pyplot as plt
import emcee
import arviz
import corner

# -----Minimization of Cubic Polynomial from Previous Data Set-----

here = os.path.dirname(os.path.abspath(__file__))
f = h5py.File(os.path.join(here, "data.hdf"), "r")

print(f.keys())

print(f["data"].keys())

xpos = f["data/xpos"][:]
ypos = f["data/ypos"][:]


# sort by x, in case the data is out of order :P
# this prevents the graphs from looking "stringy"
order = np.argsort(xpos)
xpos, ypos = xpos[order], ypos[order]

# standardize x to be used in model:
X_MEAN = xpos.mean()
X_STD = xpos.std()


# creates an array of parameters
def cubic(values, x):
    a, b, c, d = values
    x_coeff = (x - X_MEAN) / X_STD

    return a * x_coeff**3 + b * x_coeff**2 + c * x_coeff + d


# ssr = sum of squared residuals:
# how far is prediction from actual data?
def ssr(params, x, y, model_fn=cubic):
    residuals = y - model_fn(params, x)
    return np.sum(residuals**2)


# Starting guesses:
p0 = np.polyfit((xpos - X_MEAN) / X_STD, ypos, 3)

# Minimizer: Newton's Method


def nelder_mead(f, p0, scale=None, max_runs=5000, tol=1e-10):
    p0 = np.array(p0, dtype=float)
    n = len(p0)

    # per-parameter nudge: 10% of each parameter's size
    if scale is None:
        scale = 0.1 * np.maximum(np.abs(p0), 1e-2)

    # simplex has n+1 corners
    simplex = np.vstack([p0] + [p0 + np.eye(n)[i] * scale[i] for i in range(n)])
    history = []

    for i in range(1, max_runs + 1):
        vals = np.array([f(v) for v in simplex])
        order = np.argsort(vals)
        simplex, vals = simplex[order], vals[order]

        best = simplex[0]
        history.append((best.copy(), vals[0]))

        # converged when the corners have bunched up AND their heights agree
        size = np.max(np.abs(simplex[1:] - best))
        spread = abs(vals[-1] - vals[0])
        if size < tol * (1 + np.linalg.norm(best)) and spread < tol * (
            1 + abs(vals[0])
        ):
            return best, vals[0], history, i

        centroid = simplex[:-1].mean(axis=0)
        worst = simplex[-1]

        refl = centroid + (centroid - worst)
        f_refl = f(refl)

        if vals[0] <= f_refl < vals[-2]:  # second-worst, not [1], once n > 2
            simplex[-1] = refl
        elif f_refl < vals[0]:
            expand = centroid + 2 * (centroid - worst)
            simplex[-1] = expand if f(expand) < f_refl else refl
        else:
            contract = centroid + 0.5 * (worst - centroid)
            if f(contract) < vals[-1]:
                simplex[-1] = contract
            else:
                simplex[1:] = best + 0.5 * (simplex[1:] - best)

    return best, vals[0], history, None


# combine into one_argument
cubic_obj = lambda p: ssr(p, xpos, ypos)


# -------------Sample from 1D Distribution----------------


def post(x):
    return 1 / (2 * np.pi) ** 0.5 * np.exp(-1 / 2 * x**2.0) * (x**3)


def proposal2(x):
    return np.random.normal(x, 1)


def mcmc(initial, post, prop, iterations):
    x = [initial]
    p = [post(x[-1])]
    for i in tqdm.tqdm(range(iterations)):
        x_test = prop(x[-1])
        p_test = post(x_test)

        acc = p_test / p[-1]
        u = np.random.uniform(0, 1)
        if u <= acc:
            x.append(x_test)
            p.append(p_test)
        else:
            # a rejected proposal still gives a sample, you stay put
            x.append(x[-1])
            p.append(p[-1])
    return x, p


def acl(x, b):
    # batch means: if b is much longer than the acl the batch means are
    # independent, so their variance tells you the acl
    m = len(x) // b
    means = x[: m * b].reshape(m, b).mean(axis=1)
    return b * means.var(ddof=1) / x.var(ddof=1)


chain, prob = mcmc(10, post, proposal2, 1000000)
chain = np.array(chain)
samples = chain[1000:]

# exact values
e8 = np.exp(-8)
exact_mean = 1.5 * np.sqrt(np.pi / 2)
exact_var = 4 - 9 * np.pi / 8  # 0.4657

samp_mean = samples.mean()
samp_var = samples.var()

# ACL yields error bars
tau = acl(samples, 2000)
n_eff = len(samples) / tau
err_mean = np.sqrt(samp_var / n_eff)
err_var = (
    acl((samples - samp_mean) ** 2, 2000) ** 0.5
    * np.std((samples - samp_mean) ** 2)
    / np.sqrt(len(samples))
)

print(f"mean: sample = {samp_mean:.4f} ± {err_mean:.4f}   exact = {exact_mean:.4f}")
print(f"var : sample = {samp_var:.4f} ± {err_var:.4f}   exact = {exact_var:.4f}")

# raise b until the answer stops growing, that plateau is the acl
for b in [20, 100, 500, 2000, 10000]:
    print("b =", b, " acl =", round(acl(samples, b), 1))
print("independent samples:", round(len(samples) / acl(samples, 2000)))

# Vary proposal width
widths = [0.01, 0.1, 1, 3, 30]
fig, axes = plt.subplots(len(widths), 1, sharex=True, figsize=(8, 10))
for ax, w in zip(
    axes, widths
):  # zip (): iterator of tuples where the first item in each passed iterator is paired together, and then the second item in each passed iterator are paired together
    ch = np.array(mcmc(10, post, lambda x: np.random.normal(x, w), 100000)[0])
    rate = np.mean(np.diff(ch) != 0)
    ax.plot(ch[:5000])
    ax.set_title(f"width = {w}, acceptance = {rate:.2f}")
plt.tight_layout()

for cut in [0, 1000, 30, 1]:  # last parameter = where burn-in seems to end
    s = chain[cut:]
    print(f"burn-in {cut}: mean = {s.mean():.4f}, var = {s.var():.4f}")


# ----Likelihood for fitted data-----


# log due to future sampling of many data points!
def log_likelihood(theta, x, y):
    a, b, c, d, log_sigma = theta
    sigma = np.exp(log_sigma)
    resid = y - cubic((a, b, c, d), x)
    n = len(y)
    return (
        -0.5 * np.sum(resid**2) / sigma**2 - n * log_sigma - 0.5 * n * np.log(2 * np.pi)
    )


Y_SCALE = ypos.std()


def log_prior(theta):
    a, b, c, d, log_sigma = theta
    if np.all(np.abs([a, b, c, d]) < 100 * Y_SCALE) and -10 < log_sigma < 10:
        return 0.0  # log(constant), the constant doesn't matter
    return -np.inf  # zero probability outside the box


def log_posterior(theta, x, y):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, x, y)


# start walkers near the nelder-mead best fit.
best, ssr_best, _, _ = nelder_mead(cubic_obj, p0)
sigma0 = np.sqrt(ssr_best / len(ypos))  # max-likelihood guess for sigma
start = np.append(best, np.log(sigma0))

ndim, nwalkers = 5, 32
pos = start + 1e-3 * np.random.randn(nwalkers, ndim)

sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, args=(xpos, ypos))
sampler.run_mcmc(pos, 5000, progress=True)

print("autocorrelation time:", sampler.get_autocorr_time(quiet=True))
flat = sampler.get_chain(discard=1000, thin=10, flat=True)

corner.corner(flat, labels=["a", "b", "c", "d", r"$\ln\sigma$"])


# ---CONVERGENCE TEST---
# 5. do all walkers spend the same fraction of time within X of the peak
# the one-bin version of test 3. expected scatter is binomial, using the
# effective sample size per walker


plt.figure()
plt.title("Evolution of the walker")
plt.plot(chain)
plt.ylabel("x-value")
plt.xlabel("Iteration")

plt.figure()
plt.title("Evolution of the walker")
plt.plot(chain)
plt.xlim(0, 500)
plt.ylabel("x-value")
plt.xlabel("Iteration")
plt.figure()
plt.title("Posterior samples")
_ = plt.hist(samples[::100], bins=100)


plt.show()
