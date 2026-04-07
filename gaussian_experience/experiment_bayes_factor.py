"""
Experiment 1: Exact Bayes factor comparison in the Gaussian conjugate model.

Compares the cross-validated score (our method) against the exact Bayes factor
(gold standard) in the conjugate Gaussian setting where both are available in
closed form. Produces three plots:
  - Plot A: Score difference vs sigma_x' for each alpha
  - Plot B: Convergence in alpha (relative error)
  - Plot C: Dimension scaling at fixed alpha
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

out_dir = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(out_dir, exist_ok=True)

# -----------------------------------------------------------------------------
# Matplotlib configuration to match the paper style
# -----------------------------------------------------------------------------
plt.rcParams.update(
    {
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{amsmath}\usepackage{amssymb}\usepackage{bm}",
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "font.size": 9,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "lines.linewidth": 1.2,
        "lines.markersize": 3,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.minor.width": 0.4,
        "ytick.minor.width": 0.4,
        "figure.dpi": 150,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
    }
)

# Column widths (inches) for ICML format
COL_WIDTH = 3.25  # single column
TEXT_WIDTH = 6.75  # full text width

# Fix random seed for reproducibility
rng = np.random.default_rng(42)

# =============================================================================
# Global parameters (Section 4.1 setup)
# =============================================================================
sigma2 = 1.0  # noise variance  sigma^2
sigmax2_true = 1.0  # true prior variance  sigma_x^2
K = 250  # number of MC fission noise realizations


# =============================================================================
# Closed-form expressions
# =============================================================================


def exact_log_bayes_factor(y_norm2, m, sigma2, sigmax2_true, sigmax2_prime):
    r"""Exact log-Bayes factor BF(sigma_x^2, sigma_x'^2).

    BF = (m/2) log((sigma^2 + sigma_x'^2) / (sigma^2 + sigma_x^2))
         + (||y||^2 / 2) (1/(sigma^2 + sigma_x'^2) - 1/(sigma^2 + sigma_x^2))
    """
    s2 = sigma2 + sigmax2_true
    sp2 = sigma2 + sigmax2_prime
    return (m / 2) * np.log(sp2 / s2) + (y_norm2 / 2) * (1 / sp2 - 1 / s2)


def log_posterior_predictive(y, w, sigma2, sigmax2, alpha):
    r"""log p(y+ | y-, sigma_x^2) for a single fission realization w.

    Uses the expanded quadratic form from Eq. (CV-single) of the plan:
      a^2 = (1-alpha)*sigma^2 / s^2
      2ab = 2*sqrt(alpha*(1-alpha))
      b^2 = alpha*s^2 / sigma^2
    """
    m = y.shape[0]
    s2 = sigma2 + sigmax2
    tau2 = alpha * sigmax2 + sigma2

    # Expanded quadratic: a^2||y||^2 + 2ab y^T w + b^2||w||^2
    a2 = (1 - alpha) * sigma2 / s2
    b2 = alpha * s2 / sigma2
    ab = np.sqrt(alpha * (1 - alpha))
    quad = a2 * np.sum(y**2) + 2 * ab * np.dot(y, w) + b2 * np.sum(w**2)

    log_score = (m / 2) * np.log((1 - alpha) * tau2 / ((2 * np.pi) ** 2 * sigma2 * s2))
    log_score -= quad / (2 * tau2)
    return log_score


def cv_score_diff_mc(y, ws, sigma2, sigmax2_true, sigmax2_prime, alpha):
    r"""MC estimate of \overline{CV}_alpha using pre-generated noise vectors ws.

    ws: (K, m) array of noise realizations, shared across all evaluations.
    Returns (mean, std_of_mean) of the CV score differences.
    """
    K = ws.shape[0]
    diffs = np.zeros(K)
    for k in range(K):
        score_true = log_posterior_predictive(y, ws[k], sigma2, sigmax2_true, alpha)
        score_prime = log_posterior_predictive(y, ws[k], sigma2, sigmax2_prime, alpha)
        diffs[k] = score_true - score_prime
    return np.mean(diffs), np.std(diffs) / np.sqrt(K)


def expected_cv_score(y_norm2, m, sigma2, sigmax2, alpha):
    r"""E_w[log p(y+ | y-, sigma_x^2)] analytically.

    From Eq. (expected_log_score) of the plan:
      E[log p] = (m/2) log((1-alpha)*tau^2 / ((2pi)^2 sigma^2 s^2))
                 - (1-alpha)*sigma^2*||y||^2 / (2*tau^2*s^2)
                 - alpha*s^2*m / (2*tau^2)
    """
    s2 = sigma2 + sigmax2
    tau2 = alpha * sigmax2 + sigma2
    return (
        (m / 2) * np.log((1 - alpha) * tau2 / ((2 * np.pi) ** 2 * sigma2 * s2))
        - (1 - alpha) * sigma2 * y_norm2 / (2 * tau2 * s2)
        - alpha * s2 * m / (2 * tau2)
    )


def expected_cv_diff(y_norm2, m, sigma2, sigmax2_true, sigmax2_prime, alpha):
    r"""E_w[CV_alpha(sigma_x^2, sigma_x'^2)] analytically."""
    return expected_cv_score(
        y_norm2, m, sigma2, sigmax2_true, alpha
    ) - expected_cv_score(y_norm2, m, sigma2, sigmax2_prime, alpha)


# =============================================================================
# Step 1: Generate data
# =============================================================================
dims = [10, 50, 100, 1000]
data = {}  # m -> (y, y_norm2, ws)
for m in dims:
    y = rng.normal(0, np.sqrt(sigma2 + sigmax2_true), size=m)
    # Pre-generate K noise realizations, reused across all alpha and sigma_x' values
    ws = rng.normal(0, np.sqrt(sigma2), size=(K, m))
    data[m] = (y, np.sum(y**2), ws)

# Grid of misspecified sigma_x' values
sigmax_prime_grid = np.linspace(0.3, 2.0, 80)
sigmax2_prime_grid = sigmax_prime_grid**2

# Alpha values to test
alphas = [0.05, 0.1, 0.25, 0.5, 0.75]

# Standard matplotlib tab10 colors (matching the paper's default palette)
tab10 = plt.cm.tab10.colors


# =============================================================================
# Plot A — Main result: log-score difference vs sigma_x'
#   Single-panel for m=1000 (main figure, like Fig. 2 of the paper)
#   + multi-panel for all dimensions
# =============================================================================
print("Computing Plot A...")

# --- Single-panel (m = 1000), matching Fig. 2 ---
m_main = 1000
y_main, y_norm2_main, ws_main = data[m_main]

fig, ax = plt.subplots(figsize=(COL_WIDTH, 2.4))

for ai, alpha in enumerate(alphas):
    # (ii) Analytical expected CV score difference (colored dashed lines)
    ecv = np.array(
        [
            expected_cv_diff(y_norm2_main, m_main, sigma2, sigmax2_true, sp2, alpha)
            for sp2 in sigmax2_prime_grid
        ]
    )
    ax.plot(
        sigmax_prime_grid,
        ecv,
        "--",
        color=tab10[ai],
        linewidth=1.0,
        label=(
            r"$\mathbb{{E}}_\mathbf{{w}}[\mathrm{{CV}}_{{\alpha}}]$, $\alpha = {}$".format(
                alpha
            )
            if ai == 0
            else r"$\alpha = {}$".format(alpha)
        ),
    )

    # (iii) MC estimates (colored markers with error bars)
    # Same K noise realizations ws_main used for all alpha and all sigma_x' values
    mc_idx = np.linspace(0, len(sigmax_prime_grid) - 1, 12, dtype=int)
    mc_means, mc_errs = [], []
    for ji in mc_idx:
        sp2 = sigmax2_prime_grid[ji]
        mean_val, std_val = cv_score_diff_mc(
            y_main, ws_main, sigma2, sigmax2_true, sp2, alpha
        )
        mc_means.append(mean_val)
        mc_errs.append(1.96 * std_val)
    ax.errorbar(
        sigmax_prime_grid[mc_idx],
        mc_means,
        yerr=mc_errs,
        fmt="o",
        color=tab10[ai],
        markersize=2.5,
        capsize=1.5,
        linewidth=0.6,
        zorder=5,
        label=(
            r"$\overline{{\mathrm{{CV}}}}_\alpha$, $\alpha = {}$".format(alpha)
            if ai == 0
            else None
        ),
    )

# (i) Exact Bayes factor (black solid line)
bf = np.array(
    [
        exact_log_bayes_factor(y_norm2_main, m_main, sigma2, sigmax2_true, sp2)
        for sp2 in sigmax2_prime_grid
    ]
)
ax.plot(sigmax_prime_grid, bf, "k-", linewidth=1.8, label=r"Exact BF")

ax.axvline(
    x=np.sqrt(sigmax2_true), color="grey", linestyle=":", linewidth=0.5, alpha=0.6
)
ax.axhline(y=0, color="grey", linewidth=0.4, alpha=0.4)
ax.set_xlabel(r"$\sigma'_x$")
ax.set_ylabel(
    r"$\log \frac{p(\mathbf{y}^+|\mathbf{y}^-,\sigma_x^2)}"
    r"{p(\mathbf{y}^+|\mathbf{y}^-,\sigma_x^{\prime 2})}$, BF"
)
ax.legend(frameon=False, ncol=2, columnspacing=1.0)

fig.savefig(os.path.join(out_dir, "plot_A_score_vs_sigmax.pdf"))
fig.savefig(os.path.join(out_dir, "plot_A_score_vs_sigmax.png"))
plt.close(fig)
print("  -> Saved plot_A_score_vs_sigmax.pdf/png")


# --- Multi-panel (all dimensions) ---
fig, axes = plt.subplots(1, 4, figsize=(TEXT_WIDTH, 2.0), sharey=False)

for idx, m in enumerate(dims):
    ax = axes[idx]
    y, y_norm2, ws = data[m]

    for ai, alpha in enumerate(alphas):
        # Analytical expected CV (dashed)
        ecv = np.array(
            [
                expected_cv_diff(y_norm2, m, sigma2, sigmax2_true, sp2, alpha)
                for sp2 in sigmax2_prime_grid
            ]
        )
        ax.plot(
            sigmax_prime_grid,
            ecv,
            "--",
            color=tab10[ai],
            linewidth=1.0,
            label=r"$\alpha = {}$".format(alpha) if idx == 0 else None,
        )

        # MC estimates (markers)
        mc_idx = np.linspace(0, len(sigmax_prime_grid) - 1, 10, dtype=int)
        mc_means, mc_errs = [], []
        for ji in mc_idx:
            sp2 = sigmax2_prime_grid[ji]
            mean_val, std_val = cv_score_diff_mc(
                y, ws, sigma2, sigmax2_true, sp2, alpha
            )
            mc_means.append(mean_val)
            mc_errs.append(1.96 * std_val)
        ax.errorbar(
            sigmax_prime_grid[mc_idx],
            mc_means,
            yerr=mc_errs,
            fmt="o",
            color=tab10[ai],
            markersize=1.8,
            capsize=1,
            linewidth=0.5,
        )

    bf = np.array(
        [
            exact_log_bayes_factor(y_norm2, m, sigma2, sigmax2_true, sp2)
            for sp2 in sigmax2_prime_grid
        ]
    )
    ax.plot(
        sigmax_prime_grid,
        bf,
        "k-",
        linewidth=1.5,
        label=r"Exact BF" if idx == 0 else None,
    )

    ax.axvline(
        x=np.sqrt(sigmax2_true), color="grey", linestyle=":", linewidth=0.5, alpha=0.6
    )
    ax.axhline(y=0, color="grey", linewidth=0.4, alpha=0.4)
    ax.set_xlabel(r"$\sigma'_x$")
    if idx == 0:
        ax.set_ylabel(
            r"$\log \frac{p(\mathbf{y}^+|\mathbf{y}^-,\sigma_x^2)}"
            r"{p(\mathbf{y}^+|\mathbf{y}^-,\sigma_x^{\prime 2})}$"
        )
    ax.set_title(r"$m = {}$".format(m))

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc="lower center",
    ncol=6,
    frameon=False,
    bbox_to_anchor=(0.5, -0.22),
)
fig.subplots_adjust(wspace=0.35)

fig.savefig(os.path.join(out_dir, "plot_A_all_dims.pdf"))
fig.savefig(os.path.join(out_dir, "plot_A_all_dims.png"))
plt.close(fig)
print("  -> Saved plot_A_all_dims.pdf/png")


# =============================================================================
# Plot B — Convergence in alpha: relative error vs alpha  (cf. Fig. 14)
# =============================================================================
print("Computing Plot B...")

alpha_fine = np.logspace(-2, np.log10(0.9), 50)
sigmax_prime_choices = [0.5, 0.7, 1.5, 2.0]
markers_B = ["o", "s", "^", "D"]

fig, axes = plt.subplots(1, 4, figsize=(TEXT_WIDTH, 2.0), sharey=True)

for idx, m in enumerate(dims):
    ax = axes[idx]
    y, y_norm2, ws = data[m]

    for si, sxp in enumerate(sigmax_prime_choices):
        sxp2 = sxp**2
        bf_val = exact_log_bayes_factor(y_norm2, m, sigma2, sigmax2_true, sxp2)
        if np.abs(bf_val) < 1e-12:
            continue

        rel_errors = np.array(
            [
                np.abs(
                    expected_cv_diff(y_norm2, m, sigma2, sigmax2_true, sxp2, a) - bf_val
                )
                / np.abs(bf_val)
                for a in alpha_fine
            ]
        )
        ax.plot(
            alpha_fine,
            rel_errors,
            color=tab10[si],
            marker=markers_B[si],
            markevery=5,
            markersize=2.5,
            label=r"$\sigma'_x = {}$".format(sxp) if idx == 0 else None,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\alpha$")
    if idx == 0:
        ax.set_ylabel(
            r"$|\mathbb{E}[\mathrm{CV}_\alpha] - \mathrm{BF}|\,/\,|\mathrm{BF}|$"
        )
    ax.set_title(r"$m = {}$".format(m))

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc="lower center",
    ncol=4,
    frameon=False,
    bbox_to_anchor=(0.5, -0.22),
)
fig.subplots_adjust(wspace=0.12)

fig.savefig(os.path.join(out_dir, "plot_B_convergence_alpha.pdf"))
fig.savefig(os.path.join(out_dir, "plot_B_convergence_alpha.png"))
plt.close(fig)
print("  -> Saved plot_B_convergence_alpha.pdf/png")


# =============================================================================
# Plot C — Dimension scaling: relative error vs m at fixed alpha  (cf. Fig. 13)
# =============================================================================
print("Computing Plot C...")

dims_fine = [5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000]
alpha_fixed = 0.25

fig, ax = plt.subplots(figsize=(COL_WIDTH, 2.4))

for si, sxp in enumerate(sigmax_prime_choices):
    sxp2 = sxp**2
    rel_errors = []
    for m in dims_fine:
        y_loc = rng.normal(0, np.sqrt(sigma2 + sigmax2_true), size=m)
        y_norm2_loc = np.sum(y_loc**2)
        bf_val = exact_log_bayes_factor(y_norm2_loc, m, sigma2, sigmax2_true, sxp2)
        ecv_val = expected_cv_diff(
            y_norm2_loc, m, sigma2, sigmax2_true, sxp2, alpha_fixed
        )
        if np.abs(bf_val) < 1e-12:
            rel_errors.append(np.nan)
        else:
            rel_errors.append(np.abs(ecv_val - bf_val) / np.abs(bf_val))

    ax.plot(
        dims_fine,
        rel_errors,
        color=tab10[si],
        marker=markers_B[si],
        markersize=3,
        label=r"$\sigma'_x = {}$".format(sxp),
    )

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"$m$")
ax.set_ylabel(r"$|\mathbb{E}[\mathrm{CV}_\alpha] - \mathrm{BF}|\,/\,|\mathrm{BF}|$")
ax.legend(frameon=False)

fig.savefig(os.path.join(out_dir, "plot_C_dimension_scaling.pdf"))
fig.savefig(os.path.join(out_dir, "plot_C_dimension_scaling.png"))
plt.close(fig)
print("  -> Saved plot_C_dimension_scaling.pdf/png")

print("\nAll plots saved to:", out_dir)
