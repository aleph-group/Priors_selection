"""Compare Phi_1 and Phi_2 statistics for WAV, DnCNN, and DRUNet models.

Replicates the EXACT experimental setup from the ProxNest notebook:
  papers/MaxEnt/notebooks/WAV_DnCNN_galaxy_denoising.ipynb

Setup:
  - Galaxy 64x64, ISNR=15 dB, identity forward model
  - sigma = sqrt(mean(|ground_truth|^2)) * 10^(-15/20)
  - np.random.seed(0)
  - WAV: db6 wavelets, level 2, reg_param=2e4
  - DnCNN: TensorFlow SavedModel (snr_15_model.pb), trained at SNR 15
  - DRUNet: deepinv pretrained DRUNet (grayscale)

ProxNest parameters (from the notebook):
  DnCNN: delta=1e-7, lamb=5e-7, gamma=0.005*sigma^2
  DRUNet: delta=1e-7, lamb=5e-7, gamma=0.005*sigma^2
  WAV:   delta=1e-7, lamb=5e-7, gamma=5e-7

Phi statistics are computed using SKROCK posterior sampling via DegradedLikelihood.

Usage:
    conda activate carb_v2
    python dev/nested_sampling_comparison/phi_stats_proxnest_comparison.py [--fast]
"""

import sys
import os
import time
import json
import argparse

script_dir = os.path.dirname(os.path.abspath(__file__))
repo_dir = os.path.dirname(script_dir)  # Priors_selection/
sys.path.insert(0, repo_dir)
sys.path.insert(0, os.path.join(repo_dir, "proxnest_comparison", "code", "proxnest"))

os.environ["TQDM_DISABLE"] = "1"

import numpy as np
import torch
import tensorflow as tf

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from deepinv.physics import Denoising
from deepinv.models import DRUNet as DRUNetModel
from deepinv.loss.metric import PSNR, LPIPS as LPIPS_metric
from priors import WaveletPrior as RepoWaveletPrior, ParametrizedPrior, L2
from prior_comparison import DegradedLikelihood
from sampling import SKROCK
from utils import device

import ProxNest.utils as pn_utils
import ProxNest.sampling as pn_sampling
import ProxNest.optimisations as optimisations
import ProxNest.operators as operators
import ProxNest.operators.proximal_operators as prox_ops

# ---- CLI ----
parser = argparse.ArgumentParser()
parser.add_argument(
    "--fast", action="store_true", help="Reduced sampling for quick test"
)
args = parser.parse_args()

results_dir = os.path.join(script_dir, "results")
os.makedirs(results_dir, exist_ok=True)

# =============================================================================
# EXACT notebook setup
# =============================================================================
DIM = 64
ISNR = 15  # dB — same as notebook

# Load Galaxy
data_dir = os.path.join(repo_dir, "proxnest_comparison", "data")
ground_truth = np.load(os.path.join(data_dir, f"galaxy_image_{DIM}.npy"))
ground_truth -= np.nanmin(ground_truth)
ground_truth /= np.nanmax(ground_truth)
ground_truth[ground_truth < 0] = 0

# Noise level: signal-dependent (same as notebook)
sigma_np = np.sqrt(np.mean(np.abs(ground_truth) ** 2)) * 10 ** (-ISNR / 20)

# Noisy observation
np.random.seed(0)
phi_np = operators.sensing_operators.Identity()
psi_np = operators.wavelet_operators.db_wavelets(["db6"], 2, (DIM, DIM))

n = np.random.normal(0, sigma_np, ground_truth.shape)
y_np = phi_np.dir_op(ground_truth) + n
X0_np = np.abs(phi_np.adj_op(np.copy(y_np)))

noisy_psnr_np = 10 * np.log10(
    np.max(ground_truth) ** 2 / np.mean((ground_truth - y_np) ** 2)
)
print(f"Galaxy {DIM}x{DIM}, ISNR={ISNR} dB, sigma={sigma_np:.6f}")
print(f"Noisy PSNR = {noisy_psnr_np:.2f} dB")

# ---- ProxNest notebook parameters ----
WAV_reg_param = 2e4
alpha_dncnn = 0.005
alpha_drunet = 0.005  # same scaling as DnCNN

delta_step = 1e-7
lamb_pn = 5e-7
gamma_WAV_pn = 5e-7  # = 5 * delta_step
gamma_DnCNN_pn = alpha_dncnn * sigma_np**2
gamma_DRUNet_pn = alpha_drunet * sigma_np**2

print(f"\nProxNest parameters:")
print(f"  delta={delta_step:.1e}, lamb={lamb_pn:.1e}")
print(
    f"  gamma_WAV={gamma_WAV_pn:.2e}, gamma_DnCNN={gamma_DnCNN_pn:.2e}, gamma_DRUNet={gamma_DRUNet_pn:.2e}"
)

# =============================================================================
# Load DnCNN (TensorFlow, same model as notebook)
# =============================================================================
print("\nLoading DnCNN (TF, snr_15_model.pb)...", flush=True)
dncnn_model_path = os.path.join(
    repo_dir,
    "proxnest_comparison",
    "code",
    "proxnest",
    "papers",
    "MaxEnt",
    "networks",
    "DnCNN",
    "snr_15_model.pb",
)
tf_dncnn = tf.saved_model.load(dncnn_model_path)


def tf_dncnn_denoise(x_np):
    """Apply TF DnCNN to a 2D numpy array, return 2D numpy array."""
    tf_img = tf.convert_to_tensor(x_np, dtype=tf.float32)
    tf_img = tf_img[tf.newaxis, ..., tf.newaxis]  # [B, H, W, C]
    tf_out = tf_dncnn(tf_img)
    return tf.squeeze(tf_out, axis=[0, 3]).numpy()


# =============================================================================
# Load DRUNet (deepinv pretrained, grayscale)
# =============================================================================
print("\nLoading DRUNet (deepinv pretrained)...", flush=True)
drunet_sigma_d = 0.02  # denoiser noise level for DRUNet proximal operator
_drunet_model = DRUNetModel(
    in_channels=1, out_channels=1, pretrained="download", device=device
).eval()


def drunet_denoise(x_np):
    """Apply deepinv DRUNet to a 2D numpy array, return 2D numpy array."""
    with torch.no_grad():
        xt = (
            torch.tensor(x_np, dtype=torch.float32, device=device)
            .unsqueeze(0)
            .unsqueeze(0)
        )
        sigma_d_t = torch.tensor([drunet_sigma_d], dtype=torch.float32, device=device)
        return _drunet_model(xt, sigma_d_t).squeeze().cpu().numpy()


# ProxNest proximal operators (numpy-based)
proxH_WAV_np = lambda x, T: prox_ops.l1_projection(x, T, WAV_reg_param, Psi=psi_np)
proxH_DnCNN_np = lambda x, T: tf_dncnn_denoise(x)
proxH_DRUNet_np = lambda x, T: drunet_denoise(x)


# =============================================================================
# Torch tensors for Phi statistics
# =============================================================================
x_gt = (
    torch.tensor(ground_truth, dtype=torch.float32, device=device)
    .unsqueeze(0)
    .unsqueeze(0)
)
sigma_val = sigma_np
sigma_t = torch.tensor(sigma_val, device=device)

# Reproduce the same noisy observation in torch (same seed, same noise)
y_t = torch.tensor(y_np, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)

psnr_fn = PSNR()
noisy_psnr_t = psnr_fn(y_t.clamp(0, 1), x_gt).item()
print(f"Torch noisy PSNR = {noisy_psnr_t:.2f} dB")

# Identity physics
physics = Denoising(device=device)


# =============================================================================
# Define torch-compatible priors for SKROCK sampling
# =============================================================================
class TFDnCNNPrior(ParametrizedPrior):
    """RED prior wrapping the TensorFlow DnCNN model for use with SKROCK."""

    def __init__(self, param):
        super().__init__(param)

    def forward(self, x):
        x_np = x.squeeze().cpu().numpy()
        Dx = tf_dncnn_denoise(x_np)
        Dx_t = (
            torch.tensor(Dx, dtype=torch.float32, device=device)
            .unsqueeze(0)
            .unsqueeze(0)
        )
        return self.param * 0.5 * torch.sum(x * (x - Dx_t))

    def grad(self, x, lam_reg=None):
        x_np = x.squeeze().cpu().numpy()
        Dx = tf_dncnn_denoise(x_np)
        Dx_t = (
            torch.tensor(Dx, dtype=torch.float32, device=device)
            .unsqueeze(0)
            .unsqueeze(0)
        )
        return self.param * (x - Dx_t)

    def lipsch_bound(self, lam_reg=None):
        return self.param * 2  # conservative bound


# Wavelet prior: db6, level 2, reg_param=WAV_reg_param
# The repo WaveletPrior uses level=3 by default. We need level=2 as in the notebook.
from deepinv.optim.prior import WaveletPrior as _DeepInvWaveletPrior


class WaveletPriorLevel2(ParametrizedPrior):
    """Wavelet prior with level=2, matching the notebook setup."""

    def __init__(self, param, wv=None):
        super().__init__(param)
        if wv is None:
            wv = ["db6"]
        self.dinv_wav = _DeepInvWaveletPrior(level=2, wv=wv, p=1, device=device)

    def grad(self, x, lam_reg):
        return (x - self.dinv_wav.prox(x, ths=self.param, gamma=lam_reg)) / lam_reg

    def forward(self, x):
        return self.dinv_wav(x) * self.param

    def grad_param(self, x):
        return self.dinv_wav(x)


# =============================================================================
# Phi statistics sampling config
# =============================================================================
ALPHA = 0.5
S_SKROCK = 15
ETA_SKROCK = 0.05

if args.fast:
    NB_STEPS = 10
    NB_NOISE = 10
    BURNIN_RATIO = 5
    THINNING = 2
    THINNING_NOISE = 5
else:
    NB_STEPS = 100
    NB_NOISE = 10
    BURNIN_RATIO = 50
    THINNING = 2
    THINNING_NOISE = 20

BATCH_SIZE = 1
alpha_fission = torch.tensor(ALPHA, device=device)

# SKROCK step size
ls = (S_SKROCK - 0.5) ** 2 * (2 - 4 / 3 * ETA_SKROCK) - 1.5
L_f = 1.0 / (sigma_val**2 / ALPHA)  # Likelihood Lipschitz (identity A)


def compute_gamma_and_lam(prior_cfg):
    """Compute SKROCK step size and Moreau envelope level."""
    if prior_cfg["is_proximal"]:
        lam_val = prior_cfg["prior"].param
        if isinstance(lam_val, torch.Tensor):
            lam_val = lam_val.item()
        lam_reg = sigma_val / lam_val
        L_moreau = 1.0 / lam_reg
        gamma = 0.98 * ls / (L_f + L_moreau)
        lam_reg = torch.tensor(lam_reg, dtype=torch.float32, device=device)
    else:
        L_g = prior_cfg["prior"].lipsch_bound()
        if isinstance(L_g, torch.Tensor):
            L_g = L_g.item()
        gamma = 0.98 * ls / (L_f + L_g)
        lam_reg = None
    return torch.tensor(gamma, device=device), lam_reg


# =============================================================================
# Define priors
# =============================================================================
# WAV prior: In the ProxNest notebook, the wavelet prior uses:
#   proxH_WAV(x, T) = l1_projection(x, T, WAV_reg_param=2e4, Psi=psi)
#   gamma_WAV = 5e-7 (Moreau envelope parameter in ProxNest drift)
# The effective soft-threshold in ProxNest = gamma_WAV * WAV_reg_param = 5e-7 * 2e4 = 1e-2
#
# In our SKROCK framework, the Moreau gradient is:
#   grad_g(x) = (x - prox_{lam_reg * g}(x)) / lam_reg
# where g(x) = param * ||Psi x||_1
# The soft-threshold = param * lam_reg
# And lam_reg is set automatically as sigma / param (see compute_gamma_and_lam).
#
# To have a reasonable prior strength, we set param such that the denoising
# effect is sensible. Using the same convention as other experiments:
# param controls the L1 norm weight; lam_reg = sigma / param gives the Moreau level.
# With param = sigma/lam_reg, threshold = sigma.
#
# For the notebook's WAV_reg_param=2e4:
# The threshold = T * WAV_reg_param where T=gamma in ProxNest.
# But for SKROCK, lam_reg controls the Moreau smoothing and the threshold = param * lam_reg.
# We want the prior to be strong enough. Let's set param to a moderate value.
wav_param = 20.0  # Same scale as tuned_params.json wavelet_db6
wav_prior = WaveletPriorLevel2(torch.tensor(wav_param), wv=["db6"])

# DnCNN prior: param = 550.0 (tuned RED regularisation weight)
dncnn_prior = TFDnCNNPrior(torch.tensor(550.0))


class DRUNetPrior(ParametrizedPrior):
    """RED prior wrapping the deepinv DRUNet model for use with SKROCK."""

    def __init__(self, param, sigma_d=0.02):
        super().__init__(param)
        self.sigma_d = sigma_d

    def _denoise(self, x):
        with torch.no_grad():
            sigma_d_t = torch.tensor(
                [self.sigma_d], dtype=torch.float32, device=x.device
            )
            return _drunet_model(x, sigma_d_t)

    def forward(self, x):
        Dx = self._denoise(x)
        return self.param * 0.5 * torch.sum(x * (x - Dx))

    def grad(self, x, lam_reg=None):
        Dx = self._denoise(x)
        return self.param * (x - Dx)

    def lipsch_bound(self, lam_reg=None):
        return self.param * 2  # conservative bound


drunet_prior = DRUNetPrior(torch.tensor(1500.0), sigma_d=drunet_sigma_d)

priors_config = {
    "WAV_db6": {"prior": wav_prior, "is_proximal": True},
    "DnCNN": {"prior": dncnn_prior, "is_proximal": False},
    "DRUNet": {"prior": drunet_prior, "is_proximal": False},
}

# =============================================================================
# Shared noise schedule
# =============================================================================
torch.manual_seed(123)
nb_batches = NB_NOISE // BATCH_SIZE
noise_schedule = (
    torch.randn((nb_batches, BATCH_SIZE) + y_t.shape[1:], device=device) * sigma_t
)

# =============================================================================
# LPIPS metric
# =============================================================================
lpips_fn = LPIPS_metric(device=device)

# =============================================================================
# Part 1: Phi statistics (SKROCK posterior sampling)
# =============================================================================
print(f"\n{'='*70}")
print("PART 1: Phi_1 and Phi_2 statistics (SKROCK posterior sampling)")
print(f"  nb_steps={NB_STEPS}, nb_noise={NB_NOISE}, alpha={ALPHA}")
print(f"  S_SKROCK={S_SKROCK}, eta={ETA_SKROCK}")
print(f"{'='*70}")

phi_results = {}

for prior_name, prior_cfg in priors_config.items():
    print(f"\n--- {prior_name} ---")

    prior = prior_cfg["prior"]
    gamma, lam_reg = compute_gamma_and_lam(prior_cfg)

    print(f"  gamma_SKROCK={gamma:.6f}, lam_reg={lam_reg}")

    t0 = time.time()

    dl = DegradedLikelihood(
        y=y_t.clone(),
        prior=prior,
        physics=physics,
        sigma=sigma_t,
        gamma=gamma,
        sampler=SKROCK,
        sampler_kwargs={"s": S_SKROCK, "eta": ETA_SKROCK},
        batch_size=BATCH_SIZE,
        X_init=y_t.clone(),
        lam_reg=lam_reg,
        project="clamp",
        alpha=alpha_fission,
    )

    results = dl.save_samples(
        nb_steps=NB_STEPS,
        nb_noise=NB_NOISE,
        burnin_ratio=BURNIN_RATIO,
        thinning=THINNING,
        thinning_noise=THINNING_NOISE,
        noise_schedule=noise_schedule,
        compute_xp=True,
    )

    samples_x, ym_trace, yp_trace, samples_xp = results
    elapsed = time.time() - t0

    print(f"  Sampling done in {elapsed:.1f}s")
    print(f"  samples_x: {samples_x.shape}, samples_xp: {samples_xp.shape}")

    # ---- Phi_1: E_eps[ (1/N) sum_n ||y+ - x_n||^2 ] ----
    phi1_per_noise = []
    for t in range(NB_NOISE):
        yp = yp_trace[t]
        xs = samples_x[t]
        residuals = yp.unsqueeze(0) - xs
        phi1_t = torch.mean(torch.sum(residuals**2, dim=(1, 2, 3))).item()
        phi1_per_noise.append(phi1_t)

    phi1_mean = np.mean(phi1_per_noise)
    phi1_std = np.std(phi1_per_noise)

    # ---- Phi_2: E_eps[ (1/NL) sum_{n,l} LPIPS(x^-_n, x^+_l) ] ----
    phi2_per_noise = []
    with torch.no_grad():
        for t in range(NB_NOISE):
            xs_minus = samples_x[t].to(device)
            xp = samples_xp[t].unsqueeze(0).to(device)

            xs_3ch = xs_minus.clamp(0, 1).repeat(1, 3, 1, 1)
            xp_3ch = xp.clamp(0, 1).repeat(1, 3, 1, 1)

            lpips_vals = []
            for n_idx in range(xs_3ch.shape[0]):
                lp = lpips_fn(xs_3ch[n_idx : n_idx + 1], xp_3ch).item()
                lpips_vals.append(lp)
            phi2_t = np.mean(lpips_vals)
            phi2_per_noise.append(phi2_t)

    phi2_mean = np.mean(phi2_per_noise)
    phi2_std = np.std(phi2_per_noise)

    # ---- Posterior mean quality ----
    post_mean = torch.mean(samples_x, dim=(0, 1)).to(device)
    recon_psnr = psnr_fn(post_mean.unsqueeze(0).clamp(0, 1), x_gt).item()

    print(f"  Phi_1 = {phi1_mean:.4f} +/- {phi1_std:.4f}")
    print(f"  Phi_2 = {phi2_mean:.6f} +/- {phi2_std:.6f}")
    print(f"  Posterior mean PSNR = {recon_psnr:.2f} dB")

    phi_results[prior_name] = {
        "phi1_mean": phi1_mean,
        "phi1_std": phi1_std,
        "phi1_values": phi1_per_noise,
        "phi2_mean": phi2_mean,
        "phi2_std": phi2_std,
        "phi2_values": phi2_per_noise,
        "recon_psnr": recon_psnr,
        "elapsed": elapsed,
        "post_mean": post_mean.squeeze().cpu().numpy(),
    }

# =============================================================================
# Part 1b: MAP estimation via proximal gradient descent
# =============================================================================
print(f"\n{'='*70}")
print("PART 1b: MAP estimation (proximal gradient descent)")
print(f"{'='*70}")

MAP_ITERS = 500
L_f_full = 1.0 / sigma_val**2  # Lipschitz of likelihood gradient
f_full = L2(sigma_t, physics)

map_results = {}

for prior_name, prior_cfg in priors_config.items():
    print(f"\n--- {prior_name} ---")

    prior = prior_cfg["prior"]

    if prior_cfg["is_proximal"]:
        # PGD: x_{k+1} = prox_{step * g}(x_k - step * grad_f(x_k))
        lam_val = prior.param
        if isinstance(lam_val, torch.Tensor):
            lam_val = lam_val.item()
        lam_reg_map = sigma_val / lam_val
        L_moreau = 1.0 / lam_reg_map
        step = 0.99 / (L_f_full + L_moreau)
        lam_reg_map_t = torch.tensor(lam_reg_map, dtype=torch.float32, device=device)
        print(f"  Moreau PGD: step={step:.6f}, lam_reg={lam_reg_map}")

        t0 = time.time()
        x_map = y_t.clone()
        with torch.no_grad():
            for k in range(MAP_ITERS):
                # Gradient step on smoothed objective
                grad_total = f_full.grad(x_map, y_t) + prior.grad(x_map, lam_reg_map_t)
                x_map = torch.clamp(x_map - step * grad_total, 0.0, 1.0)
    else:
        # RED-style PGD: x_{k+1} = clamp(x_k - step * (grad_f + grad_g))
        L_g = prior.lipsch_bound()
        if isinstance(L_g, torch.Tensor):
            L_g = L_g.item()
        step = 0.99 / (L_f_full + L_g)
        print(f"  RED PGD: step={step:.6f}, L_g={L_g}")

        t0 = time.time()
        x_map = y_t.clone()
        with torch.no_grad():
            for k in range(MAP_ITERS):
                grad_total = f_full.grad(x_map, y_t) + prior.grad(x_map, None)
                x_map = torch.clamp(x_map - step * grad_total, 0.0, 1.0)

    elapsed = time.time() - t0
    map_psnr = psnr_fn(x_map.clamp(0, 1), x_gt).item()
    print(f"  MAP PSNR = {map_psnr:.2f} dB  ({elapsed:.1f}s, {MAP_ITERS} iters)")

    map_results[prior_name] = {
        "psnr": map_psnr,
        "elapsed": elapsed,
        "post_mean": x_map.squeeze().cpu().numpy(),
    }

# =============================================================================
# Part 2: ProxNest evidence (exact notebook parameters)
# =============================================================================
print(f"\n{'='*70}")
print("PART 2: ProxNest Bayesian evidence (exact notebook parameters)")
print(f"{'='*70}")

# ProxNest sampling parameters
if args.fast:
    SAMPLES_L, SAMPLES_D, THINNING_PN, BURN_PN = 20, 200, 5, 50
else:
    SAMPLES_L, SAMPLES_D, THINNING_PN, BURN_PN = 100, 2500, 20, 100

params_pn = pn_utils.create_parameters_dict(
    y=np.copy(y_np),
    Phi=phi_np,
    epsilon=1e-3,
    tight=False,
    nu=1,
    tol=1e-10,
    max_iter=200,
    verbose=0,
    u=0,
    pos=True,
    reality=True,
)
proxB_pn = lambda x, tau: optimisations.l2_ball_proj.sopt_fast_proj_B2(
    x, tau, params_pn
)
LogLikeliL = lambda sol: -np.linalg.norm(y_np - phi_np.dir_op(sol), "fro") ** 2 / (
    2 * sigma_np**2
)


def compute_psnr_np(est, gt):
    dr = gt.max() - gt.min()
    mse = np.mean((gt - est) ** 2)
    return 10 * np.log10(dr**2 / mse) if mse > 0 else float("inf")


pn_results = {}
for name, proxH, gamma_pn in [
    ("WAV_db6", proxH_WAV_np, gamma_WAV_pn),
    ("DnCNN", proxH_DnCNN_np, gamma_DnCNN_pn),
    ("DRUNet", proxH_DRUNet_np, gamma_DRUNet_pn),
]:
    print(
        f"\n--- {name}: delta={delta_step:.1e}, lamb={lamb_pn:.1e}, gamma={gamma_pn:.2e} ---"
    )
    print(
        f"    samplesL={SAMPLES_L}, samplesD={SAMPLES_D}, thinning={THINNING_PN}, burn={BURN_PN}"
    )

    options = pn_utils.create_options_dict(
        samplesL=SAMPLES_L,
        samplesD=SAMPLES_D,
        thinning=THINNING_PN,
        delta=delta_step,
        lamb=lamb_pn,
        burn=BURN_PN,
        sigma=sigma_np,
        gamma=gamma_pn,
    )

    t0 = time.time()
    NS_BayEvi, NS_Trace = pn_sampling.proximal_nested.ProxNestedSampling(
        np.copy(X0_np), LogLikeliL, proxH, proxB_pn, params_pn, options
    )
    elapsed = time.time() - t0

    post_mean_pn = NS_Trace["DiscardPostMean"]
    psnr_pn = compute_psnr_np(post_mean_pn, ground_truth)

    print(f"  logZ = {NS_BayEvi[0]:.2f} +/- {NS_BayEvi[1]:.3f}")
    print(f"  Posterior mean PSNR = {psnr_pn:.2f} dB")
    print(f"  Time: {elapsed:.1f}s")

    pn_results[name] = {
        "logZ": NS_BayEvi[0],
        "logZ_std": NS_BayEvi[1],
        "psnr": psnr_pn,
        "elapsed": elapsed,
        "post_mean": post_mean_pn,
    }

# =============================================================================
# Summary
# =============================================================================
print(f"\n{'='*70}")
print("SUMMARY")
print(f"{'='*70}")
print(f"Galaxy {DIM}x{DIM}, ISNR={ISNR} dB, sigma={sigma_np:.6f}")
print(f"Noisy PSNR = {noisy_psnr_np:.2f} dB")

all_names = ["WAV_db6", "DnCNN", "DRUNet"]

print(
    f"\n{'Prior':<12} {'Phi_1':>10} {'Phi_2':>10} {'PSNR(x|y-)':>12} {'MAP PSNR':>11} {'logZ':>10} {'PSNR_pn':>10}"
)
print("-" * 80)
for name in all_names:
    pr = phi_results[name]
    mp = map_results[name]
    pn = pn_results[name]
    print(
        f"{name:<12} {pr['phi1_mean']:>10.4f} {pr['phi2_mean']:>10.6f} "
        f"{pr['recon_psnr']:>12.2f} {mp['psnr']:>11.2f} {pn['logZ']:>10.2f} {pn['psnr']:>10.2f}"
    )

# Phi_1 ranking (lower is better)
phi1_best = min(phi_results.items(), key=lambda x: x[1]["phi1_mean"])[0]
phi2_best = min(phi_results.items(), key=lambda x: x[1]["phi2_mean"])[0]
logz_best = max(pn_results.items(), key=lambda x: x[1]["logZ"])[0]

print(f"\nPhi_1 winner (lower=better): {phi1_best}")
print(f"Phi_2 winner (lower=better): {phi2_best}")
print(f"logZ winner (higher=better): {logz_best}")

# =============================================================================
# Plot
# =============================================================================
n_models = len(all_names)
colors = ["C0", "C1", "C2", "C3"][:n_models]

fig, axes = plt.subplots(4, n_models + 1, figsize=(5 * (n_models + 1), 16))

# Row 1: Ground truth + SKROCK E[x|y-] posterior means
axes[0, 0].imshow(ground_truth, cmap="cubehelix", vmin=0, vmax=1)
axes[0, 0].set_title("Ground truth", fontsize=10)
axes[0, 0].axis("off")

for col, name in enumerate(all_names):
    pr = phi_results[name]
    axes[0, 1 + col].imshow(
        np.clip(pr["post_mean"], 0, 1), cmap="cubehelix", vmin=0, vmax=1
    )
    axes[0, 1 + col].set_title(
        f"SKROCK E[x|y\u207b] {name}\nPSNR={pr['recon_psnr']:.1f} dB", fontsize=10
    )
    axes[0, 1 + col].axis("off")

# Row 2: Noisy y + MAP estimates
axes[1, 0].imshow(np.clip(y_np, 0, 1), cmap="cubehelix", vmin=0, vmax=1)
axes[1, 0].set_title(f"Noisy y\n{noisy_psnr_np:.1f} dB", fontsize=10)
axes[1, 0].axis("off")

for col, name in enumerate(all_names):
    mp = map_results[name]
    axes[1, 1 + col].imshow(
        np.clip(mp["post_mean"], 0, 1), cmap="cubehelix", vmin=0, vmax=1
    )
    axes[1, 1 + col].set_title(f"MAP {name}\nPSNR={mp['psnr']:.1f} dB", fontsize=10)
    axes[1, 1 + col].axis("off")

# Row 3: ProxNest posterior means
axes[2, 0].axis("off")

for col, name in enumerate(all_names):
    pn = pn_results[name]
    axes[2, 1 + col].imshow(
        np.clip(pn["post_mean"], 0, 1), cmap="cubehelix", vmin=0, vmax=1
    )
    axes[2, 1 + col].set_title(
        f"ProxNest {name}\nlogZ={pn['logZ']:.0f}, PSNR={pn['psnr']:.1f} dB",
        fontsize=10,
    )
    axes[2, 1 + col].axis("off")

# Row 4: Bar charts (Phi_1, Phi_2, logZ, PSNR)
# Phi_1 bar chart
phi1_vals = [phi_results[n]["phi1_mean"] for n in all_names]
phi1_errs = [phi_results[n]["phi1_std"] for n in all_names]
axes[3, 0].bar(all_names, phi1_vals, yerr=phi1_errs, capsize=5, color=colors)
axes[3, 0].set_title("Phi_1 (lower=better)", fontsize=10)
axes[3, 0].set_ylabel("Phi_1")
axes[3, 0].tick_params(axis="x", rotation=20)

# Phi_2 bar chart
phi2_vals = [phi_results[n]["phi2_mean"] for n in all_names]
phi2_errs = [phi_results[n]["phi2_std"] for n in all_names]
axes[3, 1].bar(all_names, phi2_vals, yerr=phi2_errs, capsize=5, color=colors)
axes[3, 1].set_title("Phi_2 (lower=better)", fontsize=10)
axes[3, 1].set_ylabel("Phi_2")
axes[3, 1].tick_params(axis="x", rotation=20)

# logZ bar chart
logz_vals = [pn_results[n]["logZ"] for n in all_names]
axes[3, 2].bar(all_names, logz_vals, capsize=5, color=colors)
axes[3, 2].set_title("logZ (higher=better)", fontsize=10)
axes[3, 2].set_ylabel("logZ")
axes[3, 2].tick_params(axis="x", rotation=20)

# PSNR comparison (SKROCK E[x|y-], SKROCK E[x|y], ProxNest)
if n_models < len(axes[3]):
    x_pos = np.arange(n_models)
    w = 0.25
    psnr_ym = [phi_results[n]["recon_psnr"] for n in all_names]
    psnr_map = [map_results[n]["psnr"] for n in all_names]
    psnr_pn = [pn_results[n]["psnr"] for n in all_names]
    axes[3, 3].bar(
        x_pos - w, psnr_ym, w, label="SKROCK E[x|y\u207b]", color="steelblue"
    )
    axes[3, 3].bar(x_pos, psnr_map, w, label="MAP", color="seagreen")
    axes[3, 3].bar(x_pos + w, psnr_pn, w, label="ProxNest", color="coral")
    axes[3, 3].set_xticks(x_pos)
    axes[3, 3].set_xticklabels(all_names, rotation=20)
    axes[3, 3].set_title("PSNR (dB)", fontsize=10)
    axes[3, 3].set_ylabel("PSNR (dB)")
    axes[3, 3].legend(fontsize=8)

fig.suptitle(
    f"WAV_db6 vs DnCNN vs DRUNet — Galaxy {DIM}x{DIM}, ISNR={ISNR} dB",
    fontsize=12,
    fontweight="bold",
)
fig.tight_layout()

out_plot = os.path.join(results_dir, "phi_stats_proxnest_comparison.png")
fig.savefig(out_plot, dpi=150, bbox_inches="tight")
print(f"\nSaved plot to {out_plot}")

# Save numerical results
out_data = os.path.join(results_dir, "phi_stats_proxnest_comparison.npz")
save_dict = {
    "ground_truth": ground_truth,
    "y": y_np,
    "sigma": sigma_np,
    "ISNR": ISNR,
}
for name in all_names:
    pr = phi_results[name]
    mp = map_results[name]
    pn = pn_results[name]
    save_dict[f"{name}_phi1_mean"] = pr["phi1_mean"]
    save_dict[f"{name}_phi1_std"] = pr["phi1_std"]
    save_dict[f"{name}_phi1_values"] = np.array(pr["phi1_values"])
    save_dict[f"{name}_phi2_mean"] = pr["phi2_mean"]
    save_dict[f"{name}_phi2_std"] = pr["phi2_std"]
    save_dict[f"{name}_phi2_values"] = np.array(pr["phi2_values"])
    save_dict[f"{name}_phi_recon_psnr"] = pr["recon_psnr"]
    save_dict[f"{name}_phi_post_mean"] = pr["post_mean"]
    save_dict[f"{name}_map_psnr"] = mp["psnr"]
    save_dict[f"{name}_map_estimate"] = mp["post_mean"]
    save_dict[f"{name}_logZ"] = pn["logZ"]
    save_dict[f"{name}_logZ_std"] = pn["logZ_std"]
    save_dict[f"{name}_pn_psnr"] = pn["psnr"]
    save_dict[f"{name}_pn_post_mean"] = pn["post_mean"]

np.savez(out_data, **save_dict)
print(f"Saved data to {out_data}")

plt.close("all")
print("\nDone!")
