"""A/B test design and SIMULATION for the retention campaign.

IMPORTANT: nothing in this module uses real experiment data. The dataset has
no treatment/control information. All outcomes are simulated from assumed
churn rates so that the design (sample size, analysis, pitfalls) can be shown
end-to-end before a real launch.

Primary metric: churn rate within the observation window (e.g. 90 days)
among customers in the targeted high-risk segment.
    H0: p_treatment = p_control
    H1: p_treatment != p_control   (two-sided, conservative)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize, proportions_ztest


# ---------------------------------------------------------------------------
# Design: MDE and sample size
# ---------------------------------------------------------------------------

def sample_size_per_arm(p_control: float, mde_relative: float,
                        alpha: float = 0.05, power: float = 0.8) -> int:
    """Customers needed in EACH arm to detect a relative churn reduction.

    Uses Cohen's h (arcsine-transformed difference of proportions), which is
    what statsmodels' NormalIndPower expects for two proportions.
    """
    p_treat = p_control * (1 - mde_relative)
    h = abs(proportion_effectsize(p_control, p_treat))
    n = NormalIndPower().solve_power(effect_size=h, alpha=alpha, power=power,
                                     ratio=1.0, alternative="two-sided")
    return int(np.ceil(n))


def sample_size_formula(p_control: float, mde_relative: float,
                        alpha: float = 0.05, power: float = 0.8) -> int:
    """Same quantity from the textbook formula — used to cross-check statsmodels.

        n = (z_{1-a/2} * sqrt(2 p̄ q̄) + z_{1-b} * sqrt(p1 q1 + p2 q2))^2 / (p1 - p2)^2
    """
    p1, p2 = p_control, p_control * (1 - mde_relative)
    p_bar = (p1 + p2) / 2
    z_a, z_b = stats.norm.ppf(1 - alpha / 2), stats.norm.ppf(power)
    num = (z_a * np.sqrt(2 * p_bar * (1 - p_bar))
           + z_b * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2
    return int(np.ceil(num / (p1 - p2) ** 2))


def mde_for_sample(p_control: float, n_per_arm: int,
                   alpha: float = 0.05, power: float = 0.8) -> float:
    """Inverse question: with n customers per arm, what is the smallest
    relative effect we can reliably detect?"""
    h = NormalIndPower().solve_power(nobs1=n_per_arm, alpha=alpha, power=power,
                                     ratio=1.0, alternative="two-sided")
    # invert Cohen's h: h = 2*asin(sqrt(p1)) - 2*asin(sqrt(p2))
    p_treat = np.sin(np.arcsin(np.sqrt(p_control)) - h / 2) ** 2
    return float(1 - p_treat / p_control)


def power_table(p_control: float, mdes: list[float], alpha: float, power: float) -> pd.DataFrame:
    return pd.DataFrame({
        "mde_relative": mdes,
        "p_control": p_control,
        "p_treatment": [p_control * (1 - m) for m in mdes],
        "n_per_arm": [sample_size_per_arm(p_control, m, alpha, power) for m in mdes],
    })


# ---------------------------------------------------------------------------
# Simulation of one experiment and its analysis
# ---------------------------------------------------------------------------

def simulate_experiment(p_control: float, p_treatment: float, n_per_arm: int,
                        rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Bernoulli churn outcomes (1 = churned) for control and treatment."""
    control = rng.binomial(1, p_control, n_per_arm)
    treatment = rng.binomial(1, p_treatment, n_per_arm)
    return control, treatment


def analyse_ztest(control: np.ndarray, treatment: np.ndarray, alpha: float = 0.05) -> dict:
    """Two-proportion z-test (pooled SE for the test, unpooled SE for the CI)."""
    count = np.array([treatment.sum(), control.sum()])
    nobs = np.array([len(treatment), len(control)])
    z, p = proportions_ztest(count, nobs, alternative="two-sided")
    p_t, p_c = treatment.mean(), control.mean()
    diff = p_t - p_c
    se = np.sqrt(p_t * (1 - p_t) / nobs[0] + p_c * (1 - p_c) / nobs[1])
    z_crit = stats.norm.ppf(1 - alpha / 2)
    return {
        "p_control": p_c,
        "p_treatment": p_t,
        "abs_diff": diff,
        "rel_diff": diff / p_c if p_c > 0 else np.nan,
        "ci_low": diff - z_crit * se,
        "ci_high": diff + z_crit * se,
        "z": z,
        "p_value": p,
        "significant": p < alpha,
    }


def bootstrap_diff(control: np.ndarray, treatment: np.ndarray, n_boot: int,
                   rng: np.random.Generator, alpha: float = 0.05) -> dict:
    """Percentile bootstrap CI for the difference in churn rates.

    Resamples each arm independently with replacement. Vectorised with
    binomial draws: resampling n Bernoulli values with replacement and summing
    is exactly Binomial(n, observed rate).
    """
    n_c, n_t = len(control), len(treatment)
    boot_c = rng.binomial(n_c, control.mean(), n_boot) / n_c
    boot_t = rng.binomial(n_t, treatment.mean(), n_boot) / n_t
    diffs = boot_t - boot_c
    lo, hi = np.quantile(diffs, [alpha / 2, 1 - alpha / 2])
    # two-sided bootstrap p-value: how often the sign flips around 0
    p_boot = 2 * min((diffs >= 0).mean(), (diffs <= 0).mean())
    return {"ci_low": lo, "ci_high": hi, "p_value": min(p_boot, 1.0), "diffs": diffs}


def monte_carlo_power(p_control: float, p_treatment: float, n_per_arm: int,
                      n_sims: int, rng: np.random.Generator, alpha: float = 0.05) -> float:
    """Share of simulated experiments that reject H0.

    With p_treatment == p_control this is the empirical false-positive rate
    (should be ~alpha); otherwise it is the empirical power.
    """
    c = rng.binomial(n_per_arm, p_control, n_sims)
    t = rng.binomial(n_per_arm, p_treatment, n_sims)
    p_pool = (c + t) / (2 * n_per_arm)
    se = np.sqrt(p_pool * (1 - p_pool) * 2 / n_per_arm)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = (t - c) / n_per_arm / se
    p = 2 * stats.norm.sf(np.abs(np.nan_to_num(z)))
    return float((p < alpha).mean())


# ---------------------------------------------------------------------------
# Peeking
# ---------------------------------------------------------------------------

def _sequential_pvalues(c_cum: np.ndarray, t_cum: np.ndarray, n_at_look: np.ndarray) -> np.ndarray:
    """Pooled z-test p-values at each interim look (vectorised over sims x looks)."""
    p_pool = (c_cum + t_cum) / (2 * n_at_look)
    se = np.sqrt(p_pool * (1 - p_pool) * 2 / n_at_look)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = (t_cum / n_at_look - c_cum / n_at_look) / se
    return 2 * stats.norm.sf(np.abs(np.nan_to_num(z)))


def simulate_peeking(p_control: float, p_treatment: float, n_per_arm: int, n_looks: int,
                     n_sims: int, rng: np.random.Generator) -> np.ndarray:
    """Simulate n_sims experiments, each checked at n_looks equally spaced points.

    Returns a (n_sims, n_looks) array of p-values. Customers arrive in
    batches; at each look we test on everyone observed so far.
    """
    batch = n_per_arm // n_looks
    n_at_look = batch * np.arange(1, n_looks + 1)
    c_batches = rng.binomial(batch, p_control, (n_sims, n_looks))
    t_batches = rng.binomial(batch, p_treatment, (n_sims, n_looks))
    return _sequential_pvalues(c_batches.cumsum(1), t_batches.cumsum(1), n_at_look)


def peeking_error_rates(pvals: np.ndarray, alpha: float = 0.05) -> dict:
    """Rejection rate if we (a) stop at the first p < alpha, (b) look only once at the end."""
    stop_early = (pvals < alpha).any(axis=1).mean()
    fixed_horizon = (pvals[:, -1] < alpha).mean()
    return {"peeking_reject_rate": float(stop_early), "fixed_horizon_reject_rate": float(fixed_horizon)}


def calibrate_pocock_alpha(pvals_null: np.ndarray, target_alpha: float = 0.05) -> float:
    """Find a constant per-look threshold that keeps the OVERALL false positive
    rate at target_alpha across all looks (Pocock-style boundary).

    Calibrated by simulation under H0 rather than taken from tables, so it
    matches exactly the number of looks we use.
    """
    min_p = pvals_null.min(axis=1)        # an experiment rejects if ANY look is below threshold
    return float(np.quantile(min_p, target_alpha))
