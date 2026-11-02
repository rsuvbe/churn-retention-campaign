"""Hypothesis testing helpers that always return an effect size.

With ~10k customers almost any difference is "statistically significant",
so every test here reports an effect size and the decision is made on
(corrected p-value AND practically meaningful effect), not on p alone.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests


@dataclass
class TestResult:
    hypothesis: str
    test: str
    statistic: float
    p_value: float
    effect_size_name: str
    effect_size: float
    effect_magnitude: str
    detail: str


def _magnitude(value: float, thresholds: tuple[float, float, float]) -> str:
    """Map |effect| to Cohen-style labels: negligible / small / medium / large."""
    small, medium, large = thresholds
    v = abs(value)
    if v < small:
        return "negligible"
    if v < medium:
        return "small"
    if v < large:
        return "medium"
    return "large"


def chi_square(df: pd.DataFrame, group_col: str, target: str, hypothesis: str) -> TestResult:
    """Chi-square test of independence + Cramér's V.

    Use for categorical vs categorical (segment x churned).
    """
    table = pd.crosstab(df[group_col], df[target])
    chi2, p, dof, expected = stats.chi2_contingency(table)
    n = table.to_numpy().sum()
    k = min(table.shape) - 1
    cramers_v = float(np.sqrt(chi2 / (n * k)))
    # Cohen thresholds for Cramér's V depend on dof; for df*=1 they are .1/.3/.5
    if k == 1:
        thresholds = (0.1, 0.3, 0.5)
    elif k == 2:
        thresholds = (0.07, 0.21, 0.35)
    else:
        thresholds = (0.06, 0.17, 0.29)
    rates = df.groupby(group_col)[target].mean().round(3).to_dict()
    return TestResult(
        hypothesis=hypothesis,
        test="Chi-square independence",
        statistic=float(chi2),
        p_value=float(p),
        effect_size_name="Cramér's V",
        effect_size=cramers_v,
        effect_magnitude=_magnitude(cramers_v, thresholds),
        detail=f"churn rate by group: {rates}; min expected count={expected.min():.0f}",
    )


def mann_whitney(df: pd.DataFrame, value_col: str, target: str, hypothesis: str,
                 alternative: str = "two-sided") -> TestResult:
    """Mann-Whitney U + rank-biserial correlation.

    Use for a numeric, skewed or ordinal variable between churned / retained.
    It compares distributions (stochastic dominance), not means, so no
    normality assumption is needed.
    Rank-biserial r = 2*U1/(n1*n2) - 1 = P(churner > retained) - P(churner < retained)
    over all (churner, retained) pairs. Negative r -> churners have LOWER values.
    """
    churned = df.loc[df[target] == 1, value_col]
    retained = df.loc[df[target] == 0, value_col]
    # scipy returns U for the FIRST sample (churned)
    u, p = stats.mannwhitneyu(churned, retained, alternative=alternative)
    r_rb = 2 * u / (len(churned) * len(retained)) - 1
    return TestResult(
        hypothesis=hypothesis,
        test=f"Mann-Whitney U ({alternative})",
        statistic=float(u),
        p_value=float(p),
        effect_size_name="rank-biserial r (churned vs retained)",
        effect_size=float(r_rb),
        effect_magnitude=_magnitude(r_rb, (0.1, 0.3, 0.5)),
        detail=(f"median churned={churned.median():.3f}, "
                f"median retained={retained.median():.3f}"),
    )


def welch_t_test(df: pd.DataFrame, value_col: str, target: str, hypothesis: str) -> TestResult:
    """Welch's t-test (unequal variances) + Cohen's d.

    Use for a roughly symmetric numeric variable where the MEAN is the
    quantity of interest. With n in the thousands the CLT makes the test
    robust to mild non-normality.
    """
    churned = df.loc[df[target] == 1, value_col]
    retained = df.loc[df[target] == 0, value_col]
    t, p = stats.ttest_ind(churned, retained, equal_var=False)
    pooled_sd = np.sqrt((churned.var(ddof=1) + retained.var(ddof=1)) / 2)
    d = (churned.mean() - retained.mean()) / pooled_sd
    return TestResult(
        hypothesis=hypothesis,
        test="Welch t-test",
        statistic=float(t),
        p_value=float(p),
        effect_size_name="Cohen's d",
        effect_size=float(d),
        effect_magnitude=_magnitude(d, (0.2, 0.5, 0.8)),
        detail=f"mean churned={churned.mean():.2f}, mean retained={retained.mean():.2f}",
    )


def correct_pvalues(results: list[TestResult], alpha: float = 0.05,
                    method: str = "holm") -> pd.DataFrame:
    """Holm-Bonferroni correction across all hypotheses tested together.

    Holm controls the family-wise error rate like Bonferroni but is uniformly
    more powerful, so there is no reason to use plain Bonferroni.
    """
    table = pd.DataFrame([asdict(r) for r in results])
    reject, p_adj, _, _ = multipletests(table["p_value"], alpha=alpha, method=method)
    table["p_adjusted"] = p_adj
    table["reject_h0"] = reject
    # A finding is "actionable" only if significant AND not negligible in size.
    table["actionable"] = table["reject_h0"] & (table["effect_magnitude"] != "negligible")
    return table


def risk_ratio(df: pd.DataFrame, flag_col: str, target: str) -> tuple[float, float, float]:
    """Churn rate with flag, without flag, and their ratio (easy for business)."""
    with_flag = df.loc[df[flag_col] == 1, target].mean()
    without = df.loc[df[flag_col] == 0, target].mean()
    return float(with_flag), float(without), float(with_flag / without)
