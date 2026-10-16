"""CoDeC: Conformal Deferral Control for tiered device-edge-cloud inference.

Statistical core. Three components:

1. `hb_pvalue` / `hb_ucb`     -- Hoeffding-Bentkus p-values and upper confidence
                                 bounds for a mean of [0,1]-bounded losses
                                 (Bates et al., JACM 2021; Angelopoulos et al., LTT).
2. `StratifiedRiskBound`      -- self-labelled risk bound under policy-induced
                                 selection bias. The escalation event A is observable
                                 for every stream sample without consulting the cloud;
                                 cloud labels are available on all of A and on an
                                 epsilon-random subsample of its complement. Post-
                                 stratifying on A therefore beats raw inverse-propensity
                                 weighting, whose 1/epsilon weight range inflates the
                                 concentration bound by the same factor.
3. `certify_fst`              -- fixed-sequence testing along a cost-ordered path
                                 through the joint threshold grid. Because the sequence
                                 is fixed before the calibration losses are seen, each
                                 test runs at full level delta with no multiplicity
                                 correction, and no monotonicity of risk in the
                                 threshold vector is required.

All risks are *fidelity* risks -- disagreement with the top-tier (cloud) decision --
unless stated otherwise. Proposition 1 in the paper bounds the task risk by the cloud's
own error plus the certified fidelity risk.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import beta, binom

__all__ = [
    "hb_pvalue", "hb_ucb", "clopper_pearson_upper", "clopper_pearson_lower",
    "StratifiedRiskBound", "certify_fst", "certify_bonferroni",
    "conformal_pvalues", "weighted_conformal_pvalues", "ACIController",
]

_EPS = 1e-12


# --------------------------------------------------------------------------- #
# Concentration for bounded means
# --------------------------------------------------------------------------- #
def _h1(a: float, b: float) -> float:
    """KL-type rate function for Hoeffding's inequality on [0,1] variables."""
    a = min(max(a, _EPS), 1 - _EPS)
    b = min(max(b, _EPS), 1 - _EPS)
    return a * np.log(a / b) + (1 - a) * np.log((1 - a) / (1 - b))


def hb_pvalue(rhat: float, n: int, alpha: float) -> float:
    """Hoeffding-Bentkus p-value for H0: R >= alpha, given empirical mean `rhat`.

    Valid for any distribution of losses supported on [0,1]. Rejecting at level
    delta certifies R < alpha with probability at least 1 - delta.
    """
    if n <= 0:
        return 1.0
    if rhat >= alpha:
        return 1.0
    hoeff = float(np.exp(-n * _h1(min(rhat, alpha), alpha)))
    bentkus = float(np.e * binom.cdf(np.ceil(n * rhat), n, alpha))
    return float(min(1.0, hoeff, bentkus))


def hb_ucb(rhat: float, n: int, delta: float, tol: float = 1e-6) -> float:
    """Smallest alpha whose H0 is rejected at level delta: an HB upper bound on R."""
    if n <= 0:
        return 1.0
    if hb_pvalue(rhat, n, 1.0 - _EPS) > delta:
        return 1.0
    lo, hi = max(rhat, 0.0), 1.0
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if hb_pvalue(rhat, n, mid) <= delta:
            hi = mid
        else:
            lo = mid
    return float(hi)


def clopper_pearson_upper(k: int, n: int, delta: float) -> float:
    """Exact upper confidence bound for a binomial proportion."""
    if n == 0 or k >= n:
        return 1.0
    return float(beta.ppf(1 - delta, k + 1, n - k))


def clopper_pearson_lower(k: int, n: int, delta: float) -> float:
    """Exact lower confidence bound for a binomial proportion."""
    if n == 0 or k <= 0:
        return 0.0
    return float(beta.ppf(delta, k, n - k + 1))


# --------------------------------------------------------------------------- #
# Self-labelled, selection-bias-corrected risk bound
# --------------------------------------------------------------------------- #
class StratifiedRiskBound:
    """Upper confidence bound on fidelity risk from policy-generated cloud labels.

    A deployed cascade generates its own labels, but only where it chose to escalate.
    CoDeC post-stratifies on the OBSERVATION PROPENSITY, which the system knows exactly
    for every record it has processed:

        stratum 1 (pi = 1)     : the deployed policy escalated to the cloud, so the
                                 cloud decision came back as a matter of course;
        stratum eps (pi = eps) : the policy answered locally, and a cloud decision
                                 exists only where epsilon-exploration forced an
                                 escalation -- an i.i.d. Bernoulli(eps) subsample.

    Writing R = p*E[loss | pi=1] + (1-p)*E[loss | pi=eps], each factor is bounded
    separately and the confidence budget is split three ways: p by Clopper-Pearson, and
    the two conditional means by Hoeffding-Bentkus. This is tighter than inverse-
    propensity weighting for the reason that matters here: stratum MEMBERSHIP is a
    deterministic function of the deferral scores, so p is estimated from the whole
    stream rather than from the labelled part alone, and the heavily-subsampled stratum
    contributes only through the weight (1-p), which is typically small. Raw IPW instead
    applies one concentration bound to losses spread over [0, 1/eps] and pays the full
    1/eps widening on every sample.

    The stratification is valid across windows in which different thresholds were
    deployed, because it keys on the realised propensity of each record rather than on
    any particular policy. Residual drift between the buffer and the current stream is
    the job of `ACIController`, not of this bound.

    Parameters
    ----------
    mode : {"stratified", "ipw", "naive"}
        "stratified" is CoDeC. "ipw" is Horvitz-Thompson with Hoeffding-Bentkus applied
        to losses rescaled from [0, 1/epsilon] (ablation). "naive" pools the observed
        losses and ignores the bias (ablation; not valid).
    """

    def __init__(self, epsilon: float, delta: float, mode: str = "stratified"):
        if mode not in ("stratified", "ipw", "naive"):
            raise ValueError(f"unknown mode {mode!r}")
        self.epsilon = float(epsilon)
        self.delta = float(delta)
        self.mode = mode

    def bound(self, strata, delta: float | None = None):
        """Return (risk_ucb, risk_point, diagnostics).

        Parameters
        ----------
        strata : two dicts, propensity-1 stratum first, each with `losses` (observed 0/1
            fidelity losses), `n_pop` (records in the stratum whether observed or not)
            and `propensity`.
        delta : overrides the instance level (used by the Bonferroni ablation).
        """
        dtot = self.delta if delta is None else float(delta)
        obs = [np.asarray(s["losses"], dtype=float).ravel() for s in strata]
        n_pop = [int(s["n_pop"]) for s in strata]
        n_tot = sum(n_pop)
        n_obs_tot = int(sum(o.size for o in obs))
        if n_tot == 0:
            return 1.0, 1.0, {"n_obs": 0, "ess": 0.0}
        means = [float(o.mean()) if o.size else 0.0 for o in obs]
        p_hat = n_pop[0] / n_tot
        point = float(p_hat * means[0] + (1 - p_hat) * means[1]) if len(strata) > 1 else means[0]

        if self.mode == "naive":
            pooled = np.concatenate(obs) if n_obs_tot else np.zeros(0)
            rhat = float(pooled.mean()) if pooled.size else 0.0
            return hb_ucb(rhat, int(pooled.size), dtot), rhat, {
                "n_obs": n_obs_tot, "ess": float(n_obs_tot), "p_hat": p_hat}

        if self.mode == "ipw":
            parts, weights = [], []
            for s, o in zip(strata, obs):
                pi = max(float(s["propensity"]), _EPS)
                parts.append(o * (self.epsilon / pi))      # rescale into [0,1]
                weights.append(np.full(o.size, 1.0 / pi))
            allp = np.concatenate(parts) if n_obs_tot else np.zeros(0)
            allw = np.concatenate(weights) if n_obs_tot else np.zeros(0)
            rhat_scaled = float(allp.mean()) if allp.size else 0.0
            ucb = min(1.0, hb_ucb(rhat_scaled, int(allp.size), dtot) / max(self.epsilon, _EPS))
            ess = (float(allw.sum() ** 2 / max((allw ** 2).sum(), _EPS)) if allw.size else 0.0)
            return ucb, rhat_scaled / max(self.epsilon, _EPS), {
                "n_obs": n_obs_tot, "ess": ess, "p_hat": p_hat}

        # --- stratified (CoDeC) --------------------------------------------- #
        # Budget allocation: the stratum weight is estimated from the WHOLE stream and
        # is therefore sharp, so it needs only a sliver of delta; the two conditional
        # means, estimated from the labelled subsample, get the rest.
        d_p = 0.10 * dtot
        d_m = 0.45 * dtot
        # The fidelity loss is Bernoulli, so the exact binomial (Clopper-Pearson) bound
        # applies and is strictly tighter than Hoeffding-Bentkus, which must hold for
        # every [0,1]-valued loss. hb_ucb remains available for non-binary risks.
        u1 = (clopper_pearson_upper(int(round(means[0] * obs[0].size)), int(obs[0].size), d_m)
              if obs[0].size else 1.0)
        ue = (clopper_pearson_upper(int(round(means[1] * obs[1].size)), int(obs[1].size), d_m)
              if obs[1].size else 1.0)
        p_hi = clopper_pearson_upper(n_pop[0], n_tot, d_p)
        p_lo = clopper_pearson_lower(n_pop[0], n_tot, d_p)
        ucb = float(min(1.0, max(p_hi * u1 + (1 - p_hi) * ue,
                                 p_lo * u1 + (1 - p_lo) * ue)))
        return ucb, point, {
            "n_obs": n_obs_tot, "ess": float(n_obs_tot), "p_hat": p_hat,
            "u_pi1": u1, "u_pieps": ue,
            "n_obs_pi1": int(obs[0].size), "n_obs_pieps": int(obs[1].size)}


# --------------------------------------------------------------------------- #
# Certification over the joint threshold grid
# --------------------------------------------------------------------------- #
def certify_fst(candidates, risk_fn, alpha: float, delta: float):
    """Fixed-sequence test along a cost-ordered path.

    `candidates` must already run from SAFEST/most expensive to CHEAPEST, with the
    order fixed independently of the losses `risk_fn` will see. Testing stops at the
    first failure, so the certified set is a prefix and the family-wise error rate is
    at most delta with no correction and no monotonicity assumption.

    Returns (selected, certified, trace); `selected` is the cheapest certified
    candidate, or None if even the safest fails.
    """
    certified, trace = [], []
    for cand in candidates:
        ucb, point, diag = risk_fn(cand)
        ok = bool(ucb <= alpha)
        trace.append({"cand": cand, "ucb": ucb, "point": point, "certified": ok, **diag})
        if not ok:
            break
        certified.append(cand)
    return (certified[-1] if certified else None), certified, trace


def certify_bonferroni(candidates, risk_fn, alpha: float, delta: float):
    """Ablation: test every candidate at level delta/|candidates|, no ordering used."""
    m = max(len(candidates), 1)
    d = delta / m
    certified, trace = [], []
    for cand in candidates:
        ucb, point, diag = risk_fn(cand, d)
        ok = bool(ucb <= alpha)
        trace.append({"cand": cand, "ucb": ucb, "point": point, "certified": ok, **diag})
        if ok:
            certified.append(cand)
    return (certified[-1] if certified else None), certified, trace


# --------------------------------------------------------------------------- #
# Deferral scores
# --------------------------------------------------------------------------- #
def conformal_pvalues(cal_scores: np.ndarray, test_scores: np.ndarray) -> np.ndarray:
    """Marginal conformal p-values of test nonconformity scores.

    p(x) = (1 + #{i : s_i >= s(x)}) / (n + 1). Small p means atypical: the local
    tier is operating outside its calibrated competence and should defer.
    """
    cal = np.sort(np.asarray(cal_scores, dtype=float))
    ts = np.asarray(test_scores, dtype=float)
    n = cal.size
    if n == 0:
        return np.ones_like(ts)
    ge = n - np.searchsorted(cal, ts, side="left")
    return (1.0 + ge) / (n + 1.0)


def weighted_conformal_pvalues(cal_scores, cal_weights, test_scores) -> np.ndarray:
    """Weighted conformal p-values (Tibshirani et al., 2019) for biased calibration.

    Weights are inverse propensities; only their relative size matters. Used when the
    calibration pool was collected by the deferral policy itself.
    """
    cal = np.asarray(cal_scores, dtype=float)
    w = np.asarray(cal_weights, dtype=float)
    ts = np.asarray(test_scores, dtype=float)
    if cal.size == 0:
        return np.ones_like(ts)
    w = w / max(w.mean(), _EPS)          # unit mean so the test point's mass is comparable
    order = np.argsort(cal)
    cal, w = cal[order], w[order]
    csum = np.concatenate([[0.0], np.cumsum(w)])
    idx = np.searchsorted(cal, ts, side="left")
    w_ge = w.sum() - csum[idx]
    return (1.0 + w_ge) / (w.sum() + 1.0)


# --------------------------------------------------------------------------- #
# Online risk tracking under drift
# --------------------------------------------------------------------------- #
class ACIController:
    """Adaptive risk-level tracking (Gibbs & Candes, 2021) with an uplink-budget dual.

    The certification level used at round t is alpha_t. When realised risk runs hot,
    alpha_t contracts and the controller escalates more; when it runs cold, alpha_t
    relaxes and cost falls. `mu` responds to sustained budget overrun, and `debt`
    accumulates risk that could not be controlled because escalation was impossible
    (outage) or unaffordable.
    """

    def __init__(self, alpha: float, eta: float = 0.05, budget: float | None = None,
                 eta_mu: float = 0.05, alpha_min: float = 1e-3, alpha_max: float = 0.5):
        self.alpha0 = float(alpha)
        self.alpha_t = float(alpha)
        self.eta = float(eta)
        self.budget = budget
        self.eta_mu = float(eta_mu)
        self.mu = 0.0
        self.debt = 0.0
        self.alpha_min, self.alpha_max = float(alpha_min), float(alpha_max)
        self.history: list[dict] = []

    def update(self, realised_risk: float, realised_cost: float | None = None,
               feasible: bool = True) -> dict:
        err = float(realised_risk) - self.alpha0
        self.alpha_t = float(np.clip(self.alpha_t - self.eta * err,
                                     self.alpha_min, self.alpha_max))
        if self.budget is not None and realised_cost is not None:
            self.mu = max(0.0, self.mu + self.eta_mu * (float(realised_cost) - self.budget))
        if not feasible and realised_risk > self.alpha0:
            self.debt += realised_risk - self.alpha0
        elif feasible and self.debt > 0 and realised_risk < self.alpha0:
            self.debt = max(0.0, self.debt - (self.alpha0 - realised_risk))
        rec = {"alpha_t": self.alpha_t, "mu": self.mu, "debt": self.debt,
               "realised_risk": float(realised_risk), "feasible": bool(feasible)}
        self.history.append(rec)
        return rec
