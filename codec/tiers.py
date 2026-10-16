"""Three-tier device-edge-cloud cascade: tier models, cost model, cascade simulator.

Tier hierarchy is grounded in each dataset's own feature semantics rather than an
arbitrary split (see `codec.data`): the device tier sees only the features an MCU can
cheaply compute, the gateway tier a richer subset, the cloud tier everything.

Cost accounting separates what is MEASURED from what is MODELLED:

  measured  : per-sample inference latency (x86 CPU, batch size 1, median of repeats),
              payload sizes in bytes (actual float32 serialisation of the representation
              the next tier requires), model parameter counts and multiply-accumulates.
  modelled  : energy, via published per-MAC and per-byte coefficients, and cloud
              monetary cost per inference. Coefficients live in `CostModel` and are
              reported in the paper's parameter table. They are NOT measurements on
              physical IoT hardware; no such testbed was used.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

__all__ = ["TierSpec", "CostModel", "TieredSystem", "simulate_cascade"]


# --------------------------------------------------------------------------- #
# Tier definition
# --------------------------------------------------------------------------- #
@dataclass
class TierSpec:
    """One tier of the cascade."""
    name: str
    feat_idx: np.ndarray          # columns of X this tier consumes
    hidden: tuple                 # MLP hidden layer sizes
    payload_dim: int              # floats transmitted to REACH this tier from the previous
    model: object = None
    macs: int = 0
    params: int = 0
    latency_us: float = 0.0
    payload_bytes: int = 0

    def summary(self) -> dict:
        d = asdict(self)
        d.pop("feat_idx"); d.pop("model")
        d["n_features"] = int(len(self.feat_idx))
        d["hidden"] = str(self.hidden)
        return d


@dataclass
class CostModel:
    """Per-sample cost of reaching and running each tier.

    Defaults are representative values for a Cortex-M-class device with a short-range
    radio and a commodity cloud endpoint. They are parameters of the study, not
    measurements: `energy_*` and `price_cloud_per_1k` are MODELLED.
    """
    e_per_mac_pj: float = 3.0            # pJ per multiply-accumulate, MCU class
    e_per_byte_local_nj: float = 150.0   # nJ per byte, device -> gateway radio
    e_per_byte_wan_nj: float = 0.0       # gateway is mains powered: no device-battery cost
    kappa_feat_mac: int = 64             # MACs to extract one feature
    price_cloud_per_1k: float = 0.20     # USD per 1000 cloud inferences
    tput_local_kBps: float = 200.0       # device -> gateway goodput
    tput_wan_kBps: float = 500.0         # gateway -> cloud goodput
    rtt_wan_ms: float = 40.0             # gateway -> cloud round trip
    mac_rate_MMACps: tuple = (100.0, 2000.0, 50000.0)   # device / gateway / cloud
    w_energy: float = 0.5                # scalar-cost weights (normalised components)
    w_money: float = 0.3
    w_latency: float = 0.2

    def tier_costs(self, tiers: list[TierSpec]) -> dict:
        """Cumulative cost of *answering at* each tier: energy (mJ), USD, latency (ms)."""
        e, usd, lat = [], [], []
        cum_e = cum_lat = cum_usd = 0.0
        for k, t in enumerate(tiers):
            compute_macs = t.macs + self.kappa_feat_mac * len(t.feat_idx)
            cum_e += compute_macs * self.e_per_mac_pj * 1e-12 * 1e3          # mJ
            cum_lat += compute_macs / (self.mac_rate_MMACps[k] * 1e6) * 1e3   # ms
            if k == 1:      # device -> gateway hop
                cum_e += t.payload_bytes * self.e_per_byte_local_nj * 1e-9 * 1e3
                cum_lat += t.payload_bytes / (self.tput_local_kBps * 1e3) * 1e3
            elif k == 2:    # gateway -> cloud hop
                cum_e += t.payload_bytes * self.e_per_byte_wan_nj * 1e-9 * 1e3
                cum_lat += t.payload_bytes / (self.tput_wan_kBps * 1e3) * 1e3 + self.rtt_wan_ms
                cum_usd += self.price_cloud_per_1k / 1000.0
            e.append(cum_e); usd.append(cum_usd); lat.append(cum_lat)
        return {"energy_mJ": np.array(e), "usd": np.array(usd), "latency_ms": np.array(lat)}

    def scalar(self, comp: dict, ref: dict) -> np.ndarray:
        """Weighted sum of components normalised by the cloud-only reference."""
        return (self.w_energy * comp["energy_mJ"] / max(ref["energy_mJ"], 1e-12)
                + self.w_money * comp["usd"] / max(ref["usd"], 1e-12)
                + self.w_latency * comp["latency_ms"] / max(ref["latency_ms"], 1e-12))


# --------------------------------------------------------------------------- #
# Fitting and profiling
# --------------------------------------------------------------------------- #
class TieredSystem:
    """Fits the three tiers, profiles them, and precomputes per-sample cascade inputs."""

    def __init__(self, tiers: list[TierSpec], cost: CostModel, seed: int = 0):
        self.tiers = tiers
        self.cost = cost
        self.seed = seed
        self.cal_scores: list[np.ndarray] = []

    def fit(self, X, y, idx_train, idx_cal, max_iter: int = 400):
        for k, t in enumerate(self.tiers):
            clf = Pipeline([
                ("sc", StandardScaler()),
                ("mlp", MLPClassifier(hidden_layer_sizes=t.hidden, max_iter=max_iter,
                                      random_state=self.seed + k, early_stopping=False,
                                      alpha=1e-4)),
            ])
            clf.fit(X[np.ix_(idx_train, t.feat_idx)], y[idx_train])
            t.model = clf
            d_in = len(t.feat_idx)
            n_cls = len(np.unique(y))
            dims = [d_in, *t.hidden, n_cls]
            t.macs = int(sum(dims[i] * dims[i + 1] for i in range(len(dims) - 1)))
            t.params = t.macs + int(sum(dims[1:]))
            t.payload_bytes = int(t.payload_dim * 4)   # float32 wire format
            # offline conformal calibration on a labelled held-out split
            proba = clf.predict_proba(X[np.ix_(idx_cal, t.feat_idx)])
            cls = clf.classes_
            pos = {c: i for i, c in enumerate(cls)}
            true_col = np.array([pos.get(v, 0) for v in y[idx_cal]])
            self.cal_scores.append(1.0 - proba[np.arange(len(idx_cal)), true_col])
        return self

    def profile_latency(self, X, n_probe: int = 512, repeats: int = 3):
        """Measure batch-amortised per-sample inference time on this CPU.

        Reported as a cross-check that the MAC ordering of the tiers is real. It is NOT
        used in the cost model: single-sample x86 latency is dominated by framework
        dispatch (~1 ms regardless of model size), and the target devices are not x86.
        Cost-model latency is derived from MAC counts at the per-tier throughputs in
        `CostModel.mac_rate_MMACps`, which are MODELLED.
        """
        rng = np.random.default_rng(self.seed)
        probe = rng.choice(len(X), size=min(n_probe, len(X)), replace=False)
        for t in self.tiers:
            Xs = X[np.ix_(probe, t.feat_idx)]
            t.model.predict_proba(Xs[:8])            # warm up
            best = np.inf
            for _ in range(repeats):
                t0 = time.perf_counter()
                t.model.predict_proba(Xs)
                best = min(best, (time.perf_counter() - t0) / len(Xs))
            t.latency_us = float(best * 1e6)
        return self

    def precompute(self, X, idx):
        """Per-sample tier outputs and nonconformity scores for a stream slice."""
        out = {}
        for k, t in enumerate(self.tiers):
            proba = t.model.predict_proba(X[np.ix_(idx, t.feat_idx)])
            cls = t.model.classes_
            top = proba.argmax(1)
            out[f"yhat{k + 1}"] = cls[top]
            out[f"score{k + 1}"] = 1.0 - proba[np.arange(len(idx)), top]
            out[f"conf{k + 1}"] = proba[np.arange(len(idx)), top]
            out[f"entropy{k + 1}"] = -(proba * np.log(proba + 1e-12)).sum(1)
        return out

    def cost_table(self):
        comp = self.cost.tier_costs(self.tiers)
        ref = {kk: float(vv[-1]) for kk, vv in comp.items()}
        comp["scalar"] = self.cost.scalar(comp, ref)
        return comp, ref


# --------------------------------------------------------------------------- #
# Cascade simulation
# --------------------------------------------------------------------------- #
def simulate_cascade(p1, p2, lam1, lam2, explore=None, cloud_up=None):
    """Vectorised three-tier cascade.

    Parameters
    ----------
    p1, p2 : deferral scores at tiers 1 and 2 (conformal p-values; small = atypical).
    lam1, lam2 : thresholds. A sample escalates from tier k when p_k < lam_k.
    explore : optional boolean mask of epsilon-forced escalations. Forced samples are
        transmitted so the cloud label can be observed, but the SYSTEM OUTPUT remains
        the policy's own answer, so exploration does not alter the estimand.
    cloud_up : optional boolean mask of cloud reachability. Where the cloud is down an
        escalation to tier 3 cannot complete and the gateway answers instead.

    Returns dict of per-sample: tier_out (1/2/3, tier that produced the answer),
    tier_reached (billing tier), observed_cloud (cloud label available), stratum_A
    (policy escalated to cloud, i.e. propensity 1).
    """
    n = len(p1)
    esc1 = p1 < lam1                       # device -> gateway
    esc2 = esc1 & (p2 < lam2)              # gateway -> cloud
    stratum_A = esc2.copy()                # policy-driven escalation: propensity 1

    if explore is None:
        explore = np.zeros(n, dtype=bool)
    forced = explore & ~esc2               # exploration only adds escalations
    reach3 = esc2 | forced
    if cloud_up is None:
        cloud_up = np.ones(n, dtype=bool)
    completed3 = reach3 & cloud_up

    tier_out = np.where(esc2 & cloud_up, 3, np.where(esc1, 2, 1))
    tier_reached = np.where(completed3, 3, np.where(esc1 | forced, 2, 1))
    return {
        "tier_out": tier_out,
        "tier_reached": tier_reached,
        "observed_cloud": completed3,
        "stratum_A": stratum_A,
        "esc1": esc1, "esc2": esc2, "forced": forced,
        "outage_hit": reach3 & ~cloud_up,
    }
