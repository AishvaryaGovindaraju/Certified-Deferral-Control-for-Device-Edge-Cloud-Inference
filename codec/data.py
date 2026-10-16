"""Dataset loading, semantics-grounded tier feature splits, and stream construction.

Three real IoT datasets from the UCI ML Repository:

  har     UCI HAR (Anguita et al., 2013). 10299 windows, 561 features, 6 activities,
          30 subjects. Natural shift: across subjects.
  gas     Gas Sensor Array Drift (Vergara et al., 2012). 13910 samples, 128 features,
          6 analytes, 10 batches spanning 36 months. Natural shift: real sensor drift
          over time -- the reason this dataset exists.
  nbaiot  N-BaIoT (Meidan et al., 2018). Real traffic from 9 commercial IoT devices,
          115 features, benign plus BASHLITE/Mirai attack classes. Natural shift:
          across device types and at attack onset.

Tier feature splits follow each dataset's own feature construction, so the device tier
is cheap for a reason rather than by fiat:

  har     T1: time-domain mean/std of body acc, gravity acc and body gyro (18 features,
              plain windowed statistics).
          T2: all time-domain features (no FFT required).
          T3: all 561, including the frequency-domain block.
  gas     T1: the 16 steady-state Delta-R features, one per sensor.
          T2: Delta-R plus normalised |Delta-R| (32).
          T3: all 128, including the exponential-moving-average transient features.
  nbaiot  T1: the shortest decay window only (L0.01).
          T2: the two shortest windows (L0.01, L0.1).
          T3: all five decay windows (115).
"""
from __future__ import annotations

import io
import json
import os
import zipfile
from dataclasses import dataclass

import numpy as np

from .tiers import TierSpec

__all__ = ["load_dataset", "tier_specs", "build_stream", "Stream", "DATASETS"]

DATASETS = ("har", "gas", "rtiot")
PROC = os.path.join("data", "proc")


# --------------------------------------------------------------------------- #
@dataclass
class Stream:
    name: str
    setting: str
    X: np.ndarray
    y: np.ndarray
    idx_train: np.ndarray
    idx_cal: np.ndarray
    idx_order: np.ndarray
    idx_stream: np.ndarray
    window: int
    group: np.ndarray            # subject / batch / device id per sample
    group_name: str


def _har_feature_names() -> list[str]:
    cache = os.path.join(PROC, "har_features.json")
    if os.path.exists(cache):
        return json.load(open(cache))
    with zipfile.ZipFile(os.path.join("data", "raw", "har.zip")) as outer:
        inner = [n for n in outer.namelist() if n.lower().endswith(".zip")][0]
        with zipfile.ZipFile(io.BytesIO(outer.read(inner))) as z:
            txt = z.read("UCI HAR Dataset/features.txt").decode("utf-8", "ignore")
    names = [ln.strip().split(" ", 1)[1] for ln in txt.strip().splitlines()]
    json.dump(names, open(cache, "w"))
    return names


def load_dataset(name: str) -> dict:
    """Return {X, y, group, group_name, classes, feature_names}."""
    if name == "har":
        d = np.load(os.path.join(PROC, "har.npz"), allow_pickle=True)
        return {"X": d["X"], "y": d["y"], "group": d["subject"], "group_name": "subject",
                "classes": list(d["classes"]), "feature_names": _har_feature_names()}
    if name == "gas":
        d = np.load(os.path.join(PROC, "gas.npz"), allow_pickle=True)
        Xg = d["X"].astype(np.float64)
        Xg = np.sign(Xg) * np.log1p(np.abs(Xg))   # features span many decades
        return {"X": Xg, "y": d["y"], "group": d["batch"], "group_name": "batch",
                "classes": list(d["classes"]),
                "feature_names": [f"s{i // 8 + 1}_f{i % 8 + 1}" for i in range(128)]}
    if name == "rtiot":
        d = np.load(os.path.join(PROC, "rtiot.npz"), allow_pickle=True)
        return {"X": d["X"], "y": d["y"], "group": d["y"], "group_name": "attack_type",
                "classes": [str(c) for c in d["classes"]],
                "feature_names": [str(c) for c in d["feature_names"]]}
    raise ValueError(name)


# --------------------------------------------------------------------------- #
def tier_specs(name: str, data: dict) -> list[TierSpec]:
    fn = data["feature_names"]
    d_all = data["X"].shape[1]
    if name == "har":
        want = ("tBodyAcc-", "tGravityAcc-", "tBodyGyro-")
        t1 = [i for i, f in enumerate(fn)
              if f.startswith(want) and ("mean()" in f or "std()" in f)
              and f.rstrip().endswith(("-X", "-Y", "-Z"))]
        base = ("tBodyAcc-", "tGravityAcc-", "tBodyGyro-")
        t2 = [i for i, f in enumerate(fn) if f.startswith(base)]
        t3 = list(range(d_all))
        # escalation payload: the raw inertial window the gateway needs (9 x 128 floats)
        pay = [0, 9 * 128, len(t3)]
    elif name == "gas":
        t1 = [i for i in range(d_all) if i % 8 == 0]
        t2 = [i for i in range(d_all) if i % 8 in (0, 1)]
        t3 = list(range(d_all))
        pay = [0, len(t3), len(t3)]
    elif name == "rtiot":
        # T1: counters a NIC or flow table exposes without inspecting headers or payload
        basic = ("flow_duration", "fwd_pkts_tot", "bwd_pkts_tot", "fwd_data_pkts_tot",
                 "bwd_data_pkts_tot", "fwd_pkts_per_sec", "bwd_pkts_per_sec",
                 "flow_pkts_per_sec", "down_up_ratio", "id.orig_p", "id.resp_p")
        t1 = [i for i, f in enumerate(fn) if f in basic]
        # T2: adds header-size and TCP-flag aggregates plus protocol/service identity
        t2 = sorted(set(t1) | {i for i, f in enumerate(fn)
                               if ("header_size" in f) or ("flag_count" in f)
                               or f.startswith(("proto_", "svc_"))})
        # T3: adds payload, inter-arrival, subflow, bulk, active/idle and window statistics
        t3 = list(range(d_all))
        pay = [0, len(t3), len(t3)]
    else:
        raise ValueError(name)
    for nm, idx in (("T1", t1), ("T2", t2), ("T3", t3)):
        if len(idx) == 0:
            raise RuntimeError(f"{name}/{nm}: empty feature split")
    return [
        TierSpec("device", np.array(sorted(t1)), (8,), pay[0]),
        TierSpec("gateway", np.array(sorted(t2)), (64,), pay[1]),
        TierSpec("cloud", np.array(sorted(t3)), (256, 128), pay[2]),
    ]


# --------------------------------------------------------------------------- #
def build_stream(name: str, data: dict, setting: str, seed: int = 0,
                 window: int = 400) -> Stream:
    """Split into train / offline-calibration / cost-order / deployment stream.

    setting="shift" keeps the dataset's natural non-stationarity: the stream is ordered
    by the grouping variable (subject, batch, device) and the tiers never see the
    groups they will be deployed on. setting="iid" shuffles everything, giving the
    exchangeable regime in which the offline guarantee is supposed to hold.
    """
    X, y, g = data["X"], data["y"], data["group"]
    rng = np.random.default_rng(seed)
    groups = np.unique(g)

    if setting == "iid":
        perm = rng.permutation(len(X))
        n = len(perm)
        n_tr, n_cal, n_ord = int(0.25 * n), int(0.15 * n), int(0.10 * n)
        idx_train = perm[:n_tr]
        idx_cal = perm[n_tr:n_tr + n_cal]
        idx_order = perm[n_tr + n_cal:n_tr + n_cal + n_ord]
        idx_stream = perm[n_tr + n_cal + n_ord:]
    elif setting == "shift":
        if name == "gas":
            tr_groups = groups[:5]            # batches 1-5 (earliest in time)
            st_groups = groups[5:]            # batches 6-10, in temporal order
        elif name == "har":
            tr_groups = groups[:12]
            st_groups = groups[12:]
        else:
            tr_groups = groups[:3]
            st_groups = groups[3:]
        in_tr = np.isin(g, tr_groups)
        pool = np.flatnonzero(in_tr)
        pool = rng.permutation(pool)
        n_tr = int(0.55 * len(pool)); n_cal = int(0.30 * len(pool))
        idx_train = pool[:n_tr]
        idx_cal = pool[n_tr:n_tr + n_cal]
        idx_order = pool[n_tr + n_cal:]
        # deployment stream: held-out groups, in group order, shuffled within group
        parts = []
        for gg in st_groups:
            ii = np.flatnonzero(g == gg)
            parts.append(rng.permutation(ii))
        idx_stream = np.concatenate(parts)
    elif setting == "priorshift":
        # The public RT-IoT2022 capture is organised by attack type rather than
        # chronologically, so a positional split would hand the stream classes the tiers
        # never trained on. We instead CONSTRUCT a prior-shift schedule: training is
        # exchangeable and sees every class, while the deployment stream's class mixture
        # rotates from phase to phase, as an attack campaign's does. This shift is
        # synthetic by construction and labelled as such throughout.
        perm = rng.permutation(len(X))
        n = len(perm)
        n_tr, n_cal, n_ord = int(0.25 * n), int(0.15 * n), int(0.10 * n)
        idx_train = perm[:n_tr]
        idx_cal = perm[n_tr:n_tr + n_cal]
        idx_order = perm[n_tr + n_cal:n_tr + n_cal + n_ord]
        pool = perm[n_tr + n_cal + n_ord:]
        cls = np.unique(y)
        avail = {int(c): list(pool[y[pool] == c]) for c in cls}
        # Six campaign phases of two deployment windows each: enough rotation to exercise
        # the controller while keeping the per-run cost of a 12-class problem tractable.
        n_phase = 6
        per_phase = 2 * window
        parts = []
        for p in range(n_phase):
            focus = {int(cls[(2 * p + j) % len(cls)]) for j in range(3)}
            w = np.array([(0.70 / len(focus)) if int(c) in focus
                          else (0.30 / max(len(cls) - len(focus), 1)) for c in cls])
            want = np.floor(w / w.sum() * per_phase).astype(int)
            take = {int(c): min(want[i], len(avail[int(c)])) for i, c in enumerate(cls)}
            short = per_phase - sum(take.values())
            while short > 0:
                room = [(len(avail[int(c)]) - take[int(c)], int(c)) for c in cls]
                room = [r for r in room if r[0] > 0]
                if not room:
                    break
                room.sort(reverse=True)
                give = min(short, room[0][0])
                take[room[0][1]] += give
                short -= give
            chosen = []
            for c in cls:
                k = take[int(c)]
                if k:
                    chosen += avail[int(c)][:k]
                    avail[int(c)] = avail[int(c)][k:]
            parts.append(rng.permutation(np.asarray(chosen, dtype=int)))
        idx_stream = np.concatenate(parts)
    else:
        raise ValueError(setting)

    return Stream(name, setting, X, y, idx_train, idx_cal, idx_order, idx_stream,
                  window, g, data["group_name"])
