"""Deployment-loop driver: CoDeC, seven baselines, and the ablation variants.

Strict information separation. At window t the controller may use
  * deferral scores of the current and past windows (the device and gateway compute
    them anyway, and they need no labels), and
  * cloud decisions on samples that were escalated -- either by the policy or by
    epsilon-exploration -- in PAST windows.
It may never use ground-truth stream labels, nor cloud decisions on samples that were
answered locally and not explored. Ground truth enters only through the evaluator, which
reports realised accuracy and realised risk after the fact.

Every buffered record is stored as raw per-sample state (deferral scores, per-tier
decisions, realised propensity), never as a pre-aggregated loss: a candidate threshold
changes which tier answers, so its loss must be recomputed from scratch on the buffer.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from .core import (ACIController, StratifiedRiskBound, certify_bonferroni,
                   certify_fst, conformal_pvalues, hb_ucb)
from .data import build_stream, load_dataset, tier_specs
from .tiers import CostModel, TieredSystem, simulate_cascade

LAM_GRID = np.array([0.0, 0.005, 0.01, 0.02, 0.05, 0.08, 0.12, 0.18,
                     0.25, 0.35, 0.5, 0.7, 1.01])

BASELINES = ["device_only", "cloud_only", "fixed_conf", "entropy", "learned_router",
             "cost_static", "offline_crc", "online_naive", "online_ipw"]
ABLATIONS = ["codec_no_explore", "codec_no_aci", "codec_bonferroni", "codec_rawconf",
             "codec_two_tier"]
METHODS = BASELINES + ["codec"] + ABLATIONS
ONLINE = {"online_naive", "online_ipw", "codec"} | set(ABLATIONS)

_MODEL_CACHE: dict = {}


# --------------------------------------------------------------------------- #
def get_system(ds, setting, model_seed, max_train=20000, window=500):
    key = (ds, setting, model_seed, window)
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]
    data = load_dataset(ds)
    stream = build_stream(ds, data, setting, seed=model_seed, window=window)
    sysm = TieredSystem(tier_specs(ds, data), CostModel(), seed=model_seed)
    sysm.fit(stream.X, stream.y, stream.idx_train[:max_train], stream.idx_cal)
    sysm.profile_latency(stream.X[stream.idx_cal])
    comp, ref = sysm.cost_table()
    out = dict(data=data, stream=stream, sysm=sysm, comp=comp, ref=ref,
               pre_order=sysm.precompute(stream.X, stream.idx_order),
               pre_stream=sysm.precompute(stream.X, stream.idx_stream))
    _MODEL_CACHE[key] = out
    return out


def _answers_from(y1, y2, y3, esc1, esc2):
    return np.where(esc2, y3, np.where(esc1, y2, y1))


def _cost_components(comp, ref, cost, tier_reached):
    e = comp["energy_mJ"][tier_reached - 1]
    u = comp["usd"][tier_reached - 1]
    l = comp["latency_ms"][tier_reached - 1]
    sc = (cost.w_energy * e / max(ref["energy_mJ"], 1e-12)
          + cost.w_money * u / max(ref["usd"], 1e-12)
          + cost.w_latency * l / max(ref["latency_ms"], 1e-12))
    return dict(energy_mJ=float(e.mean()), usd=float(u.mean()),
                latency_ms=float(l.mean()), scalar=float(sc.mean()))


def _cand_order(s1, s2, comp, ref, cost, grid=LAM_GRID):
    """Order candidates most-expensive-first using UNLABELLED scores only."""
    cands, costs = [], []
    for l1 in grid:
        for l2 in grid:
            esc1 = s1 < l1
            esc2 = esc1 & (s2 < l2)
            tr = np.where(esc2, 3, np.where(esc1, 2, 1))
            cands.append((float(l1), float(l2)))
            costs.append(_cost_components(comp, ref, cost, tr)["scalar"])
    order = np.argsort(-np.asarray(costs))
    return [cands[i] for i in order], {cands[i]: float(costs[i]) for i in order}


def _scores_for(method, pre, pv1, pv2, n_cls, router_p=None):
    """(score1, score2) on the [0,1] scale the threshold grid lives on."""
    if method == "fixed_conf":
        return pre["conf1"], pre["conf2"]
    if method == "entropy":
        lg = np.log(max(n_cls, 2))
        return 1.0 - pre["entropy1"] / lg, 1.0 - pre["entropy2"] / lg
    if method == "learned_router":
        return 1.0 - router_p, pre["conf2"]
    if method == "codec_rawconf":
        return pre["conf1"], pre["conf2"]
    return pv1, pv2


# --------------------------------------------------------------------------- #
def run_config(ds, setting, model_seed, stream_seed, alpha=0.05, delta=0.10,
               epsilon=0.10, outage_rate=0.0, budget=None, methods=None,
               eta=0.05, window=500, buffer_n=3000):
    methods = list(methods or METHODS)
    S = get_system(ds, setting, model_seed, window=window)
    sysm, stream, comp, ref = S["sysm"], S["stream"], S["comp"], S["ref"]
    cost = sysm.cost
    pre_s, pre_o = S["pre_stream"], S["pre_order"]
    y_true = stream.y[stream.idx_stream]
    n_cls = len(np.unique(stream.y))
    n = len(y_true)
    starts = list(range(0, n - window + 1, window))
    rng = np.random.default_rng(10_000 * (stream_seed + 1) + model_seed)

    # cloud reachability: two-state chain, mean outage run ~3 windows
    up = np.ones(n, dtype=bool)
    if outage_rate > 0:
        p_down, p_up, st = outage_rate / (3 * window), 1.0 / (3 * window), True
        for i in range(n):
            st = (rng.random() > p_down) if st else (rng.random() < p_up)
            up[i] = st

    pv1_s = conformal_pvalues(sysm.cal_scores[0], pre_s["score1"])
    pv2_s = conformal_pvalues(sysm.cal_scores[1], pre_s["score2"])
    pv1_o = conformal_pvalues(sysm.cal_scores[0], pre_o["score1"])
    pv2_o = conformal_pvalues(sysm.cal_scores[1], pre_o["score2"])
    y3_s = pre_s["yhat3"]

    # learned router trained on the offline split
    feat_o = np.column_stack([pre_o["conf1"], pre_o["entropy1"], pre_o["score1"], pv1_o])
    tgt_o = (pre_o["yhat1"] != pre_o["yhat3"]).astype(int)
    if tgt_o.sum() >= 5 and (1 - tgt_o).sum() >= 5:
        router = Pipeline([("sc", StandardScaler()),
                           ("lr", LogisticRegression(max_iter=1000))]).fit(feat_o, tgt_o)
        rp_o = router.predict_proba(feat_o)[:, 1]
        rp_s = router.predict_proba(np.column_stack(
            [pre_s["conf1"], pre_s["entropy1"], pre_s["score1"], pv1_s]))[:, 1]
    else:
        rp_o = np.zeros(len(tgt_o)); rp_s = np.zeros(n)

    # ---- offline threshold selection on the labelled offline split --------- #
    def offline_pick(s1, s2, certified=False):
        """Cheapest config whose offline fidelity risk passes (point estimate, or HB)."""
        best, best_c = None, np.inf
        for l1 in LAM_GRID:
            for l2 in LAM_GRID:
                esc1 = s1 < l1
                esc2 = esc1 & (s2 < l2)
                ans = _answers_from(pre_o["yhat1"], pre_o["yhat2"], pre_o["yhat3"],
                                    esc1, esc2)
                loss = (ans != pre_o["yhat3"]).astype(float)
                r = (hb_ucb(float(loss.mean()), loss.size, delta) if certified
                     else float(loss.mean()))
                if r <= alpha:
                    tr = np.where(esc2, 3, np.where(esc1, 2, 1))
                    c = _cost_components(comp, ref, cost, tr)["scalar"]
                    if c < best_c:
                        best, best_c = (float(l1), float(l2)), c
        return best or (1.01, 1.01)

    lam_init = {
        "device_only": (0.0, 0.0), "cloud_only": (1.01, 1.01),
        "fixed_conf": offline_pick(pre_o["conf1"], pre_o["conf2"]),
        "entropy": offline_pick(1 - pre_o["entropy1"] / np.log(max(n_cls, 2)),
                                1 - pre_o["entropy2"] / np.log(max(n_cls, 2))),
        "learned_router": offline_pick(1 - rp_o, pre_o["conf2"]),
        "cost_static": offline_pick(pv1_o, pv2_o),
        "offline_crc": offline_pick(pv1_o, pv2_o, certified=True),
    }

    wrows, rrows = [], []
    for method in methods:
        s1_all, s2_all = _scores_for(method, pre_s, pv1_s, pv2_s, n_cls, rp_s)
        two_tier = (method == "codec_two_tier")
        eps = 0.0 if method == "codec_no_explore" else (
            epsilon if method in ONLINE else 0.0)
        mode = {"online_naive": "naive", "online_ipw": "ipw"}.get(method, "stratified")
        rb = StratifiedRiskBound(epsilon=max(eps, 1e-6), delta=delta, mode=mode)
        aci = ACIController(alpha, eta=(0.0 if method == "codec_no_aci" else eta),
                            budget=budget)
        certify = certify_bonferroni if method == "codec_bonferroni" else certify_fst
        lam = lam_init.get(method, (1.01, 1.01))

        buf = {k: [] for k in ("s1", "s2", "y1", "y2", "y3", "pi1", "obs")}
        acc_h, cost_h, risk_h, viol_h, nsel = [], [], [], [], 0
        prev_s1 = prev_s2 = None

        for wi, s0 in enumerate(starts):
            sl = slice(s0, s0 + window)
            S1, S2 = s1_all[sl], s2_all[sl]
            expl = rng.random(window) < eps
            res = simulate_cascade(S1, S2, lam[0], lam[1], explore=expl, cloud_up=up[sl])
            ans = _answers_from(pre_s["yhat1"][sl], pre_s["yhat2"][sl], y3_s[sl],
                                res["esc1"], res["esc2"] & up[sl])
            fid = (ans != y3_s[sl]).astype(float)
            acc = float((ans == y_true[sl]).mean())
            cmp_ = _cost_components(comp, ref, cost, res["tier_reached"])
            rr = float(fid.mean())
            feasible = bool(res["outage_hit"].sum() == 0)
            budget_ok = (budget is None) or (cmp_["scalar"] <= budget)

            wrows.append(dict(
                dataset=ds, setting=setting, method=method, model_seed=model_seed,
                stream_seed=stream_seed, window=wi, alpha=alpha, delta=delta,
                epsilon=eps, outage_rate=outage_rate, budget=(budget or -1.0),
                acc=acc, fidelity_risk=rr, violation=int(rr > alpha),
                esc_gateway=float(res["esc1"].mean()),
                esc_cloud=float((res["esc2"] & up[sl]).mean()),
                explore_rate=float(res["forced"].mean()),
                bytes_per_sample=float(np.where(res["tier_reached"] >= 2,
                                                sysm.tiers[1].payload_bytes, 0).mean()),
                lam1=lam[0], lam2=lam[1], alpha_t=aci.alpha_t, debt=aci.debt,
                risk_est=float("nan"),
                feasible=int(feasible), budget_ok=int(budget_ok), **cmp_))
            acc_h.append(acc); cost_h.append(cmp_["scalar"])
            risk_h.append(rr); viol_h.append(int(rr > alpha))

            if method in ONLINE:
                obs = res["observed_cloud"]
                buf["s1"].append(S1); buf["s2"].append(S2)
                buf["y1"].append(pre_s["yhat1"][sl]); buf["y2"].append(pre_s["yhat2"][sl])
                buf["y3"].append(np.where(obs, y3_s[sl], -1))
                buf["pi1"].append(res["stratum_A"]); buf["obs"].append(obs)
                B = {k: np.concatenate(v)[-buffer_n:] for k, v in buf.items()}
                # The controller may NOT see the true window risk: that would need a
                # cloud decision for every sample. It tracks the stratified self-labelled
                # estimate instead, formed only from observed cloud decisions.
                mA = res["stratum_A"]
                oA, oN = mA & obs, (~mA) & obs
                pA = float(mA.mean())
                rr_hat = (pA * (float(fid[oA].mean()) if oA.any() else 0.0)
                          + (1 - pA) * (float(fid[oN].mean()) if oN.any() else 0.0))
                wrows[-1]["risk_est"] = rr_hat
                aci.update(rr_hat, cmp_["scalar"], feasible and budget_ok)

                if prev_s1 is not None and int(B["obs"].sum()) >= 40:
                    cands_t, cost_t = _cand_order(prev_s1, prev_s2, comp, ref, cost)
                    if two_tier:
                        seen, tmp = set(), []
                        for c in cands_t:
                            c2 = (c[0], 1.01)
                            if c2 not in seen:
                                seen.add(c2); tmp.append(c2)
                        cands_t = tmp

                    def risk_fn(cand, d=None, B=B):
                        l1, l2 = cand
                        e1 = B["s1"] < l1
                        e2 = e1 & (B["s2"] < l2)
                        a = _answers_from(B["y1"], B["y2"], B["y3"], e1, e2)
                        loss = (a != B["y3"]).astype(float)
                        m1 = B["pi1"]; mo = B["obs"]
                        return rb.bound([
                            {"losses": loss[m1 & mo], "n_pop": int(m1.sum()),
                             "propensity": 1.0},
                            {"losses": loss[(~m1) & mo], "n_pop": int((~m1).sum()),
                             "propensity": max(eps, 1e-6)},
                        ], delta=d)

                    sel, certified, _ = certify(cands_t, risk_fn, aci.alpha_t, delta)
                    if budget is not None and certified:
                        afford = [c for c in certified if cost_t.get(c, np.inf) <= budget]
                        sel = afford[-1] if afford else certified[-1]
                    if sel is not None:
                        lam = sel; nsel += 1
            prev_s1, prev_s2 = S1, S2

        rrows.append(dict(
            dataset=ds, setting=setting, method=method, model_seed=model_seed,
            stream_seed=stream_seed, alpha=alpha, delta=delta, epsilon=eps,
            outage_rate=outage_rate, budget=(budget or -1.0), n_windows=len(starts),
            acc=float(np.mean(acc_h)), fidelity_risk=float(np.mean(risk_h)),
            violation_rate=float(np.mean(viol_h)), cost_scalar=float(np.mean(cost_h)),
            energy_mJ=float(np.mean([w["energy_mJ"] for w in wrows[-len(starts):]])),
            latency_ms=float(np.mean([w["latency_ms"] for w in wrows[-len(starts):]])),
            usd=float(np.mean([w["usd"] for w in wrows[-len(starts):]])),
            esc_cloud=float(np.mean([w["esc_cloud"] for w in wrows[-len(starts):]])),
            n_reselect=nsel, final_debt=float(aci.debt)))
    return wrows, rrows
