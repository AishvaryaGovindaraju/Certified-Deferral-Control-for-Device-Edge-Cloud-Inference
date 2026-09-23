# CoDeC — Certified Deferral Control for Device–Edge–Cloud Inference

Submission package for IEEE Internet of Things Journal. Everything quoted in the paper is
computed by the code here and read from `results/*.csv`; no number in the text is typed by
hand.

## Contents

| Path | What it is |
|---|---|
| `main.tex` | IEEEtran journal source (`\documentclass[journal]{IEEEtran}`) |
| `refs.bib` | 61 references, every entry resolved against OpenAlex; 58 carry DOIs |
| `CoDeC_IoTJ.pdf` | Rendered double-column preview (see "Compiling" below) |
| `codec/core.py` | Risk-control core: stratified bound, fixed-sequence certification, tracking controller |
| `codec/tiers.py` | Tier models, cost model, vectorised cascade simulator |
| `codec/data.py` | Dataset loading, semantics-grounded tier feature splits, stream construction |
| `codec/experiment.py` | Deployment loop: CoDeC, nine baselines, five ablations |
| `scripts/run_main.py` | Main sweep → `results/main_*.csv`, `results/rtiot_*.csv` |
| `scripts/run_sweeps.py` | Sensitivity sweeps → `results/sweep_*.csv` |
| `scripts/profile_tiers.py` | Table I inputs → `results/tier_profile.csv`, `results/cost_model.json` |
| `scripts/make_figures.py` | All figures + `results/paper_numbers.json` |
| `scripts/paper_content.py` | Single source of the paper prose (both emitters read it) |
| `scripts/make_paper.py` | Emits `main.tex` |
| `scripts/render_pdf.py` | Emits `CoDeC_IoTJ.pdf` |

## Compiling the submission

The sandbox this was produced in has no TeX distribution, so `CoDeC_IoTJ.pdf` is rendered
directly with ReportLab at IEEE journal geometry (US Letter, 0.625 in side margins, two
3.5 in columns, 0.25 in gutter). It reads the same prose and the same numbers as
`make_paper.py`, so the preview and the source cannot disagree — but it is **not** a
LaTeX build. For the camera-ready:

```
latexmk -pdf main.tex        # or upload main.tex + refs.bib + figs/ to Overleaf
```

`main.tex` requires `IEEEtran.cls` and `IEEEtran.bst` (both on CTAN and preinstalled on
Overleaf). Figures are referenced as `figs/*.pdf`, which are written alongside the PNGs.
Expect roughly 8–9 pages in the LaTeX build (≈4.9k words of prose, 6 figures, 3 tables);
the ReportLab preview paginates more loosely and runs longer.

## Reproducing the results

Python 3.13; `numpy scipy pandas scikit-learn matplotlib reportlab`.

```
python scripts/fetch_data.py         # UCI HAR, Gas Sensor Array Drift, RT-IoT2022
python scripts/run_main.py  --datasets har,gas --settings iid,shift        --model-seeds 4 --stream-seeds 5
python scripts/run_main.py  --datasets rtiot   --settings iid,priorshift   --model-seeds 2 --stream-seeds 2 --out-prefix results/rtiot
python scripts/run_sweeps.py
python scripts/profile_tiers.py
python scripts/make_figures.py
python scripts/make_paper.py && python scripts/render_pdf.py
```

Wall-clock on 16 x86 cores, single-threaded per configuration: main sweep ≈ 75 min,
RT-IoT2022 ≈ 15 min, sensitivity sweeps ≈ 35 min. Everything is CPU-only; no GPU is used
anywhere, including by the "cloud" tier.

## Datasets

All three come from the UCI Machine Learning Repository and are downloaded by
`scripts/fetch_data.py`:

- **UCI HAR** (id 240) — 10,299 windows, 561 features, 6 activities, 30 subjects. Shift
  setting holds out subjects.
- **Gas Sensor Array Drift** (id 224) — 13,910 samples, 128 features, 6 analytes, 10
  batches over 36 months. Shift setting trains on batches 1–5 and deploys on 6–10 in
  temporal order. This is the only genuinely long-horizon drift in the paper.
- **RT-IoT2022** (id 942) — IoT intrusion traffic, 12 classes, subsampled to ≤6000 flows
  per class. **The capture is ordered by attack type, not chronologically**, so a
  positional split would confront the stream with unseen classes. We therefore use it for
  the exchangeable regime plus a *constructed* prior-shift schedule, and label that shift
  synthetic everywhere it appears.

An IoT botnet capture (N-BaIoT, id 442) was also intended. The UCI static endpoint serves
that ~1 GB archive with chunked transfer-encoding and no `Range` support, so a stalled
transfer cannot resume; four attempts failed and it was dropped rather than reported
partially.

## What is measured and what is modelled

Stated in the paper's setup section and repeated here because it bounds every cost claim.

**Measured**: feature counts, model parameter counts, multiply-accumulates, payload bytes
(actual float32 serialisation), realised accuracy, realised fidelity risk, escalation
rates, certification counts.

**Modelled**: energy (per-MAC and per-byte coefficients from the literature), transport
latency (payload / goodput + RTT), compute latency (MACs / per-tier throughput), and cloud
price per inference. Single-sample wallclock on x86 is dominated by framework dispatch and
does not order with model size, so it is reported only as a batch-amortised cross-check,
never as a device latency. **No physical IoT testbed was used.** Absolute energy and
latency figures are therefore parametric; the comparisons between methods hold under any
coefficient choice because all methods are costed with the same model.

## Honest limitations

These are in the paper; they are listed here so they are not missed.

1. **The controller exceeds its tolerated violation rate on the harshest drift.** On the
   gas array, 0.19 of windows violate against δ = 0.10. The buffer sweep localises this to
   staleness, not to the certificate.
2. **Our default buffer length was not the best available.** At 2000 records the violation
   rate is roughly half that at the 3000 used throughout the main experiments, at
   essentially the same cost. The reported CoDeC numbers are conservative in this
   parameter.
3. **Under exchangeability the controller is unnecessary** — offline conformal risk control
   is both valid and cheaper there, and the paper says so.
4. **Inverse-propensity weighting is safer than CoDeC on every stream**, just more
   expensive. What the stratified estimator buys is price (≈8% lower cost at the same
   certified level), not the safe/unsafe distinction.
5. **The cloud tier is the reference, not ground truth.** Proposition 1 brackets task risk
   by cloud error plus certified fidelity risk; if the cloud tier is poor the guarantee
   still holds and simply promises less.
6. **Assumption diagnostics were not run** (no formal tests of the concentration
   assumptions beyond the direct coverage audit in Fig. 2).

