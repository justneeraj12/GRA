# GRA 

A compact research codebase for studying classical localization algorithms (RSSI, TOA, TDOA, AOA) and their sensitivity to measurement noise. This repository contains simulation scripts, a Monte Carlo profiling engine, and plotting utilities used to produce the figures in the accompanying literature review.

## Quick overview

- Objective: compare localization accuracy of common methods under varying noise levels and anchor geometries.
- Methods implemented: RSSI (log-distance), TOA, TDOA, and AOA.
- Outputs: per-method RMSE vs. noise, sample trajectory/heatmap plots, and simple demonstrator scripts.

---

## Highlights

- Clean, well-commented Python scripts in `SIGNAL_MODELLING/`.
- Monte Carlo engine to measure Root Mean Squared Error (RMSE) across noise sweeps.
- Ready-to-run demos: `RSSI_modeling.py`, `TOA_modeling.py`, `TDOA_modeling.py`, `AOA_modeling.py`.

## Mathematical models (short)

This section summarizes the measurement models used in the code. Formulas are written in KaTeX-compatible notation.

### 1) RSSI — Log-distance path loss

We model received signal strength (in dBm) with the log-distance path-loss model:

$$
\mathrm{RSSI}(d) = P_t - 10 n \log_{10}\left(\frac{d}{d_0}\right) + X_\sigma
$$

where
- $P_t$ is the reference transmit power at distance $d_0$,
- $n$ is the path-loss exponent,
- $d$ is the transmitter–receiver distance,
- $X_\sigma \sim \mathcal{N}(0,\sigma^2)$ is the shadowing/noise term.

Localization with RSSI is typically solved by converting RSSI to range estimates and then using non-linear least squares.

### 2) TOA — Time of Arrival

Time-of-arrival gives a direct range estimate assuming known clock alignment:

$$
t_i = t_\text{tx} + \frac{\|p - a_i\|}{c} + w_i
$$

where $a_i$ is the i-th anchor position, $p$ is the unknown target position, $c$ is propagation speed, and $w_i$ is additive timing noise (often modeled Gaussian).

Converting TOA to range: $r_i = c (t_i - t_\text{tx})$.

### 3) TDOA — Time Difference of Arrival

Subtracting two TOA measurements cancels the transmit time $t_\text{tx}$:

$$
\Delta t_{ij} = t_i - t_j = \frac{\|p-a_i\| - \|p-a_j\|}{c} + (w_i - w_j)
$$

TDOA yields hyperbolic loci and commonly requires a reference anchor.

### 4) AOA — Angle of Arrival

AOA measures the bearing from an anchor to the target. For a noisy bearing measurement $\theta_i$:

$$
\heta_i = \operatorname{atan2}(p_y - a_{i,y},\; p_x - a_{i,x}) + \eta_i
$$

where $\eta_i$ is angular noise (e.g., wrapped Gaussian).

---

## Error metric

We use Root Mean Squared Error (RMSE) over Monte Carlo runs to evaluate estimator performance:

$$
\mathrm{RMSE} = \sqrt{\frac{1}{N}\sum_{k=1}^N \|\hat{p}^{(k)} - p_{\text{true}}\|_2^2}
$$

where $\hat{p}^{(k)}$ is the estimated position in trial $k$ and $N$ is the number of Monte Carlo runs.

## Quick start (local)

1. Create and activate a virtual environment (recommended):

```bash
python3 -m venv venv
source venv/bin/activate
```

2. Install minimal dependencies:

```bash
pip install numpy scipy matplotlib
```

3. Run the main profiling engine (example):

```bash
python SIGNAL_MODELLING/ADDED_NOISE.py
```

4. Run individual demos:

```bash
python SIGNAL_MODELLING/RSSI_modeling.py
python SIGNAL_MODELLING/TOA_modeling.py
python SIGNAL_MODELLING/TDOA_modeling.py
python SIGNAL_MODELLING/AOA_modeling.py
```

Notes:
- If you get missing-package errors, install the reported package via pip. The scripts use only standard scientific Python packages.
- For long Monte Carlo runs, lower `NUM_RUNS` or `N_STEPS` in `SIGNAL_MODELLING/ADDED_NOISE.py`.

## File structure

- `SIGNAL_MODELLING/`
	- `ADDED_NOISE.py` — Monte Carlo profiler and plotting utilities (central engine).
	- `RSSI_modeling.py`, `TOA_modeling.py`, `TDOA_modeling.py`, `AOA_modeling.py` — single-method demos.

## Practical tips & edge cases

- RSSI: avoid taking log of zero — ensure minimum distance clamping.
- TOA/TDOA: watch for clock offset — TDOA cancels transmit time but requires a reference anchor.
- AOA: wrap angles and use circular statistics for averaging.
- Anchors geometry: poor anchor placement (collinear or clustered) degrades results — include geometry-aware experiments.

## Reproducibility

- Set the random seed inside scripts if you want repeatable experiments (search for `np.random.seed` in `SIGNAL_MODELLING/`).

## Suggested experiments (for your report)

1. RMSE vs. noise variance for each method (plot all methods on one figure).
2. Effect of anchor geometry: sample two geometries (good vs. bad) and compare RMSE.
3. Runtime vs. number of Monte Carlo trials (profiling for reproducibility).

## Contribution & license

This code is provided for academic research. Feel free to reuse and adapt; please cite or acknowledge if used in published work.

Last updated: October 30, 2025

