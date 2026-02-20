<p align="center">
  <h1 align="center">GOD MODE</h1>
  <p align="center">
    <strong>The Theoretical Upper Bound of ETF Trading</strong><br>
    <em>A benchmark of perfect foresight performance — how fast is the speed of light in your trading environment?</em>
  </p>
</p>

<p align="center">
  <a href="#philosophy">Philosophy</a> &nbsp;&bull;&nbsp;
  <a href="#key-results">Key Results</a> &nbsp;&bull;&nbsp;
  <a href="#how-it-works">How It Works</a> &nbsp;&bull;&nbsp;
  <a href="#quickstart">Quickstart</a> &nbsp;&bull;&nbsp;
  <a href="#project-structure">Project Structure</a> &nbsp;&bull;&nbsp;
  <a href="#license">License</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.8%2B-blue?logo=python&logoColor=white" alt="Python 3.8+">
  <img src="https://img.shields.io/badge/optimization-Optuna-blueviolet?logo=data:image/svg+xml;base64," alt="Optuna">
  <img src="https://img.shields.io/badge/analytics-QuantStats-orange" alt="QuantStats">
  <img src="https://img.shields.io/badge/license-MIT-green" alt="MIT License">
</p>

---

## Philosophy

In algorithmic trading, there is a critical distinction between **captured alpha** and **available alpha**. Most strategies capture a tiny fraction of the market's potential — but how tiny?

God Mode answers a deceptively simple question:

> *If we had a perfect oracle that could predict future prices with 100% accuracy, what is the mathematical limit of profit?*

This is the **speed of light** for alpha generation. No real strategy can ever exceed it. By establishing this ceiling, we gain:

| Purpose | Insight |
|---|---|
| **Reality Check** | If a real strategy claims performance anywhere near God Mode, it is almost certainly overfitted or flawed. |
| **Opportunity Cost** | Quantifies how much money is left on the table due to uncertainty and friction. |
| **System Stress Test** | Verifies that the execution engine handles extreme growth scenarios without numerical overflow or logic errors. |

---

## Key Results

Running from **2016-01-01** to **2025-09-26** across a universe of **15 ETFs** with **0.1% transaction costs**:

| Metric | Value |
|---|---|
| Initial Capital | $100,000 |
| Final Capital | **$10,345,152,572** |
| Total Return | **10,345,053%** |
| Sharpe Ratio | 5.85 |
| Sortino Ratio | 9.29 |
| Max Drawdown | -20.38% |
| Optimal Rebalance Window | 4 days |
| Optimal Portfolio Concentration | 1 asset |

**The takeaway:** we pay roughly **99.999%** of potential profit as the price for not knowing the future.

---

## How It Works

### The Oracle Function

At each rebalancing step *t*, God Mode peers into the future:

$$R_{i,\;t \to t+\Delta t} = \frac{P_{i,\;t+\Delta t}}{P_{i,\;t}} - 1$$

It then solves for the allocation that maximizes portfolio return over the next window:

$$\mathbf{w}^*_t = \arg\max_{\mathbf{w}} \sum_{i=1}^{N} w_i \cdot R_{i,\;t \to t+\Delta t}$$

subject to weights summing to 1 and individual weight bounds.

### Execution Realism

Despite using future data for *decisions*, the execution is fully realistic:

1. **Decision** at market close on day *t* (using future prices as the oracle)
2. **Execution** at Market-On-Open (MOO) on day *t+1*
3. **Returns** measured from open to close on day *t+1*

This ensures execution parity with real strategies — the only advantage God Mode has is *knowing which asset to pick*, not *getting a better price*.

### Parameter Optimization

The two hyperparameters were optimized via **Optuna** (2,400 trials, 12 parallel processes):

| Parameter | Search Space | Optimal Value |
|---|---|---|
| Rebalance Window | 1–60 days | **4 days** |
| Portfolio Concentration | 1–5 assets | **1 asset** |

The composite objective (60% Total Return + 20% Sharpe + 20% Sortino) converged on maximum concentration with a 4-day holding period — short enough to capture regime shifts, long enough to offset transaction cost drag.

---

## ETF Universe

The strategy dynamically switches across asset classes, always selecting the optimal instrument for each window:

| Category | ETFs |
|---|---|
| US Equity (Large Cap) | SPY, QQQ, IVV, VOO, VTI |
| US Equity (Small Cap / Dividend) | IWM, SCHD |
| International Equity | VT, VXUS |
| Fixed Income | BND, TLT, SGOV |
| Commodities | GLD, SLV |
| Crypto | IBIT |

---

## Quickstart

### Prerequisites

- Python 3.8+

### Installation

```bash
git clone https://github.com/yourusername/godmode.git
cd godmode
pip install -r requirements.txt
```

### Run the Simulation

```bash
python run_godmode_etf_v01.py
```

### Run Parameter Optimization

```bash
python etf_godmode_params_optimize.py
```

### Outputs

After execution, find results in `reports/god_mode_benchmark/`:
- Performance charts (PNG)
- Interactive QuantStats HTML report
- Multi-sheet Excel report (executive summary, daily performance, transaction log, holdings, risk metrics)
- Weight evolution visualizations (heatmaps, stacked area charts, line charts)

---

## Project Structure

```
godmode/
├── run_godmode_etf_v01.py            # Main simulation — runs the God Mode backtest
├── etf_godmode_params_optimize.py    # Optuna-based hyperparameter optimization
├── quantstats_reporter.py            # QuantStats performance analytics integration
├── etf_weight_visualizer.py          # Portfolio weight charts and heatmaps
├── professional_excel_reporter.py    # Multi-sheet Excel report generator
├── god_mode_report.pdf               # Detailed methodology whitepaper
├── data_etf/                         # Historical ETF price data (OHLCV CSVs)
│   ├── SPY.csv
│   ├── QQQ.csv
│   └── ... (15 ETFs)
└── reports/
    └── god_mode_benchmark/           # Generated reports and visualizations
```

---

## Transaction Cost Model

God Mode doesn't live in a frictionless vacuum. Every trade incurs **0.1% cost** on dollars traded:

$$\text{Cost}_t = \tau \times \text{Turnover}_t \times \text{Portfolio Value}_t$$

where $\tau = 0.001$. With ~63 trades/year at 200% turnover each, annual cost drag is ~31.5% — enormous for a real strategy, negligible against the exponential returns of perfect foresight.

The 0.1% rate conservatively aggregates bid-ask spreads (0.02–0.05%), market impact (0.05–0.15%), and slippage.

---

## Citation

If you use this benchmark in your research:

```bibtex
@misc{khadamolama2026godmode,
  author       = {KhademOlama, Ehsan},
  title        = {God Mode: The Theoretical Upper Bound of ETF Trading},
  year         = {2026},
  url          = {https://github.com/yourusername/godmode}
}
```

---

## License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.

---

<p align="center">
  <sub>Built for benchmarking, not for trading. If your strategy beats God Mode, check for bugs.</sub>
</p>
