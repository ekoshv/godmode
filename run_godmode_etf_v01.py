"""
Bare-Metal ETF "GOD MODE" Simulator (VERSION 01)

This script implements a THEORETICAL "GOD MODE" strategy that sees the future.
It serves as a benchmark for the MAXIMUM POSSIBLE performance achievable with
perfect foresight.

MECHANISM:
1. At each rebalance point, look FORWARD by the rebalance window.
2. Calculate the future return of every ETF over that specific holding period.
3. Select the Top N performing ETFs for that upcoming period.
4. Invest 100% (distributed among Top N) into those future winners.

THIS IS FOR BENCHMARKING PURPOSES ONLY.
IT CONTAINS INTENTIONAL LOOK-AHEAD BIAS.
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
import quantstats as qs
from typing import Optional, Dict, Any

from etf_weight_visualizer import ETFWeightVisualizer
from quantstats_reporter import QuantStatsReporter
from professional_excel_reporter import ProfessionalExcelReporter

# --- Configuration ---
START_DATE = "2016-01-01"
END_DATE = "2025-09-26"
ETF_DATA_FOLDER = "data_etf"
ETF_UNIVERSE = [
    'SPY', 'QQQ', 'IVV', 'VOO', 'VTI', 'IWM', 'VT', 'VXUS',
    'BND', 'TLT', 'GLD', 'SLV', 'IBIT', 'SCHD', 'SGOV'
]

# --- Quality ETF Universe (Benchmark) ---
QUALITY_ETFS = ['SPY', 'QQQ', 'IVV', 'VOO', 'VTI', 'SCHD', 'GLD']

# --- Strategy Parameters ---
REBALANCE_WINDOW = 4       # How often we switch to the new best assets
TOP_N_PORTFOLIO = 1         # Concentrate on the absolute best winners (God knows best)
MAX_WEIGHT = 1.0            # God doesn't need diversification limits (set to 1.0 or lower if desired)

# --- Capital Deployment ---
USE_COMPOUND_MODE = True    # Reinvest profits
INITIAL_CAPITAL = 100000.0

# --- Transaction Costs ---
# Even God has to pay the broker
ENABLE_TRANSACTION_COSTS = True
TRANSACTION_COST_RATE = 0.001  # 0.1%

# --- Output ---
REPORTS_DIR = "reports/god_mode_benchmark"
OUTPUT_FILENAME = "god_mode_performance.png"
DEBUG_MODE = True


# --- Data Loading Functions ---

def load_all_etf_data(data_folder: str, etf_universe: list) -> dict:
    """Load all ETF data from CSV files."""
    etf_data = {}
    print("Loading ETF data...")
    for etf_name in etf_universe:
        csv_path = os.path.join(data_folder, f"{etf_name}.csv")
        if not os.path.exists(csv_path):
            print(f"  [WARNING] {etf_name}: File not found, skipping")
            continue
        try:
            df = pd.read_csv(csv_path)
            df.columns = df.columns.str.strip()
            date_col = next((col for col in ['Date', 'date', 'timestamp'] if col in df.columns), None)
            if not date_col:
                continue
            
            df['date'] = pd.to_datetime(df[date_col], errors='coerce', utc=True).dt.tz_localize(None)
            df.dropna(subset=['date'], inplace=True)

            close_col = next((col for col in ['Close', 'close'] if col in df.columns), None)
            if not close_col:
                continue
            
            df['close'] = df[close_col]
            
            open_col = next((col for col in ['Open', 'open'] if col in df.columns), None)
            df['open'] = df[open_col] if open_col else df['close']
            
            df = df.sort_values('date').drop_duplicates(subset=['date'], keep='last')
            etf_data[etf_name] = df[['date', 'close', 'open']].set_index('date')
            print(f"  [OK] {etf_name}: {len(df)} days loaded")
        except Exception as e:
            print(f"  [ERROR] {etf_name}: {str(e)}")
    return etf_data


def create_price_dataframe(etf_data: dict, start_date: str, end_date: str) -> pd.DataFrame:
    """Combine all ETF prices into a single DataFrame."""
    all_close = {name: df['close'] for name, df in etf_data.items()}
    close_df = pd.DataFrame(all_close).sort_index()
    close_df = close_df.loc[start_date:end_date]
    close_df = close_df.dropna(how='all')
    return close_df


def create_open_dataframe(etf_data: dict, start_date: str, end_date: str) -> pd.DataFrame:
    """Combine all ETF open prices."""
    all_open = {name: (df['open'] if 'open' in df.columns else df['close']) for name, df in etf_data.items()}
    open_df = pd.DataFrame(all_open).sort_index()
    open_df = open_df.loc[start_date:end_date].dropna(how='all')
    return open_df


# --- Core Logic ---

def run_god_mode_strategy(price_df: pd.DataFrame, initial_capital: float, 
                         excel_reporter: ProfessionalExcelReporter = None,
                         open_df: pd.DataFrame = None) -> tuple:
    """
    Execute GOD MODE Strategy.
    Uses future prices to determine optimal allocation for the next period.
    """
    use_moo = open_df is not None and not open_df.equals(price_df)
    execution_prices = open_df if use_moo else price_df
    
    daily_returns = price_df.pct_change()
    equity = [initial_capital]
    weights = pd.Series(0.0, index=price_df.columns)
    
    # We start 'rebalancing' from day 0 because we know the future immediately
    last_rebalance_day = -REBALANCE_WINDOW 
    weight_tracking = []
    
    transaction_costs_list = []
    cumulative_transaction_costs = 0.0
    
    # MOO tracking
    pending_execution = None

    print("\n" + "="*80)
    print("GOD MODE STRATEGY SIMULATION (PERFECT FORESIGHT)")
    print("="*80)
    print(f"  Rebalance Window: {REBALANCE_WINDOW} days")
    print(f"  Top N Winners: {TOP_N_PORTFOLIO}")
    print(f"  Allocation: 100% to Top N (Equal Weight)")
    print("="*80 + "\n")

    # Start loop
    for i in range(1, len(daily_returns)):
        current_date = price_df.index[i]
        
        # --- 1. Handle MOO Execution (if scheduled) ---
        executing_today = False
        scheduled_weights_for_return = None
        
        if pending_execution and pending_execution.get('date') == current_date:
            executing_today = True
            
            # Apply transaction costs from the trade scheduled yesterday
            pending_cost = pending_execution.get('transaction_cost', 0.0)
            if pending_cost > 0:
                equity[-1] -= pending_cost
                cumulative_transaction_costs += pending_cost
            
            # Update weights to the new target
            scheduled_weights = pending_execution.get('weights')
            if scheduled_weights is not None:
                weights = scheduled_weights.copy()
                scheduled_weights_for_return = weights.copy()
            
            # Recalculate today's return using Open-to-Close if MOO
            if use_moo and i < len(execution_prices):
                # Return from Open (Execution) to Close
                moo_rets = (price_df.iloc[i] / execution_prices.iloc[i]) - 1.0
                daily_returns.iloc[i] = moo_rets.fillna(0.0)
            
            pending_execution = None

        # --- 2. Check Rebalance ---
        # God Mode rebalances based on FUTURE returns from 'i' to 'i + REBALANCE_WINDOW'
        should_rebalance = False
        
        # We rebalance if enough time has passed since last rebalance
        # Note: We can rebalance at step 0 (effectively day 1) if we want to start immediately
        # But here we are iterating from 1.
        if i - last_rebalance_day >= REBALANCE_WINDOW:
            should_rebalance = True
            
        old_weights = weights.copy()
        
        if should_rebalance:
            last_rebalance_day = i
            
            # LOOK AHEAD: Define the future window
            start_idx = i
            end_idx = min(i + REBALANCE_WINDOW, len(price_df))
            
            # If no future left (end of data), we can't optimize, just hold or cash?
            # We'll just hold previous or do nothing.
            if start_idx < end_idx:
                future_prices = price_df.iloc[start_idx:end_idx]
                
                # Calculate future return for each ETF
                # Return = (Price at end of window / Price now) - 1
                future_returns = {}
                
                for etf in price_df.columns:
                    # Get price series for this ETF in the future window
                    series = future_prices[etf].dropna()
                    
                    if not series.empty:
                        # We need at least start and end prices
                        # ideally the series covers the whole window, or at least has 2 points
                        if len(series) >= 2:
                            start_p = series.iloc[0] # Price 'now' (or closest valid)
                            end_p = series.iloc[-1]  # Price at end of window
                            
                            if start_p > 0:
                                ret = (end_p / start_p) - 1.0
                                future_returns[etf] = ret
                
                # GOD DECISION: Pick the ones with highest future return
                sorted_returns = sorted(future_returns.items(), key=lambda x: x[1], reverse=True)
                
                # Pick Top N positive returns? Or just Top N regardless?
                # Usually Top N. God knows to short losers, but we are Long Only here.
                # So we pick Top N. If all are negative, we pick the "least negative" (best capital preservation)
                # or we could go to Cash. Let's stick to Long Only Top N for now.
                
                top_picks = [x[0] for x in sorted_returns[:TOP_N_PORTFOLIO]]
                
                # Assign Weights
                new_weights = pd.Series(0.0, index=price_df.columns)
                if top_picks:
                    weight_val = 1.0 / len(top_picks)
                    # Cap at MAX_WEIGHT if desired, though for God Mode usually 100% allocation is fine
                    weight_val = min(weight_val, MAX_WEIGHT)
                    
                    for etf in top_picks:
                        new_weights[etf] = weight_val
                
                # --- Transaction Costs Calculation ---
                if ENABLE_TRANSACTION_COSTS:
                    turnover = sum(abs(new_weights.get(etf, 0.0) - weights.get(etf, 0.0)) for etf in price_df.columns)
                    transaction_cost = TRANSACTION_COST_RATE * turnover * equity[-1]
                    
                    if use_moo:
                        # Schedule for tomorrow (or next available)
                        if i + 1 < len(price_df):
                            pending_execution = {
                                'date': price_df.index[i+1],
                                'weights': new_weights.copy(),
                                'transaction_cost': transaction_cost
                            }
                            transaction_costs_list.append({
                                'date': price_df.index[i],
                                'cost': transaction_cost,
                                'top_picks': top_picks
                            })
                    else:
                        # Execute immediately
                        equity[-1] -= transaction_cost
                        cumulative_transaction_costs += transaction_cost
                        weights = new_weights.copy()
                        
                        transaction_costs_list.append({
                            'date': price_df.index[i],
                            'cost': transaction_cost,
                            'top_picks': top_picks
                        })
                else:
                    # No costs
                    if use_moo:
                        if i + 1 < len(price_df):
                            pending_execution = {
                                'date': price_df.index[i+1],
                                'weights': new_weights.copy(),
                                'transaction_cost': 0.0
                            }
                    else:
                        weights = new_weights.copy()

                # Debug
                if DEBUG_MODE and len(transaction_costs_list) > 0 and transaction_costs_list[-1]['date'] == price_df.index[i]:
                    print(f"Date: {current_date.strftime('%Y-%m-%d')} | Rebalancing for next {REBALANCE_WINDOW} days")
                    print(f"  Top Picks (Future Winners): {top_picks}")
                    print(f"  Future Returns: {[f'{future_returns[x]:.2%}' for x in top_picks]}")

        # --- 3. Calculate Daily Return ---
        
        # Determine effective weights held today
        if use_moo:
            if executing_today and scheduled_weights_for_return is not None:
                # We switched into new weights at Open
                daily_weights = scheduled_weights_for_return
            elif should_rebalance:
                # We decided to rebalance, but execution is tomorrow. Hold old weights today.
                daily_weights = old_weights
            else:
                daily_weights = weights
        else:
            daily_weights = weights
            
        # Portfolio Return
        valid_etfs = [e for e in price_df.columns if daily_weights[e] > 0 and not pd.isna(daily_returns.iloc[i][e])]
        if valid_etfs:
            total_w = sum(daily_weights[e] for e in valid_etfs)
            if total_w > 0:
                # Weighted return of held assets
                port_ret = sum(daily_returns.iloc[i][e] * (daily_weights[e]/total_w) for e in valid_etfs)
            else:
                port_ret = 0.0
        else:
            port_ret = 0.0
            
        # Update Equity
        if USE_COMPOUND_MODE:
            dollar_ret = equity[-1] * port_ret
        else:
            dollar_ret = min(equity[-1], INITIAL_CAPITAL) * port_ret
            
        equity.append(equity[-1] + dollar_ret)
        
        # Tracking
        track = {'date': current_date, 'cumulative_transaction_costs': cumulative_transaction_costs}
        for e in price_df.columns:
            track[f'{e}_weight'] = daily_weights.get(e, 0.0)
        weight_tracking.append(track)
        
        if excel_reporter:
            excel_reporter.log_daily_position(current_date, daily_weights, price_df.iloc[i], equity[-1], daily_returns.iloc[i])

    # Compile results
    results_df = pd.DataFrame({'God Mode': equity}, index=price_df.index)
    track_df = pd.DataFrame(weight_tracking)
    
    print(f"\nFinal Equity: ${equity[-1]:,.2f}")
    print(f"Total Costs: ${cumulative_transaction_costs:,.2f}")
    
    return results_df, track_df


def main():
    etf_data = load_all_etf_data(ETF_DATA_FOLDER, ETF_UNIVERSE)
    if not etf_data:
        print("No ETF data loaded.")
        return

    price_df = create_price_dataframe(etf_data, START_DATE, END_DATE)
    open_df = create_open_dataframe(etf_data, START_DATE, END_DATE)

    # Align indices
    common_dates = price_df.index.intersection(open_df.index)
    price_df = price_df.loc[common_dates]
    open_df = open_df.loc[common_dates]

    print(f"Aligned {len(price_df)} trading days.\n")

    excel_reporter = ProfessionalExcelReporter(INITIAL_CAPITAL, REPORTS_DIR)

    # Run God Mode Strategy
    god_equity, portfolio_tracking_df = run_god_mode_strategy(
        price_df, INITIAL_CAPITAL, excel_reporter, open_df=open_df
    )

    # Benchmark: Equal Weight of Quality ETFs
    # (Just a simple reference to see how much better God is)
    print("\nCalculating Benchmark (Buy & Hold SPY)...")
    if 'SPY' in price_df.columns:
        spy_prices = price_df['SPY']
        spy_ret = (spy_prices / spy_prices.iloc[0])
        spy_equity = spy_ret * INITIAL_CAPITAL
        benchmark_df = pd.DataFrame({'SPY (Buy & Hold)': spy_equity}, index=price_df.index)
    else:
        benchmark_df = pd.DataFrame({'Benchmark': [INITIAL_CAPITAL]*len(price_df)}, index=price_df.index)

    # Plotting
    print("\nGenerating Plot...")
    os.makedirs(REPORTS_DIR, exist_ok=True)
    
    plt.style.use('seaborn-v0_8-darkgrid')
    plt.figure(figsize=(16, 9))
    
    plt.plot(god_equity.index, god_equity['God Mode'], color='gold', linewidth=2.5, label='GOD MODE (Perfect Foresight)')
    plt.plot(benchmark_df.index, benchmark_df.iloc[:, 0], color='gray', linestyle='--', linewidth=1.5, label='SPY (Benchmark)')
    
    plt.yscale('log')
    plt.title('GOD MODE vs Benchmark (The Ceiling of Possibility)', fontsize=16, fontweight='bold')
    plt.ylabel('Portfolio Value ($)', fontsize=14)
    plt.xlabel('Date', fontsize=14)
    
    formatter = mticker.FormatStrFormatter('$%.0f')
    plt.gca().yaxis.set_major_formatter(formatter)
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.legend(fontsize=12, loc='upper left')
    
    output_path = os.path.join(REPORTS_DIR, OUTPUT_FILENAME)
    plt.savefig(output_path, dpi=300)
    print(f"[OK] Plot saved to: {output_path}")

    # Generate Weights Viz
    if not portfolio_tracking_df.empty:
        try:
            print("\nGenerating Weight Visualizations...")
            viz = ETFWeightVisualizer(portfolio_tracking_df, REPORTS_DIR)
            viz.create_all_plots()
        except Exception as e:
            print(f"Weight Viz Error: {e}")

    # QuantStats Report
    try:
        print("\nGenerating QuantStats Report...")
        qs.extend_pandas()
        god_returns = god_equity['God Mode'].pct_change().dropna()
        bench_returns = benchmark_df.iloc[:, 0].pct_change().dropna()
        
        qs_output = os.path.join(REPORTS_DIR, "god_mode_report.html")
        qs.reports.html(god_returns, benchmark=bench_returns, output=qs_output, title="God Mode ETF Strategy")
        print(f"[OK] QS Report saved: {qs_output}")
    except Exception as e:
        print(f"QS Error: {e}")

    # Excel Report
    try:
        excel_path = excel_reporter.generate_comprehensive_report(
            god_equity.iloc[:, 0],
            benchmark_df.iloc[:, 0],
            ETF_UNIVERSE
        )
        print(f"[OK] Excel Report saved: {excel_path}")
    except Exception as e:
        print(f"Excel Error: {e}")

if __name__ == "__main__":
    main()

