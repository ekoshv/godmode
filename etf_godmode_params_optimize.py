"""
Bare-Metal ETF "GOD MODE" Parameter Optimization using Optuna
Based on the robust infrastructure of etf_bm_params_optimize_17.py

Target Script: run_godmode_etf_v01.py
Strategy: Perfect Foresight (Future Peeking)

Optimized Parameters:
- REBALANCE_WINDOW: How far into the future to look (and how often to trade).
- TOP_N_PORTFOLIO: Number of assets to hold.

Goal: Find the optimal look-ahead window and concentration to maximize returns
while accounting for transaction costs.
"""

import os
import pandas as pd
import numpy as np
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
import json
from datetime import datetime
from multiprocessing import Pool
import warnings
import quantstats as qs

warnings.filterwarnings('ignore')

# --- Configuration ---
START_DATE = "2016-01-01"
END_DATE = "2025-09-26"
ETF_DATA_FOLDER = "data_etf"
ETF_UNIVERSE = [
    'SPY', 'QQQ', 'IVV', 'VOO', 'VTI', 'IWM', 'VT', 'VXUS',
    'BND', 'TLT', 'GLD', 'SLV', 'IBIT', 'SCHD', 'SGOV'
]
QUALITY_ETFS = ['SPY', 'QQQ', 'IVV', 'VOO', 'VTI', 'SCHD', 'GLD'] # For Benchmark

# Fixed Strategy Config
INITIAL_CAPITAL = 100000.0
USE_COMPOUND_MODE = True # Generally God Mode wants to compound
ENABLE_TRANSACTION_COSTS = True
TRANSACTION_COST_RATE = 0.001  # 0.1%

# Optimization Config
N_PROCESSES = 24  # Adjust based on your CPU cores
N_TRIALS_PER_PROCESS = 100
N_STARTUP_TRIALS = 30
JOURNAL_STORAGE_PATH = "reports/god_mode_benchmark/optuna_journal_god.log"
RESULTS_DIR = "reports/god_mode_benchmark/optuna_results"

# Objective Weights (Composite Score)
# For God Mode, we prioritize absolute Return heavily, but Sharpe matters too.
SHARPE_WEIGHT = 0.20
SORTINO_WEIGHT = 0.20
RETURN_WEIGHT = 0.60
DRAWDOWN_WEIGHT = 0.00

# --- Helper Functions ---

def custom_sharpe(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
    """Calculate annualized Sharpe ratio."""
    if returns.empty or returns.std() == 0:
        return 0.0
    excess_returns = returns - risk_free_rate / 252
    return np.sqrt(252) * excess_returns.mean() / excess_returns.std()

def custom_sortino(returns: pd.Series, required_return: float = 0.0) -> float:
    """Calculate annualized Sortino ratio."""
    if returns.empty:
        return 0.0
    mean_return = returns.mean() * 252
    downside_returns = returns[returns < 0]
    if downside_returns.empty:
        return 100.0
    downside_std = downside_returns.std() * np.sqrt(252)
    if downside_std == 0:
        return 100.0
    sortino = (mean_return - required_return) / downside_std
    return np.clip(sortino, -100.0, 100.0)

def custom_max_drawdown(prices: pd.Series) -> float:
    """Calculate maximum drawdown."""
    if prices.empty:
        return 0.0
    cumulative_max = prices.cummax()
    drawdown = (prices - cumulative_max) / cumulative_max
    return drawdown.min()

# --- Data Loading ---

# Global cache
GLOBAL_ETF_DATA = None
GLOBAL_CLOSE_DF = None
GLOBAL_OPEN_DF = None

def load_all_etf_data(data_folder: str, etf_universe: list) -> dict:
    etf_data = {}
    for etf_name in etf_universe:
        csv_path = os.path.join(data_folder, f"{etf_name}.csv")
        if not os.path.exists(csv_path):
            continue
        try:
            df = pd.read_csv(csv_path)
            df.columns = df.columns.str.strip()
            date_col = next((col for col in ['Date', 'date', 'timestamp'] if col in df.columns), None)
            if not date_col: continue
            
            df['date'] = pd.to_datetime(df[date_col], errors='coerce', utc=True).dt.tz_localize(None)
            df.dropna(subset=['date'], inplace=True)

            close_col = next((col for col in ['Close', 'close'] if col in df.columns), None)
            if not close_col: continue
            
            df['close'] = df[close_col]
            
            open_col = next((col for col in ['Open', 'open'] if col in df.columns), None)
            df['open'] = df[open_col] if open_col else df['close']
            
            df = df.sort_values('date').drop_duplicates(subset=['date'], keep='last')
            etf_data[etf_name] = df[['date', 'close', 'open']].set_index('date')
        except Exception:
            continue
    return etf_data

def create_price_dataframe(etf_data: dict, start_date: str, end_date: str) -> pd.DataFrame:
    all_close = {name: df['close'] for name, df in etf_data.items()}
    close_df = pd.DataFrame(all_close).sort_index()
    close_df = close_df.loc[start_date:end_date].dropna(how='all')
    return close_df

def create_open_dataframe(etf_data: dict, start_date: str, end_date: str) -> pd.DataFrame:
    all_open = {name: (df['open'] if 'open' in df.columns else df['close']) for name, df in etf_data.items()}
    open_df = pd.DataFrame(all_open).sort_index()
    open_df = open_df.loc[start_date:end_date].dropna(how='all')
    return open_df

def load_global_data():
    global GLOBAL_ETF_DATA, GLOBAL_CLOSE_DF, GLOBAL_OPEN_DF
    if GLOBAL_ETF_DATA is None:
        print("Loading ETF data (one-time)...")
        GLOBAL_ETF_DATA = load_all_etf_data(ETF_DATA_FOLDER, ETF_UNIVERSE)
        GLOBAL_CLOSE_DF = create_price_dataframe(GLOBAL_ETF_DATA, START_DATE, END_DATE)
        GLOBAL_OPEN_DF = create_open_dataframe(GLOBAL_ETF_DATA, START_DATE, END_DATE)
        
        # Sync indices
        common = GLOBAL_CLOSE_DF.index.intersection(GLOBAL_OPEN_DF.index)
        GLOBAL_CLOSE_DF = GLOBAL_CLOSE_DF.loc[common]
        GLOBAL_OPEN_DF = GLOBAL_OPEN_DF.loc[common]

# --- Backtest Engine (God Mode) ---

def run_god_mode_backtest(params: dict) -> dict:
    """
    Run God Mode strategy with given parameters.
    """
    # Extract params
    rebalance_window = params['rebalance_window']
    top_n = params['top_n_portfolio']
    max_weight = 1.0 # God Mode usually doesn't limit weight if top_n is small, but we distribute equally
    
    price_df = GLOBAL_CLOSE_DF
    open_df = GLOBAL_OPEN_DF
    
    # MOO Execution logic
    use_moo = True
    execution_prices = open_df if use_moo else price_df
    
    daily_returns = price_df.pct_change()
    equity = [INITIAL_CAPITAL]
    weights = pd.Series(0.0, index=price_df.columns)
    
    last_rebalance_day = -rebalance_window
    pending_execution = None
    
    optimizer_returns_list = []
    
    # We start iterating from 1
    for i in range(1, len(daily_returns)):
        current_date = price_df.index[i]
        
        # --- 1. Execution ---
        executing_today = False
        pending_cost = 0.0
        scheduled_weights_for_return = None
        
        if pending_execution and pending_execution.get('date') == current_date:
            executing_today = True
            pending_cost = pending_execution.get('transaction_cost', 0.0)
            if pending_cost > 0:
                equity[-1] -= pending_cost
            
            weights = pending_execution.get('weights').copy()
            scheduled_weights_for_return = weights.copy()
            
            if use_moo and i < len(execution_prices):
                moo_rets = (price_df.iloc[i] / execution_prices.iloc[i]) - 1.0
                daily_returns.iloc[i] = moo_rets.fillna(0.0)
            
            pending_execution = None

        # --- 2. Rebalance Check ---
        # God Mode rebalances based on FUTURE returns from 'i' to 'i + REBALANCE_WINDOW'
        should_rebalance = False
        if i - last_rebalance_day >= rebalance_window:
            should_rebalance = True
        
        old_weights = weights.copy()
        
        if should_rebalance:
            last_rebalance_day = i
            
            # LOOK AHEAD
            start_idx = i
            end_idx = min(i + rebalance_window, len(price_df))
            
            new_weights = pd.Series(0.0, index=price_df.columns)
            
            if start_idx < end_idx:
                future_prices = price_df.iloc[start_idx:end_idx]
                future_returns = {}
                
                for etf in price_df.columns:
                    series = future_prices[etf].dropna()
                    if len(series) >= 2:
                        ret = (series.iloc[-1] / series.iloc[0]) - 1.0
                        future_returns[etf] = ret
                
                # Pick Top N Winners
                sorted_returns = sorted(future_returns.items(), key=lambda x: x[1], reverse=True)
                top_picks = [x[0] for x in sorted_returns[:top_n]]
                
                if top_picks:
                    weight_val = 1.0 / len(top_picks)
                    for etf in top_picks:
                        new_weights[etf] = weight_val
            
            # Transaction Costs & Scheduling
            turnover = sum(abs(new_weights.get(e,0) - weights.get(e,0)) for e in price_df.columns)
            cost = 0.0
            if ENABLE_TRANSACTION_COSTS:
                cost = TRANSACTION_COST_RATE * turnover * equity[-1]
            
            if use_moo:
                if i + 1 < len(price_df):
                    pending_execution = {
                        'date': price_df.index[i+1],
                        'weights': new_weights,
                        'transaction_cost': cost
                    }
                else:
                    equity[-1] -= cost
                    weights = new_weights
            else:
                equity[-1] -= cost
                weights = new_weights
        
        # --- 3. Returns ---
        if use_moo:
            if executing_today and scheduled_weights_for_return is not None:
                daily_weights = scheduled_weights_for_return
            elif should_rebalance:
                daily_weights = old_weights
            else:
                daily_weights = weights
        else:
            daily_weights = weights
            
        # Calculate Port Return
        valid_etfs = [e for e in price_df.columns if daily_weights[e] > 0 and not pd.isna(daily_returns.iloc[i][e])]
        if valid_etfs:
            total_w = sum(daily_weights[e] for e in valid_etfs)
            if total_w > 0:
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
        optimizer_returns_list.append(port_ret)

    # Metrics
    returns_series = pd.Series(optimizer_returns_list)
    
    sharpe = custom_sharpe(returns_series)
    sortino = custom_sortino(returns_series)
    
    # Calculate Drawdown
    equity_s = pd.Series(equity)
    roll_max = equity_s.cummax()
    dd = (equity_s - roll_max) / roll_max
    max_dd = dd.min()
    
    final_value = equity[-1]
    total_return = (final_value / INITIAL_CAPITAL) - 1.0

    return {
        'final_value': final_value,
        'total_return': total_return,
        'sharpe_ratio': sharpe,
        'sortino_ratio': sortino,
        'max_drawdown': max_dd,
        'returns_list': optimizer_returns_list
    }

# --- Optuna Objective ---

def objective(trial: optuna.Trial) -> float:
    params = {
        'rebalance_window': trial.suggest_int('rebalance_window', 1, 60), # 1 day to ~2 months
        'top_n_portfolio': trial.suggest_int('top_n_portfolio', 1, 5),    # 1 to 5 assets
    }
    
    try:
        results = run_god_mode_backtest(params)
        
        # Composite Score (Heavy weighting on absolute return)
        score = (
            (results['sharpe_ratio'] * SHARPE_WEIGHT) +
            (results['sortino_ratio'] * SORTINO_WEIGHT) +
            (results['total_return'] * RETURN_WEIGHT)
        )
        
        # Store attributes
        trial.set_user_attr('final_value', results['final_value'])
        trial.set_user_attr('sharpe', results['sharpe_ratio'])
        trial.set_user_attr('total_return', results['total_return'])
        trial.set_user_attr('max_drawdown', results['max_drawdown'])
        
        return score
    except Exception as e:
        print(f"Trial failed: {e}")
        return float('-inf')

# --- Main/Support Functions ---

def save_results(study: optuna.Study):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    best_params = study.best_params
    
    load_global_data()
    print("\nRunning Final QuantStats Report for Best Params...")
    final_res = run_god_mode_backtest(best_params)
    
    # Save JSON
    summary = {
        'best_params': best_params,
        'best_value': study.best_value,
        'metrics': {k:v for k,v in final_res.items() if not isinstance(v, list)}
    }
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, f'god_best_params_{timestamp}.json'), 'w') as f:
        json.dump(summary, f, indent=2)
        
    print("\n" + "="*90)
    print("GOD MODE OPTIMIZATION RESULTS")
    print("="*90)
    print(f"Best Objective Value: {study.best_value:.4f}")
    print(f"Total Trials: {len(study.trials)}")
    print()
    print("Best Parameters:")
    for key, value in best_params.items():
        print(f"  {key}: {value}")
    print()
    print("Performance Metrics:")
    print(f"  Final Value:      ${final_res['final_value']:,.2f}")
    print(f"  Total Return:     {final_res['total_return']:>7.2%}")
    print(f"  Sharpe Ratio:     {final_res['sharpe_ratio']:>7.3f}")
    print(f"  Sortino Ratio:    {final_res['sortino_ratio']:>7.3f}")
    print(f"  Max Drawdown:     {final_res['max_drawdown']:>7.2%}")
    print()
    print("="*90)
    
    # --- Copy-Paste Section ---
    print("\n" + "="*90)
    print("COPY-PASTE CONFIGURATION FOR run_godmode_etf_v01.py")
    print("="*90)
    print("# Copy the following optimized parameters to run_godmode_etf_v01.py:\n")
    print("# --- Strategy Parameters ---")
    print(f"REBALANCE_WINDOW = {best_params['rebalance_window']}")
    print(f"TOP_N_PORTFOLIO = {best_params['top_n_portfolio']}")
    print("MAX_WEIGHT = 1.0  # God doesn't limit diversification")
    print()
    print("# --- Fixed Configuration ---")
    print(f"USE_COMPOUND_MODE = {USE_COMPOUND_MODE}")
    print(f"ENABLE_TRANSACTION_COSTS = {ENABLE_TRANSACTION_COSTS}")
    if ENABLE_TRANSACTION_COSTS:
        print(f"TRANSACTION_COST_RATE = {TRANSACTION_COST_RATE}")
    print("\n" + "="*90)

def run_process(pid):
    load_global_data()
    
    study = optuna.create_study(
        study_name="etf_god_optimization",
        storage=JournalStorage(JournalFileBackend(file_path=JOURNAL_STORAGE_PATH)),
        direction='maximize',
        sampler=TPESampler(n_startup_trials=N_STARTUP_TRIALS),
        load_if_exists=True
    )
    
    study.optimize(objective, n_trials=N_TRIALS_PER_PROCESS, show_progress_bar=False)

if __name__ == "__main__":
    if os.path.exists(JOURNAL_STORAGE_PATH):
        try:
            os.remove(JOURNAL_STORAGE_PATH)
        except:
            pass
        
    print(f"Starting God Mode Optimization with {N_PROCESSES} processes...")
    with Pool(processes=N_PROCESSES) as pool:
        pool.map(run_process, range(N_PROCESSES))
        
    # Final Reporting
    study = optuna.load_study(
        study_name="etf_god_optimization",
        storage=JournalStorage(JournalFileBackend(file_path=JOURNAL_STORAGE_PATH))
    )
    save_results(study)

