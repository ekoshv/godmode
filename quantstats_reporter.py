"""
QuantStats Integration Module
Provides professional-grade performance analytics and reporting using QuantStats library.
Handles proper daily returns calculation for our 5-day rebalancing system.
"""
import pandas as pd
import quantstats as qs
from typing import Dict, Optional, Tuple
import os

class QuantStatsReporter:
    """
    Professional performance reporting using QuantStats library.
    Handles proper daily returns calculation for rebalancing-based backtesting.
    """
    
    def __init__(self, rebalance_window: int = 5):
        """
        Initialize QuantStats reporter.
        
        Args:
            rebalance_window: Days between rebalancing (for proper interpolation)
        """
        self.rebalance_window = rebalance_window
        
        # Extend pandas with QuantStats functionality
        qs.extend_pandas()
        
        print(f"QuantStats Reporter initialized with {rebalance_window}-day rebalancing")
    
    def prepare_daily_returns(self, results_df: pd.DataFrame, 
                            benchmark_df: Optional[pd.DataFrame] = None) -> Tuple[pd.Series, Optional[pd.Series]]:
        """
        Prepare daily returns for QuantStats analysis.
        
        Args:
            results_df: Backtest results with portfolio values
            benchmark_df: Optional benchmark data
            
        Returns:
            Tuple of (portfolio_daily_returns, benchmark_daily_returns)
        """
        # Ensure date column is datetime and sorted
        results_df = results_df.copy()
        results_df['date'] = pd.to_datetime(results_df['date'])
        results_df = results_df.sort_values('date').reset_index(drop=True)
        
        # Check if we have true daily data or rebalancing data
        date_diff = results_df['date'].diff().dt.days
        avg_date_diff = date_diff.mean()
        
        if avg_date_diff <= 1.5:  # True daily data (allowing for weekends)
            # True daily data - use directly
            results_df = results_df.set_index('date')
            portfolio_returns = results_df['portfolio_value'].pct_change().dropna()
            
            # Handle benchmark if provided
            benchmark_returns = None
            if benchmark_df is not None and 'benchmark_value' in benchmark_df.columns:
                benchmark_df = benchmark_df.copy()
                benchmark_df['date'] = pd.to_datetime(benchmark_df['date'])
                benchmark_df = benchmark_df.sort_values('date').reset_index(drop=True)
                benchmark_df = benchmark_df.set_index('date')
                benchmark_returns = benchmark_df['benchmark_value'].pct_change().dropna()
        else:
            # Rebalancing data - interpolate to daily
            start_date = results_df['date'].min()
            end_date = results_df['date'].max()
            
            # Generate all trading days (weekdays only)
            all_dates = pd.date_range(start=start_date, end=end_date, freq='B')
            
            # Create DataFrame with all trading days
            daily_df = pd.DataFrame({'date': all_dates})
            daily_df = daily_df.merge(results_df[['date', 'portfolio_value']], on='date', how='left')
            
            # Forward fill missing values (interpolate between rebalancing days)
            daily_df['portfolio_value'] = daily_df['portfolio_value'].fillna(method='ffill')
            
            # Calculate daily returns
            daily_df['daily_return'] = daily_df['portfolio_value'].pct_change()
            portfolio_returns = daily_df.set_index('date')['daily_return'].dropna()
            
            # Handle benchmark if provided
            benchmark_returns = None
            if benchmark_df is not None and 'benchmark_value' in benchmark_df.columns:
                benchmark_df = benchmark_df.copy()
                benchmark_df['date'] = pd.to_datetime(benchmark_df['date'])
                benchmark_df = benchmark_df.sort_values('date').reset_index(drop=True)
                
                # Merge benchmark data
                daily_df = daily_df.merge(benchmark_df[['date', 'benchmark_value']], on='date', how='left')
                daily_df['benchmark_value'] = daily_df['benchmark_value'].fillna(method='ffill')
                daily_df['benchmark_daily_return'] = daily_df['benchmark_value'].pct_change()
                benchmark_returns = daily_df.set_index('date')['benchmark_daily_return'].dropna()
        
        # Quiet summary only if needed elsewhere
        
        return portfolio_returns, benchmark_returns
    
    def generate_html_report(self, portfolio_returns: pd.Series, 
                           benchmark_returns: Optional[pd.Series] = None,
                           output_path: str = "reports/quantstats_report.html",
                           title: str = "Portfolio Performance Analysis") -> str:
        """
        Generate comprehensive HTML performance report.
        
        Args:
            portfolio_returns: Daily portfolio returns
            benchmark_returns: Optional benchmark returns
            output_path: Path to save HTML report
            title: Report title
            
        Returns:
            Path to generated report
        """
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Generate HTML report
        if benchmark_returns is not None:
            qs.reports.html(
                portfolio_returns, 
                benchmark=benchmark_returns,
                output=output_path,
                title=title
            )
            print(f"Generated comparative HTML report: {output_path}")
        else:
            qs.reports.html(
                portfolio_returns,
                output=output_path,
                title=title
            )
            print(f"Generated standalone HTML report: {output_path}")
        
        return output_path
    
    def generate_tearsheet(self, portfolio_returns: pd.Series,
                          benchmark_returns: Optional[pd.Series] = None,
                          output_path: str = "reports/quantstats_tearsheet.html") -> str:
        """
        Generate tearsheet (summary) report using QuantStats basic report.
        
        Args:
            portfolio_returns: Daily portfolio returns
            benchmark_returns: Optional benchmark returns
            output_path: Path to save tearsheet
            
        Returns:
            Path to generated tearsheet
        """
        import sys
        import io
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save original stdout and set UTF-8 encoding to handle Unicode characters
        original_stdout = sys.stdout
        utf8_stdout = None
        try:
            utf8_stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
            sys.stdout = utf8_stdout
        except:
            pass  # If stdout redirection fails, continue anyway
        
        try:
            # Try using basic report (tearsheet equivalent)
            if benchmark_returns is not None:
                qs.reports.basic(
                    portfolio_returns,
                    benchmark=benchmark_returns,
                    output=output_path
                )
            else:
                qs.reports.basic(
                    portfolio_returns,
                    output=output_path
                )
        except AttributeError:
            # Fallback to HTML report if basic doesn't exist
            # Keep quiet; will generate HTML instead
            if benchmark_returns is not None:
                qs.reports.html(
                    portfolio_returns,
                    benchmark=benchmark_returns,
                    output=output_path
                )
            else:
                qs.reports.html(
                    portfolio_returns,
                    output=output_path
                )
        finally:
            # Restore original stdout
            sys.stdout = original_stdout
            # Close the UTF-8 wrapper if it was created
            if utf8_stdout is not None and utf8_stdout != original_stdout:
                try:
                    utf8_stdout.detach()  # Detach the buffer before closing to avoid closing sys.stdout.buffer
                except:
                    pass
        
        print(f"Generated tearsheet: {output_path}")
        return output_path
    
    def get_performance_metrics(self, portfolio_returns: pd.Series,
                               benchmark_returns: Optional[pd.Series] = None) -> Dict:
        """
        Get comprehensive performance metrics using QuantStats with error handling.
        
        Args:
            portfolio_returns: Daily portfolio returns
            benchmark_returns: Optional benchmark returns
            
        Returns:
            Dictionary with performance metrics
        """
        metrics = {}
        
        # Define metrics to calculate with their QuantStats functions
        basic_metrics = [
            ('total_return', 'comp'),
            ('annualized_return', 'cagr'),
            ('volatility', 'volatility'),
            ('sharpe_ratio', 'sharpe'),
            ('sortino_ratio', 'sortino'),
            ('calmar_ratio', 'calmar'),
            ('max_drawdown', 'max_drawdown'),
            # Some stats modules may not support these in this environment; omit noisy ones
            ('skewness', 'skew'),
            ('kurtosis', 'kurtosis'),
            ('tail_ratio', 'tail_ratio')
        ]
        
        # Calculate basic metrics with error handling
        for metric_name, metric_func_name in basic_metrics:
            func = getattr(qs.stats, metric_func_name, None)
            if callable(func):
                try:
                    metrics[metric_name] = func(portfolio_returns)
                except (AttributeError, TypeError, ValueError):
                    metrics[metric_name] = None
            else:
                metrics[metric_name] = None
        
        # Benchmark comparison if available (limited to outperformance only)
        if benchmark_returns is not None:
            try:
                benchmark_cagr = qs.stats.cagr(benchmark_returns)
                metrics['outperformance'] = metrics.get('annualized_return') - benchmark_cagr if metrics.get('annualized_return') is not None else None
            except (AttributeError, TypeError, ValueError):
                metrics['outperformance'] = None
        
        return metrics
    
    def create_performance_plots(self, portfolio_returns: pd.Series,
                                benchmark_returns: Optional[pd.Series] = None,
                                output_dir: str = "reports/quantstats_plots") -> Dict[str, str]:
        """
        Create individual performance plots with error handling.
        
        Args:
            portfolio_returns: Daily portfolio returns
            benchmark_returns: Optional benchmark returns
            output_dir: Directory to save plots
            
        Returns:
            Dictionary with plot file paths
        """
        os.makedirs(output_dir, exist_ok=True)
        
        plot_files = {}
        
        # Define plots to try creating
        plots_to_create = [
            ('cumulative_returns', 'returns', 'Cumulative returns plot'),
            ('rolling_sharpe', 'rolling_sharpe', 'Rolling Sharpe ratio'),
            ('drawdown', 'drawdown', 'Drawdown plot'),
            ('monthly_returns', 'monthly_returns', 'Monthly returns heatmap'),
            ('rolling_volatility', 'rolling_volatility', 'Rolling volatility')
        ]
        
        for plot_name, plot_method, description in plots_to_create:
            try:
                plot_path = os.path.join(output_dir, f'{plot_name}.png')
                
                if plot_method == 'returns':
                    qs.plots.returns(portfolio_returns, benchmark=benchmark_returns, savefig=plot_path)
                elif plot_method == 'rolling_sharpe':
                    qs.plots.rolling_sharpe(portfolio_returns, benchmark=benchmark_returns, savefig=plot_path)
                elif plot_method == 'drawdown':
                    qs.plots.drawdown(portfolio_returns, savefig=plot_path)
                elif plot_method == 'monthly_returns':
                    qs.plots.monthly_returns(portfolio_returns, savefig=plot_path)
                elif plot_method == 'rolling_volatility':
                    qs.plots.rolling_volatility(portfolio_returns, benchmark=benchmark_returns, savefig=plot_path)
                
                plot_files[plot_name] = plot_path
                print(f"  - Created {description}")
                
            except AttributeError:
                print("  - Skipped plot: method not available")
            except (TypeError, ValueError, RuntimeError):
                print("  - Failed to create plot")
        
        print(f"Created {len(plot_files)} QuantStats plots in {output_dir}")
        
        return plot_files
    
    def print_metrics_summary(self, metrics: Dict):
        """
        Print a formatted summary of performance metrics with None handling.
        
        Args:
            metrics: Dictionary of performance metrics
        """
        def safe_format(value, format_str=".2%", default="N/A"):
            """Safely format a value, handling None values."""
            if value is None:
                return default
            try:
                if format_str.endswith('%'):
                    return f"{value:.2%}"
                elif format_str.endswith('f'):
                    return f"{value:.2f}"
                elif format_str.endswith('.0f'):
                    return f"{value:.0f}"
                else:
                    return str(value)
            except (TypeError, ValueError):
                return default
        
        print("\nQUANTSTATS SUMMARY (key metrics)")
        
        # Returns
        print(f"Total Return:           {safe_format(metrics.get('total_return'))}")
        print(f"Annualized Return:      {safe_format(metrics.get('annualized_return'))}")
        print(f"Volatility:            {safe_format(metrics.get('volatility'))}")
        
        # Risk-adjusted returns
        print("\nRisk-Adjusted Returns:")
        print(f"Sharpe Ratio:          {safe_format(metrics.get('sharpe_ratio'), '.2f')}")
        print(f"Sortino Ratio:         {safe_format(metrics.get('sortino_ratio'), '.2f')}")
        print(f"Calmar Ratio:          {safe_format(metrics.get('calmar_ratio'), '.2f')}")
        
        # Risk metrics
        print("\nRisk Metrics:")
        print(f"Max Drawdown:          {safe_format(metrics.get('max_drawdown'))}")
        # Omit noisy/unsupported metrics in console
        
        # Distribution metrics
        print("\nDistribution:")
        print(f"Skewness:              {safe_format(metrics.get('skewness'), '.2f')}")
        print(f"Kurtosis:              {safe_format(metrics.get('kurtosis'), '.2f')}")
        print(f"Tail Ratio:            {safe_format(metrics.get('tail_ratio'), '.2f')}")
        
        # Benchmark comparison
        if 'alpha' in metrics:
            print("\nBenchmark Comparison:")
            # Print only information ratio and outperformance if available
            if metrics.get('information_ratio') is not None:
                print(f"Information Ratio:     {safe_format(metrics.get('information_ratio'), '.2f')}")
            if metrics.get('outperformance') is not None:
                print(f"Outperformance:        {safe_format(metrics.get('outperformance'))}")
    
    def generate_comprehensive_report(self, results_df: pd.DataFrame,
                                    benchmark_df: Optional[pd.DataFrame] = None,
                                    output_dir: str = "reports") -> Dict[str, str]:
        """
        Generate comprehensive QuantStats report suite.
        
        Args:
            results_df: Backtest results
            benchmark_df: Optional benchmark data
            output_dir: Output directory
            
        Returns:
            Dictionary with all generated file paths
        """
        print("\nGenerating QuantStats comprehensive reports...")
        
        # Prepare daily returns
        portfolio_returns, benchmark_returns = self.prepare_daily_returns(results_df, benchmark_df)
        
        # Generate reports
        report_files = {}
        
        # HTML reports
        report_files['html_report'] = self.generate_html_report(
            portfolio_returns, benchmark_returns,
            os.path.join(output_dir, "quantstats_full_report.html")
        )
        
        report_files['tearsheet'] = self.generate_tearsheet(
            portfolio_returns, benchmark_returns,
            os.path.join(output_dir, "quantstats_tearsheet.html")
        )
        
        # Performance plots
        plot_files = self.create_performance_plots(
            portfolio_returns, benchmark_returns,
            os.path.join(output_dir, "quantstats_plots")
        )
        report_files.update(plot_files)
        
        # Get metrics
        metrics = self.get_performance_metrics(portfolio_returns, benchmark_returns)
        # Skip verbose console metrics; reports already saved to files
        
        # Save metrics to CSV
        metrics_file = os.path.join(output_dir, "quantstats_metrics.csv")
        pd.Series(metrics).to_csv(metrics_file)
        report_files['metrics_csv'] = metrics_file
        
        print("\nQuantStats reports generated successfully!")
        print("All files saved in:", f"{output_dir}/")
        
        return report_files
