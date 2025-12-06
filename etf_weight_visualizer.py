"""
ETF Portfolio Weight Visualizer
Tracks and visualizes how ETF allocations change over time in the optimizer.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import cm
from typing import List
import os


class ETFWeightVisualizer:
    """
    Visualizer for ETF portfolio weights over time.
    """
    
    def __init__(self, portfolio_df: pd.DataFrame, reports_dir: str):
        """
        Initialize the weight visualizer.
        
        Args:
            portfolio_df: Portfolio results DataFrame with weight tracking
            reports_dir: Directory to save plots
        """
        self.portfolio_df = portfolio_df.copy()
        self.reports_dir = reports_dir
        
        # Ensure date column is datetime
        if 'date' in self.portfolio_df.columns:
            self.portfolio_df['date'] = pd.to_datetime(self.portfolio_df['date'])
            self.portfolio_df = self.portfolio_df.sort_values('date')
        
        # Create weight tracking if not exists (extract from rebalance days)
        self._create_weight_history()
        
        print("ETF Weight Visualizer initialized:")
        print(f"  Assets found: {len(self.etf_names)}")
        if self.has_cash:
            print(f"    - ETFs: {len(self.etf_names) - 1}")
            print("    - Cash option: Enabled")
        else:
            print(f"    - ETFs: {len(self.etf_names)}")
        print(f"  Date range: {self.portfolio_df['date'].min()} to {self.portfolio_df['date'].max()}")
        print(f"  Data points: {len(self.portfolio_df)}")
    
    def _create_weight_history(self):
        """Create weight history DataFrame from portfolio data."""
        # Check if we already have weight columns
        existing_weight_cols = [col for col in self.portfolio_df.columns if '_weight' in col.lower()]
        
        if existing_weight_cols:
            # Already have weight columns
            self.weight_columns = existing_weight_cols
            self.etf_names = [col.replace('_weight', '').replace('_WEIGHT', '') 
                             for col in self.weight_columns]
        else:
            # Need to extract from current_weights or other source
            # For now, create empty structure (will be populated by run_simple_etf)
            self.weight_columns = []
            self.etf_names = []
        
        self.has_cash = any('CASH' in name for name in self.etf_names)
    
    def plot_weight_evolution_stacked(self) -> str:
        """
        Create a stacked area chart showing weight evolution over time.
        
        Returns:
            Path to saved plot file
        """
        _, ax = plt.subplots(figsize=(16, 10))
        
        # Prepare data for stacked area
        dates = self.portfolio_df['date'].values
        weights_data = []
        
        # Separate CASH from ETFs for special coloring
        etfs_only = [col for col in self.weight_columns if 'CASH' not in col]
        
        for col in etfs_only:
            weights_data.append(self.portfolio_df[col].fillna(0).values)
        
        # Add CASH last (if present)
        if self.has_cash:
            cash_col = [col for col in self.weight_columns if 'CASH' in col][0]
            weights_data.append(self.portfolio_df[cash_col].fillna(0).values)
        
        # Create color scheme (ETFs + special color for CASH)
        n_etfs = len(etfs_only)
        colors = list(cm.tab20(np.linspace(0, 1, n_etfs)))  # noqa: E1101
        
        # Add gold/yellow color for CASH
        if self.has_cash:
            colors.append('#FFD700')  # Gold for cash
        
        # Prepare labels
        labels = [col.replace('_weight', '') for col in etfs_only]
        if self.has_cash:
            labels.append('[CASH]')
        
        ax.stackplot(dates, *weights_data, 
                    labels=labels,
                    colors=colors,
                    alpha=0.8)
        
        ax.set_xlabel('Date', fontsize=14, fontweight='bold')
        ax.set_ylabel('Weight Allocation', fontsize=14, fontweight='bold')
        ax.set_title('ETF Portfolio Weight Evolution (Stacked)', 
                     fontsize=16, fontweight='bold', pad=20)
        
        # Format y-axis as percentage
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.YearLocator())
        plt.xticks(rotation=45, ha='right')
        
        # Add legend
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), 
                 fontsize=10, framealpha=0.9)
        
        # Add grid
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_ylim([0, 1])
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.reports_dir, 'etf_weight_evolution_stacked.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  [OK] Stacked weight evolution plot saved: {plot_path}")
        return plot_path
    
    def plot_weight_evolution_lines(self) -> str:
        """
        Create a line chart showing individual ETF weights over time.
        
        Returns:
            Path to saved plot file
        """
        _, ax = plt.subplots(figsize=(16, 10))
        
        dates = self.portfolio_df['date'].values
        
        # Separate CASH for special coloring
        etfs_only = [(col, col.replace('_weight', '')) for col in self.weight_columns if 'CASH' not in col]
        n_etfs = len(etfs_only)
        colors = list(cm.tab20(np.linspace(0, 1, n_etfs)))  # noqa: E1101
        
        # Plot each ETF's weight
        for i, (col, etf_name) in enumerate(etfs_only):
            weights = self.portfolio_df[col].fillna(0).values
            ax.plot(dates, weights, label=etf_name, 
                   color=colors[i], linewidth=2, alpha=0.7)
        
        # Plot CASH with special styling
        if self.has_cash:
            cash_col = [col for col in self.weight_columns if 'CASH' in col][0]
            cash_weights = self.portfolio_df[cash_col].fillna(0).values
            ax.plot(dates, cash_weights, label='[CASH]', 
                   color='#FFD700', linewidth=3, alpha=0.9, linestyle='--')
        
        ax.set_xlabel('Date', fontsize=14, fontweight='bold')
        ax.set_ylabel('Weight Allocation', fontsize=14, fontweight='bold')
        ax.set_title('ETF Portfolio Weight Evolution (Individual Lines)', 
                     fontsize=16, fontweight='bold', pad=20)
        
        # Format y-axis as percentage
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.YearLocator())
        plt.xticks(rotation=45, ha='right')
        
        # Add legend
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), 
                 fontsize=10, framealpha=0.9, ncol=1)
        
        # Add grid
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_ylim([0, ax.get_ylim()[1]])
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.reports_dir, 'etf_weight_evolution_lines.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  [OK] Line weight evolution plot saved: {plot_path}")
        return plot_path
    
    def plot_weight_heatmap(self) -> str:
        """
        Create a heatmap showing ETF weights over time.
        
        Returns:
            Path to saved plot file
        """
        _, ax = plt.subplots(figsize=(16, 8))
        
        # Prepare data for heatmap (ETFs as rows, dates as columns)
        # Downsample to monthly for readability
        self.portfolio_df['year_month'] = self.portfolio_df['date'].dt.to_period('M')
        
        # Calculate average weight per month
        monthly_weights = []
        months = []
        
        for period in self.portfolio_df['year_month'].unique():
            period_data = self.portfolio_df[self.portfolio_df['year_month'] == period]
            monthly_weights.append([period_data[col].fillna(0).mean() for col in self.weight_columns])
            months.append(str(period))
        
        weights_matrix = np.array(monthly_weights).T
        
        # Create heatmap
        im = ax.imshow(weights_matrix, aspect='auto', cmap='YlOrRd', 
                      interpolation='nearest', vmin=0, vmax=0.7)
        
        # Set ticks
        ax.set_yticks(np.arange(len(self.etf_names)))
        ax.set_yticklabels(self.etf_names, fontsize=10)
        
        # Set x-ticks (show every 6 months)
        step = max(1, len(months) // 20)
        ax.set_xticks(np.arange(0, len(months), step))
        ax.set_xticklabels([months[i] for i in range(0, len(months), step)], 
                          rotation=45, ha='right', fontsize=9)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Weight Allocation', rotation=270, labelpad=20, 
                      fontsize=12, fontweight='bold')
        cbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
        
        # Labels and title
        ax.set_xlabel('Time Period (Month)', fontsize=14, fontweight='bold')
        ax.set_ylabel('ETF / Asset', fontsize=14, fontweight='bold')
        ax.set_title('ETF Portfolio Weight Heatmap (Monthly Average)', 
                     fontsize=16, fontweight='bold', pad=20)
        
        # Add grid
        ax.set_xticks(np.arange(len(months)) - 0.5, minor=True)
        ax.set_yticks(np.arange(len(self.etf_names)) - 0.5, minor=True)
        ax.grid(which='minor', color='white', linestyle='-', linewidth=0.5)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.reports_dir, 'etf_weight_heatmap.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  [OK] Weight heatmap saved: {plot_path}")
        return plot_path
    
    def generate_weight_statistics(self) -> pd.DataFrame:
        """
        Generate summary statistics for ETF weights.
        
        Returns:
            DataFrame with weight statistics per ETF
        """
        stats = []
        
        for col, etf_name in zip(self.weight_columns, self.etf_names):
            weights = self.portfolio_df[col].fillna(0)
            
            # Calculate statistics
            stats.append({
                'ETF': etf_name,
                'Mean_Weight': weights.mean(),
                'Median_Weight': weights.median(),
                'Max_Weight': weights.max(),
                'Min_Weight': weights.min(),
                'Std_Weight': weights.std(),
                'Days_Active': (weights > 0.01).sum(),  # Days with >1% allocation
                'Days_Dominant': (weights > 0.30).sum(),  # Days with >30% allocation
                'Pct_Time_Active': (weights > 0.01).sum() / len(weights),
                'Pct_Time_Dominant': (weights > 0.30).sum() / len(weights)
            })
        
        stats_df = pd.DataFrame(stats)
        stats_df = stats_df.sort_values('Mean_Weight', ascending=False)
        
        return stats_df
    
    def plot_weight_statistics(self, stats_df: pd.DataFrame) -> str:
        """
        Create visualization of weight statistics.
        
        Args:
            stats_df: DataFrame with weight statistics
            
        Returns:
            Path to saved plot file
        """
        _, axes = plt.subplots(2, 2, figsize=(18, 12))
        
        colors = cm.tab20(np.linspace(0, 1, len(stats_df)))  # noqa: E1101
        
        # Plot 1: Average weight per ETF
        ax1 = axes[0, 0]
        bars1 = ax1.bar(stats_df['ETF'], stats_df['Mean_Weight'], 
                       color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        ax1.set_xlabel('ETF', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Average Weight', fontsize=12, fontweight='bold')
        ax1.set_title('Average Weight Allocation per ETF', fontsize=14, fontweight='bold')
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
        ax1.grid(True, alpha=0.3, axis='y', linestyle='--')
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Add value labels
        for bar in bars1:
            height = bar.get_height()
            if height > 0.01:
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1%}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # Plot 2: Weight volatility (std)
        ax2 = axes[0, 1]
        _ = ax2.bar(stats_df['ETF'], stats_df['Std_Weight'], 
                       color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        ax2.set_xlabel('ETF', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Weight Standard Deviation', fontsize=12, fontweight='bold')
        ax2.set_title('Weight Volatility per ETF', fontsize=14, fontweight='bold')
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
        ax2.grid(True, alpha=0.3, axis='y', linestyle='--')
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Plot 3: Percentage of time active (>1%)
        ax3 = axes[1, 0]
        bars3 = ax3.bar(stats_df['ETF'], stats_df['Pct_Time_Active'] * 100, 
                       color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        ax3.set_xlabel('ETF', fontsize=12, fontweight='bold')
        ax3.set_ylabel('% of Time Active (Weight > 1%)', fontsize=12, fontweight='bold')
        ax3.set_title('ETF Activity Percentage', fontsize=14, fontweight='bold')
        ax3.set_ylim([0, 105])
        ax3.grid(True, alpha=0.3, axis='y', linestyle='--')
        plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Add value labels
        for bar in bars3:
            height = bar.get_height()
            if height > 2:
                ax3.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.0f}%', ha='center', va='bottom', fontsize=9)
        
        # Plot 4: Percentage of time dominant (>30%)
        ax4 = axes[1, 1]
        bars4 = ax4.bar(stats_df['ETF'], stats_df['Pct_Time_Dominant'] * 100, 
                       color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        ax4.set_xlabel('ETF', fontsize=12, fontweight='bold')
        ax4.set_ylabel('% of Time Dominant (Weight > 30%)', fontsize=12, fontweight='bold')
        ax4.set_title('ETF Dominance Percentage', fontsize=14, fontweight='bold')
        ax4.set_ylim([0, max(105, stats_df['Pct_Time_Dominant'].max() * 110)])
        ax4.grid(True, alpha=0.3, axis='y', linestyle='--')
        plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Add value labels
        for bar in bars4:
            height = bar.get_height()
            if height > 1:  # Only label if > 1%
                ax4.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.0f}%', ha='center', va='bottom', fontsize=9)
        
        plt.suptitle('ETF Portfolio Weight Statistics', 
                    fontsize=18, fontweight='bold', y=1.00)
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.reports_dir, 'etf_weight_statistics.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  [OK] Weight statistics plot saved: {plot_path}")
        return plot_path
    
    def plot_concentration_evolution(self) -> str:
        """
        Plot concentration parameter evolution (for proportional risk control).
        
        Returns:
            Path to saved plot file
        """
        if 'concentration' not in self.portfolio_df.columns:
            print("  [WARNING]  No concentration data available (proportional control disabled)")
            return None
        
        _, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
        
        dates = self.portfolio_df['date'].values
        
        # Plot 1: Concentration level over time
        concentration = self.portfolio_df['concentration'].ffill()
        ax1.plot(dates, concentration, color='purple', linewidth=2.5, alpha=0.8,
                label='Dynamic Concentration')
        ax1.axhline(y=0.5, color='gray', linestyle='--', linewidth=2, 
                   label='Base Level (0.5)', alpha=0.6)
        ax1.fill_between(dates, 0, concentration, alpha=0.3, color='purple')
        
        ax1.set_ylabel('Concentration Level', fontsize=12, fontweight='bold')
        ax1.set_title('ETF Portfolio Concentration Evolution (Proportional Risk Control)', 
                     fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.set_ylim([0, 1])
        
        # Add interpretation regions
        ax1.axhspan(0, 0.3, alpha=0.1, color='blue', label='Defensive')
        ax1.axhspan(0.3, 0.7, alpha=0.1, color='green')
        ax1.axhspan(0.7, 1.0, alpha=0.1, color='red', label='Aggressive')
        
        # Plot 2: Portfolio Sharpe ratio over time
        if 'portfolio_sharpe' in self.portfolio_df.columns:
            sharpe = self.portfolio_df['portfolio_sharpe'].ffill()
            ref_sharpe = self.portfolio_df.get('benchmark_sharpe', pd.Series([0]*len(self.portfolio_df))).ffill()
            
            ax2.plot(dates, sharpe, color='blue', linewidth=2.5, alpha=0.8,
                    label='Portfolio Sharpe')
            ax2.plot(dates, ref_sharpe, color='orange', linestyle='--', linewidth=2,
                    alpha=0.8, label='Benchmark Sharpe')
            ax2.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
            
            ax2.set_xlabel('Date', fontsize=12, fontweight='bold')
            ax2.set_ylabel('Sharpe Ratio', fontsize=12, fontweight='bold')
            ax2.set_title('Rolling Sharpe Ratio (Control Signal)', fontsize=14, fontweight='bold')
            ax2.legend(fontsize=10)
            ax2.grid(True, alpha=0.3, linestyle='--')
        
        # Format x-axis
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax2.xaxis.set_major_locator(mdates.YearLocator())
        plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.reports_dir, 'etf_concentration_evolution.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  [OK] Concentration evolution plot saved: {plot_path}")
        return plot_path
    
    def create_all_plots(self) -> List[str]:
        """
        Create all weight visualization plots.
        
        Returns:
            List of paths to saved plot files
        """
        print("\nGenerating ETF portfolio weight visualizations...")
        
        plot_files = []
        
        if not self.weight_columns:
            print("  [WARNING]  No weight columns found. Skipping weight visualizations.")
            return plot_files
        
        try:
            # Stacked area chart
            plot_files.append(self.plot_weight_evolution_stacked())
        except Exception as e:
            print(f"  [WARNING]  Stacked plot failed: {str(e)}")
        
        try:
            # Line chart
            plot_files.append(self.plot_weight_evolution_lines())
        except Exception as e:
            print(f"  [WARNING]  Line plot failed: {str(e)}")
        
        try:
            # Heatmap
            plot_files.append(self.plot_weight_heatmap())
        except Exception as e:
            print(f"  [WARNING]  Heatmap failed: {str(e)}")
        
        try:
            # Weight statistics
            stats_df = self.generate_weight_statistics()
            
            # Save statistics to Excel
            stats_path = os.path.join(self.reports_dir, 'etf_weight_statistics.xlsx')
            stats_df.to_excel(stats_path, index=False)
            print(f"  [OK] Weight statistics saved: {stats_path}")
            
            plot_files.append(self.plot_weight_statistics(stats_df))
        except Exception as e:
            print(f"  [WARNING]  Statistics plot failed: {str(e)}")
        
        try:
            # Concentration evolution (for proportional risk control)
            conc_plot = self.plot_concentration_evolution()
            if conc_plot:
                plot_files.append(conc_plot)
        except Exception as e:
            print(f"  [WARNING]  Concentration plot failed: {str(e)}")
        
        print(f"\n[OK] Generated {len(plot_files)} weight visualization plots")
        return plot_files

