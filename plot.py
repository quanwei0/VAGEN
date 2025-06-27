import re
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional
from collections import defaultdict
import numpy as np

class WandbDataParser:
    """Parse training data from wandb sync output file"""
    
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.step_data = {}
        self.metrics_data = defaultdict(list)
        
    def parse_file(self) -> Dict[int, Dict[str, float]]:
        """Parse file and extract data for all steps"""
        print("Starting file parsing...")
        
        with open(self.file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Find all lines containing step data
        step_pattern = r'line: "step:(\d+) - (.+?)"'
        matches = re.findall(step_pattern, content)
        
        print(f"Found {len(matches)} steps of data")
        
        for step_num, metrics_line in matches:
            step_num = int(step_num)
            metrics = self._parse_metrics_line(metrics_line)
            self.step_data[step_num] = metrics
            
            # Add data to metrics_data for time series analysis
            for metric_name, value in metrics.items():
                self.metrics_data[metric_name].append({
                    'step': step_num,
                    'value': value
                })
        
        return self.step_data
    
    def _parse_metrics_line(self, line: str) -> Dict[str, float]:
        """Parse single line of metrics data"""
        metrics = {}
        
        # Use regex to extract metric_name:value pairs
        # Updated regex to handle complex metric names with parentheses and special characters
        pattern = r'([^:]+?):([0-9.-]+)(?:\s|$|-)'
        matches = re.findall(pattern, line)
        
        for metric_name, value in matches:
            # Clean up metric name
            metric_name = metric_name.strip()
            try:
                metrics[metric_name] = float(value)
            except ValueError:
                continue
                
        return metrics
    
    def get_available_metrics(self) -> List[str]:
        """Get all available metric names"""
        all_metrics = set()
        for step_data in self.step_data.values():
            all_metrics.update(step_data.keys())
        return sorted(list(all_metrics))
    
    def get_metric_over_time(self, metric_name: str) -> pd.DataFrame:
        """Get time series data for a specific metric"""
        if metric_name not in self.metrics_data:
            raise ValueError(f"Metric '{metric_name}' does not exist")
        
        df = pd.DataFrame(self.metrics_data[metric_name])
        return df.sort_values('step')
    
    def plot_metric(self, metric_name: str, save_path: Optional[str] = None, 
                   figsize: tuple = (10, 6)):
        """Plot time series curve for a single metric"""
        try:
            df = self.get_metric_over_time(metric_name)
        except ValueError as e:
            print(f"Error: {e}")
            return
        
        plt.figure(figsize=figsize)
        plt.plot(df['step'], df['value'], marker='o', linewidth=2, markersize=8)
        plt.title(f'{metric_name} over time', fontsize=14, fontweight='bold')
        plt.xlabel('Training Step', fontsize=12)
        plt.ylabel(metric_name, fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Add value annotations
        for i, row in df.iterrows():
            plt.annotate(f'{row["value"]:.3f}', 
                        (row['step'], row['value']),
                        textcoords="offset points", 
                        xytext=(0,10), 
                        ha='center', fontsize=9)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Chart saved to: {save_path}")
        
        plt.show()
    
    def plot_multiple_metrics(self, metric_names: List[str], 
                            save_path: Optional[str] = None,
                            figsize: tuple = (12, 8)):
        """Plot comparison chart for multiple metrics"""
        if not metric_names:
            print("Please provide at least one metric name")
            return
        
        # Check if all metrics exist
        available_metrics = self.get_available_metrics()
        invalid_metrics = [m for m in metric_names if m not in available_metrics]
        if invalid_metrics:
            print(f"The following metrics do not exist: {invalid_metrics}")
            return
        
        fig, axes = plt.subplots(len(metric_names), 1, figsize=figsize, 
                                sharex=True)
        if len(metric_names) == 1:
            axes = [axes]
        
        for i, metric_name in enumerate(metric_names):
            df = self.get_metric_over_time(metric_name)
            
            axes[i].plot(df['step'], df['value'], marker='o', 
                        linewidth=2, markersize=6)
            axes[i].set_ylabel(metric_name, fontsize=10)
            axes[i].grid(True, alpha=0.3)
            axes[i].set_title(f'{metric_name}', fontsize=11)
            
            # Add value annotations
            for _, row in df.iterrows():
                axes[i].annotate(f'{row["value"]:.3f}', 
                               (row['step'], row['value']),
                               textcoords="offset points", 
                               xytext=(0,5), 
                               ha='center', fontsize=8)
        
        axes[-1].set_xlabel('Training Step', fontsize=12)
        plt.suptitle('Multiple Metrics Comparison over Time', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Chart saved to: {save_path}")
        
        plt.show()
    
    def get_summary_stats(self) -> pd.DataFrame:
        """Get statistical summary for all metrics"""
        summary_data = []
        
        for metric_name in self.get_available_metrics():
            df = self.get_metric_over_time(metric_name)
            values = df['value'].values
            
            summary_data.append({
                'metric_name': metric_name,
                'min_value': np.min(values),
                'max_value': np.max(values),
                'mean_value': np.mean(values),
                'std_dev': np.std(values),
                'final_value': values[-1] if len(values) > 0 else None,
                'data_points': len(values)
            })
        
        return pd.DataFrame(summary_data)
    
    def search_metrics(self, keyword: str) -> List[str]:
        """Search for metric names containing the keyword"""
        available_metrics = self.get_available_metrics()
        return [metric for metric in available_metrics if keyword.lower() in metric.lower()]
    
    def print_summary(self):
        """Print data summary"""
        print("=" * 60)
        print("WANDB Data Parsing Summary")
        print("=" * 60)
        print(f"Total training steps: {len(self.step_data)}")
        print(f"Available metrics count: {len(self.get_available_metrics())}")
        print()
        
        print("Training steps list:")
        steps = sorted(self.step_data.keys())
        print(f"  {steps}")
        print()
        
        print("Metric categories:")
        metrics = self.get_available_metrics()
        categories = defaultdict(list)
        
        for metric in metrics:
            if '/' in metric:
                category = metric.split('/')[0]
                categories[category].append(metric)
            else:
                categories['others'].append(metric)
        
        for category, metric_list in categories.items():
            print(f"  {category}: {len(metric_list)} metrics")
        
        print("=" * 60)


if __name__ == "__main__":
    """Example usage"""
    # Initialize parser
    parser = WandbDataParser('/root/VAGEN/wandb/wandb_sync_output.txt')
    
    # Parse file
    step_data = parser.parse_file()
    
    # Print summary
    parser.print_summary()
    
    # Search for SokobanEnvConfig metrics
    print("\nSearching for SokobanEnvConfig success metrics:")
    sokoban_metrics = parser.search_metrics('SokobanEnvConfig')
    
    # Filter for only success metrics
    success_metrics = [m for m in sokoban_metrics if '/success/' in m]
    train_success = [m for m in success_metrics if 'train/success' in m]
    val_success = [m for m in success_metrics if 'val/success' in m or 'eval/success' in m]
    
    if train_success or val_success:
        print("\n" + "="*80)
        print("SOKOBAN SUCCESS RATE DATA")
        print("="*80)
        
        # Print train success
        for metric in train_success:
            print(f"\n📊 TRAIN SUCCESS: {metric}")
            try:
                df = parser.get_metric_over_time(metric)
                print(f"Total data points: {len(df)}")
                print(f"Value range: {df['value'].min():.3f} - {df['value'].max():.3f}")
                print(f"Final value: {df['value'].iloc[-1]:.3f}")
                print(f"Average value: {df['value'].mean():.3f}")
                print("\nStep-by-step data:")
                for _, row in df.iterrows():
                    print(f"  Step {int(row['step']):3d}: {row['value']:.3f}")
            except Exception as e:
                print(f"Error processing metric: {e}")
        
        # Print val success  
        for metric in val_success:
            print(f"\n📊 VAL SUCCESS: {metric}")
            try:
                df = parser.get_metric_over_time(metric)
                print(f"Total data points: {len(df)}")
                print(f"Value range: {df['value'].min():.3f} - {df['value'].max():.3f}")
                print(f"Final value: {df['value'].iloc[-1]:.3f}")
                print(f"Average value: {df['value'].mean():.3f}")
                print("\nStep-by-step data:")
                for _, row in df.iterrows():
                    print(f"  Step {int(row['step']):3d}: {row['value']:.3f}")
            except Exception as e:
                print(f"Error processing metric: {e}")
                
    else:
        print("No train/success or val/success SokobanEnvConfig metrics found!")
        print("Available SokobanEnvConfig metrics:")
        for metric in sokoban_metrics:
            print(f"  - {metric}")
